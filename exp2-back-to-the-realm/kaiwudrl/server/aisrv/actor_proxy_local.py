#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file actor_proxy_local.py
# @brief
# @author kaiwu
# @date 2023-11-28


import multiprocessing
import os
import time
import numpy as np

# 按照需要导入
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.choose_deep_learning_frameworks import *
import datetime
import copy
from kaiwudrl.common.utils.common_func import TimeIt
from kaiwudrl.common.config.app_conf import AppConf

if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
    from kaiwudrl.common.pybind11.zmq_ops.zmq_ops import dump_arrays
from kaiwudrl.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from kaiwudrl.common.utils.common_func import (
    get_uuid,
    compress_data,
    get_mean_and_max,
)
from kaiwudrl.common.monitor.monitor_proxy_process import MonitorProxy
from kaiwudrl.common.config.algo_conf import AlgoConf
from kaiwudrl.server.common.load_model_common import LoadModelCommon

from kaiwudrl.common.algorithms.model_wrapper_common import (
    create_standard_model_wrapper,
    create_normal_model_wrapper,
)
from kaiwudrl.server.common.predict_common import PredictCommon
from kaiwudrl.server.common.actor_to_aisrv_response_common import (
    ActorToAisrvResponseCommon,
)
from kaiwudrl.common.ipc.zmq_util import ZmqServer, ZmqClient

# 按照需要加载pb的
if CONFIG.aisrv_actor_protocol == KaiwuDRLDefine.PROTOCOL_PROTOBUF:
    from kaiwudrl.common.protocol import aisrv_actor_req_resp_pb2


class ActorProxyLocal(multiprocessing.Process):
    def __init__(self, policy_name, index, actor_addr, context) -> None:
        super(ActorProxyLocal, self).__init__()

        self.policy_name = policy_name
        """
        支持业务自定义和从alloc获取的情况
        1. 默认端口
        2. alloc服务下发的IP和端口
        3. 从配置文件读取的IP和端口
        """
        # 获取ip
        self.actor_address = actor_addr[0]
        # 获取port
        self.actor_port = actor_addr[1]

        """
        aisrv <--> actor之间, actor是支持多个aisrv的, 故actor需要知道各个aisrv的client_id, 才能准确回包, 故这里采用uuid方式
        该值会透传给actor, 在actor采用zmq进行回包时, 带上该client_id参数
        """
        self.client_id = get_uuid()

        # 标志该index是多少, 主要用于看主和从关系
        self.index = index

        # slots
        self.slots = context.slots
        self.slot_group_name = f"{policy_name}_predict_{self.index}"
        self.slots.register_group(self.slot_group_name)

        # send msg queue
        self.msg_queue = multiprocessing.Queue(CONFIG.queue_size)

        # 统计数目
        self.send_to_actor_succ_cnt = 0
        self.send_to_actor_error_cnt = 0

        self.recv_from_actor_succ_cnt = 0
        self.recv_from_actor_error_cnt = 0

        # 采用压缩算法时, 压缩耗时, 解压缩耗时, 压缩大小
        self.max_compress_time = 0
        self.max_decompress_time = 0
        self.max_compress_size = 0
        self.actor_from_zmq_queue_cost_time_ms = 0
        self.actor_from_zmq_queue_size = 0

        self.current_sync_model_version_from_learner = 0

        # 本地预测分段耗时
        self.get_and_predict_cost_time_ms = 0
        self.send_cost_time_ms = 0

        # 设置最后处理时间
        self.last_predict_stat_time = 0
        self.last_load_last_new_model_time = 0

        # policy和model_wrapper对象的map, 为了支持多agent
        self.policy_model_wrapper_maps = {}

        # 设置公共的预测类, 便于每次预测时调用
        self.predict_common_object = None

        # 设置公共的加载model文件类, 便于每次加载时使用
        self.load_model_common_object = None

        # 由外界传入model_file_sync_wrapper对象
        if CONFIG.predict_local_or_remote == KaiwuDRLDefine.PREDICT_AS_LOCAL:
            if CONFIG.need_to_start_learner:
                self.model_file_sync_wrapper = context.model_file_sync_wrapper

                if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                    self.model_file_sync_wrapper_lock = context.model_file_sync_wrapper_lock

    # 需要区分是哪个agent发送的请求
    def put_predict_data(self, slot_id, agent_id, message_id, model_version, agent_main_id, predict_data) -> None:
        if not predict_data or self.msg_queue.full():
            return False

        """
        因为on-policy情况下需要修改model_version, 如果这里的compose_id是tuple则转换为list修改耗费性能, 故设计为list
        """
        self.msg_queue.put(
            (
                [slot_id, agent_id, message_id, model_version, agent_main_id],
                predict_data,
            )
        )
        return True

    # 同一个对局的两个agent数据同时发送过来，需要actor同时处理
    def put_predict_data_v2(
        self,
        slot_id,
        agent_id_0,
        agent_id_1,
        message_id_0,
        message_id_1,
        predict_data_0,
        predict_data_1,
    ) -> None:
        if not predict_data_0 or self.msg_queue.full():
            return False

        self.msg_queue.put(
            [
                ((slot_id, agent_id_0, message_id_0), predict_data_0),
                ((slot_id, agent_id_1, message_id_1), predict_data_1),
            ]
        )
        return True

    def get_predict_data(self, slot_id):

        input_pipe = self.slots.get_input_pipe(self.slot_group_name, slot_id)

        # 设置queue的超时时间
        if input_pipe.poll(CONFIG.queue_wait_timeout):
            return input_pipe.recv()

        return None

    # input_tensors作为TensorFlow 预测和训练的入口
    def input_tensors(self):
        # 等待输入
        def wait_for_inputs(queue_size):
            with tf.control_dependencies(enqueue_ops):
                tf.no_op()

            self.logger.debug(f"predict current predict_input_queue size is {input_queue.size()}")
            return input_queue.size()

        # actor上zmq的端口需要和aisrv上的zmq端口一致
        receiver = ZMQPullSocket(
            f"tcp://{CONFIG.ip_address}:{CONFIG.zmq_server_op_port}",
            self.dtypes,
            hwm=CONFIG.zmq_ops_hwm,
        )
        self.logger.info(f"predict zmq-ops server start at {CONFIG.ip_address}:{CONFIG.zmq_server_op_port}")

        enqueue_tensors = [
            tensor if not CONFIG.use_rnn else tf.expand_dims(tensor, axis=1) for tensor in receiver.pull()
        ]
        input_shapes = [
            (shape[1:] if not CONFIG.use_rnn else tf.TensorShape([1] + shape[1:].as_list())) for shape in self.shapes
        ]

        # 利用TensorFlow的FIFOQueue
        input_queue = tf.queue.FIFOQueue(
            CONFIG.predict_input_queue_size,
            self.dtypes,
            input_shapes,
            name="predict_input_queue",
        )

        enqueue_op = input_queue.enqueue_many(enqueue_tensors)
        enqueue_ops = [enqueue_op] * CONFIG.predict_input_threads

        # 创建了CONFIG.predict_input_threads线程, 每个线程里运行的是enqueue_op操作
        tf.compat.v1.train.add_queue_runner(tf.compat.v1.train.QueueRunner(input_queue, enqueue_ops=enqueue_ops))

        qsize_tensor = input_queue.size()

        # TensorFlow while loop处理
        self.dequeue_size_tensor = tf.while_loop(
            lambda queue_size: tf.less(queue_size, CONFIG.predict_input_queue_deq_min),
            wait_for_inputs,
            [qsize_tensor],
        )

        self.dequeue_tensors = input_queue.dequeue_many(self.dequeue_size_tensor)

        return dict(zip(self.names, self.dequeue_tensors))

    def predict_hooks(self):
        return []

    def standard_serialize_buffer_data(self):
        """
        针对不同的数据类型, 进行序列化
        消息的数据格式:
        (actor_id, slot_id, agent_id) | data
        1. pickle, 可以直接组装msg
        2. protobuf, 需要对KaiwuServerRequest类赋值
        """
        # 序列化 pre_req(业务State定义的类里的key的顺序) + client_id + compose_id(agent_id, slot_id)

        assert self.cur_buf_size == 1
        # self.logger.info(f'predict cur_buf_size is:{self.cur_buf_size}', g_not_server_label)

        # 采用pickle序列化
        if CONFIG.aisrv_actor_protocol == KaiwuDRLDefine.PROTOCOL_PICKLE:

            msg = {
                "data": self.buffer_data,
                "client_id": self.client_id_buf[: self.cur_buf_size],
                "compose_id": self.compose_id_buf[: self.cur_buf_size],
            }

        # 采用protobuf序列化
        elif CONFIG.aisrv_actor_protocol == KaiwuDRLDefine.PROTOCOL_PROTOBUF:

            kaiwu_server_request = aisrv_actor_req_resp_pb2.AisrvActorRequest()
            kaiwu_server_request.client_id = self.client_id_buf[: self.cur_buf_size][0]
            kaiwu_server_request.sample_size = 1
            # 形如[[0, 1, 2]]
            kaiwu_server_request.compose_id.extend(self.compose_id_buf[: self.cur_buf_size])

            # 接入数据前处理过程
            input_array = np.concatenate(
                [
                    self.buffer_data["observation"][0],
                    self.buffer_data["lstm_cell"][0],
                    self.buffer_data["lstm_hidden"][0],
                ],
                axis=1,
            ).reshape(-1)
            kaiwu_server_request.feature.extend(input_array.tolist())

            msg = kaiwu_server_request

        # 在单个进程类不需要进行压缩和解压缩
        return msg

    def serialize_buffer_data(self):
        """
        针对不同的数据类型, 进行序列化
        消息的数据格式:
        (actor_id, slot_id, agent_id) | data
        1. pickle, 可以直接组装msg
        2. protobuf, 需要对KaiwuServerRequest类赋值
        """

        # 序列化 pre_req(业务State定义的类里的key的顺序) + client_id + compose_id(agent_id, slot_id)

        assert self.cur_buf_size == 1
        # self.logger.info(f'predict cur_buf_size is:{self.cur_buf_size}', g_not_server_label)

        # 采用pickle序列化
        if CONFIG.aisrv_actor_protocol == KaiwuDRLDefine.PROTOCOL_PICKLE:
            """
            如果是标准化直接按照整个buff传输, 如果是非标准化按照不同的key传输
            """
            if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
                pred_req = [self.buffer_data[key][: self.cur_buf_size] for key in self.state_keys]

            elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
                pred_req = self.buffer_data

            else:

                pass

            msg = {
                "data": pred_req,
                "client_id": self.client_id_buf[: self.cur_buf_size],
                "compose_id": self.compose_id_buf[: self.cur_buf_size],
            }

        # 采用protobuf序列化
        elif CONFIG.aisrv_actor_protocol == KaiwuDRLDefine.PROTOCOL_PROTOBUF:

            kaiwu_server_request = aisrv_actor_req_resp_pb2.AisrvActorRequest()
            kaiwu_server_request.client_id = self.client_id_buf[: self.cur_buf_size][0]
            kaiwu_server_request.sample_size = 1
            # 形如[[0, 1, 2]]
            kaiwu_server_request.compose_id.extend(self.compose_id_buf[: self.cur_buf_size])

            # 接入数据前处理过程
            input_array = np.concatenate(
                [
                    self.buffer_data["observation"][0],
                    self.buffer_data["lstm_cell"][0],
                    self.buffer_data["lstm_hidden"][0],
                ],
                axis=1,
            ).reshape(-1)
            kaiwu_server_request.feature.extend(input_array.tolist())

            msg = kaiwu_server_request

        # 在单个进程类不需要进行压缩和解压缩
        return msg

    def compress_request_data(self, msg):

        # 增加lz4的压缩
        with TimeIt() as ti:
            compress_msg = compress_data(msg)

        # 压缩耗时和压缩包大小
        if self.max_compress_time < ti.interval:
            self.max_compress_time = ti.interval

        compress_msg_len = len(compress_msg)
        if self.max_compress_size < compress_msg_len:
            self.max_compress_size = compress_msg_len

        if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
            return dump_arrays(compress_msg)

        return compress_msg

    def predict_stat_reset(self):
        self.predict_common_object.set_actor_batch_predict_cost_time_ms(0)
        self.actor_from_zmq_queue_cost_time_ms = 0
        self.actor_from_zmq_queue_size = 0
        if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
            self.predict_common_object.set_actor_load_last_model_cost_ms(0)
        else:
            self.load_model_common_object.set_actor_load_last_model_cost_ms(0)

        self.max_decompress_time = 0
        self.max_compress_size = 0
        self.max_compress_time = 0
        self.send_to_actor_succ_cnt = 0
        self.send_to_actor_error_cnt = 0
        self.actor_to_aisrv_response_common_object.set_recv_from_actor_succ_cnt(0)
        self.recv_from_actor_error_cnt = 0
        self.get_and_predict_cost_time_ms = 0
        self.send_cost_time_ms = 0

    def predict_stat(self):
        """
        这里增加predict的统计项
        """

        predict_count = self.predict_common_object.predict_stat()

        if int(CONFIG.use_prometheus) and not self.index and CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE:

            # 注意msg_queue.qsize()可能出现异常报错, 故采用try-catch模式
            try:
                msg_queue_size = self.msg_queue.qsize()
            except NotImplementedError:
                msg_queue_size = 0
            except Exception as e:
                msg_queue_size = 0

            if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
                if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                    actor_load_last_model_error_cnt = (
                        self.load_model_common_object.get_actor_load_last_model_error_cnt()
                    )
                    actor_load_last_model_succ_cnt = self.load_model_common_object.get_actor_load_last_model_succ_cnt()
                    actor_load_last_model_cost_ms = self.load_model_common_object.get_actor_load_last_model_cost_ms()
                else:
                    actor_load_last_model_error_cnt = self.predict_common_object.get_actor_load_last_model_error_cnt()
                    actor_load_last_model_succ_cnt = self.predict_common_object.get_actor_load_last_model_succ_cnt()
                    actor_load_last_model_cost_ms = self.predict_common_object.get_actor_load_last_model_cost_ms()
            else:
                actor_load_last_model_error_cnt = self.load_model_common_object.get_actor_load_last_model_error_cnt()
                actor_load_last_model_succ_cnt = self.load_model_common_object.get_actor_load_last_model_succ_cnt()
                actor_load_last_model_cost_ms = self.load_model_common_object.get_actor_load_last_model_cost_ms()

            actor_batch_predict_cost_time_ms = self.predict_common_object.get_actor_batch_predict_cost_time_ms()
            recv_from_actor_succ_cnt = self.actor_to_aisrv_response_common_object.get_recv_from_actor_succ_cnt()

            monitor_data = {
                KaiwuDRLDefine.MONITOR_ACTOR_PREDICT_SUCC_CNT: predict_count,
                KaiwuDRLDefine.MONITOR_ACTOR_FROM_ZMQ_QUEUE_SIZE: self.actor_from_zmq_queue_size,
                KaiwuDRLDefine.MONITOR_ACTOR_FROM_ZMQ_QUEUE_COST_TIME_MS: self.actor_from_zmq_queue_cost_time_ms,
                KaiwuDRLDefine.MONITOR_ACTOR_BATCH_PREDICT_COST_TIME_MS: actor_batch_predict_cost_time_ms,
                KaiwuDRLDefine.ACTOR_LOAD_LAST_MODEL_COST_MS: actor_load_last_model_cost_ms,
                KaiwuDRLDefine.ACTOR_LOAD_LAST_MODEL_SUCC_CNT: actor_load_last_model_succ_cnt,
                KaiwuDRLDefine.ACTOR_LOAD_LAST_MODEL_ERROR_CNT: actor_load_last_model_error_cnt,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_DECOMPRESS_TIME: self.max_decompress_time,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_TIME: self.max_compress_time,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_SIZE: self.max_compress_size,
                KaiwuDRLDefine.MONITOR_AISRV_ACTOR_PROXY_QUEUE_LEN: msg_queue_size,
                KaiwuDRLDefine.MONITOR_AISRV_SENDTO_ACTOR_SUCC_CNT: self.send_to_actor_succ_cnt,
                KaiwuDRLDefine.MONITOR_AISRV_SENDTO_ACTOR_ERROR_CNT: self.send_to_actor_error_cnt,
                KaiwuDRLDefine.MONITOR_AISRV_RECVFROM_ACTOR_SUCC_CNT: recv_from_actor_succ_cnt,
                KaiwuDRLDefine.MONITOR_AISRV_RECVFROM_ACTOR_ERROR_CNT: self.recv_from_actor_error_cnt,
                KaiwuDRLDefine.MONITOR_ACTOR_GET_AND_PREDICT_COST_MS: self.get_and_predict_cost_time_ms,
                KaiwuDRLDefine.MONITOR_ACTOR_SENDTO_AISRV_BATCH_COST_TIME_MS: self.send_cost_time_ms,
            }

            # 针对aisrv发出去的请求, 有响应包的场景, 只是计算最大值和平均值时延
            if int(CONFIG.use_prometheus):
                mean_value, max_value = get_mean_and_max(self.time_cost_map.values())

                monitor_data[KaiwuDRLDefine.MONITOR_AISRV_ACTOR_MEAN_TIME_COST] = mean_value
                monitor_data[KaiwuDRLDefine.MONITOR_AISRV_ACTOR_MAX_TIME_COST] = max_value

            self.time_cost_map.clear()

            # 针对aisrv发出去的请求, 没有响应包的场景
            timeout_cnt = 0
            for key in list(self.timeout_map.keys()):
                value = self.timeout_map.get(key)

                # 计算下来是s为单位
                time_dela = (int(round(time.time() * 1000)) - value) / 1000

                if time_dela > CONFIG.aisrv_actor_timeout_second_threshold:
                    timeout_cnt += 1
                    self.logger.error(
                        f"predict message id {key} timeout after {time_dela} seconds",
                        g_not_server_label,
                    )

                    del self.timeout_map[key]
                    continue

            if int(CONFIG.use_prometheus):
                monitor_data[
                    f"{KaiwuDRLDefine.MONITOR_AISRV_ACTOR_TIMEOUT_GT}{CONFIG.aisrv_actor_timeout_second_threshold}"
                ] = timeout_cnt

            # on-policy情况下actor的主进程进行上报下面指标操作
            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                monitor_data[
                    KaiwuDRLDefine.ON_POLICY_PULL_FROM_MODELPOOL_ERROR_CNT
                ] = self.on_policy_pull_from_modelpool_error_cnt
                monitor_data[
                    KaiwuDRLDefine.ON_POLICY_PULL_FROM_MODELPOOL_SUCCESS_CNT
                ] = self.on_policy_pull_from_modelpool_success_cnt
                monitor_data[
                    KaiwuDRLDefine.ON_POLICY_ACTOR_CHANGE_MODEL_VERSION_ERROR_COUNT
                ] = self.actor_change_model_version_error_count
                monitor_data[
                    KaiwuDRLDefine.ON_POLICY_ACTOR_CHANGE_MODEL_VERSION_SUCCESS_COUNT
                ] = self.actor_change_model_version_success_count

            self.monitor_proxy.put_data({self.current_pid: monitor_data})

        # 指标复原, 计算的是周期性的上报指标
        self.predict_stat_reset()

        self.logger.info(f"predict now predict count is {predict_count}")

    def before_run(self) -> None:

        # 日志处理, 需要放在before run里进行初始化, 否则容易出现卡死
        self.logger = KaiwuLogger()
        self.current_pid = os.getpid()
        self.logger.setLoggerFormat(
            (
                f"/{CONFIG.svr_name}/predict_pid{self.current_pid}_log_"
                f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log"
            ),
        )

        # 填充client_id
        self.client_id_buf = np.empty((CONFIG.proxy_batch_size * 2), np.int32)
        self.client_id_buf.fill(self.client_id)

        # buff 的处理, 填充了COMPOSE_ID(agent_id, slot_id, message_id, model_version, agent_main_id), 由于需要存储不同的字段类型故采用list
        self.compose_id_buf = [None] * CONFIG.proxy_batch_size * 2

        """
        如果是标准化直接按照整个buff传输, 如果是非标准化按照不同的key传输
        """
        if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
            self.state_space = AppConf[CONFIG.app].policies[self.policy_name].state.state_space()
            self.state_keys = self.state_space.keys()
            self.buffer_data = {
                key: np.zeros(
                    (CONFIG.proxy_batch_size * 2,) + self.state_space[key].shape,
                    self.state_space[key].dtype,
                )
                for key in self.state_keys
            }

        elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
            self.buffer_data = {}

        else:

            pass

        self.cur_buf_size = 0

        """
        下面的操作是做下原本在actor上类似的操作, 移植过来
        """
        # 加载配置文件conf/algo_conf.json
        AlgoConf.load_conf(CONFIG.algo_conf)

        # 加载配置文件conf/app_conf.json
        AppConf.load_conf(CONFIG.app_conf)

        # 访问普罗米修斯的类, 只有第0个predict进程需要启动, 故需要提前来设置下self.monitor_proxy变量
        self.monitor_proxy = None
        if int(CONFIG.use_prometheus) and not self.index:
            self.monitor_proxy = MonitorProxy()
            self.monitor_proxy.start()

        # policy_name 主要是和conf/app_conf.json设置一致
        self.policy_conf = AppConf[CONFIG.app].policies

        # model_wrapper
        if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
            create_normal_model_wrapper(
                self.policy_conf,
                self.policy_model_wrapper_maps,
                None,
                self.logger,
            )

        elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
            # 无论是remote, local, none这里都需要执行下面操作
            create_standard_model_wrapper(
                self.policy_conf,
                self.policy_model_wrapper_maps,
                None,
                self.logger,
                self.monitor_proxy,
            )

        else:

            pass

        # 注册定时器任务
        # set_schedule_event(
        #    CONFIG.prometheus_stat_per_minutes, self.predict_stat)

        # 设置公共的加载文件类, 便于每次加载文件时调用
        self.load_model_common_object = LoadModelCommon(self.logger)
        self.load_model_common_object.set_model_file_sync_wrapper(self.model_file_sync_wrapper)
        self.load_model_common_object.set_policy_model_wrapper_maps(self.policy_model_wrapper_maps)

        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_TRAIN:
            # on-policy条件下是在actor同步到model文件后开始加载, 其他条件下是周期性加载
            if CONFIG.algorithm_on_policy_or_off_policy != KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                pass

        # 如果是在eval模式下下则执行第一次加载
        elif CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EXAM:
            if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
                self.load_model_common_object.load_last_new_model(self.policy_name)

            elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:

                # 单机单进程的版本是在aisrv上预测这里不需要
                if CONFIG.wrapper_type != KaiwuDRLDefine.WRAPPER_LOCAL:
                    self.load_model_common_object.standard_load_last_new_model_by_framework(self.policy_name)

            else:
                pass

        else:
            pass

        # 预先加载模型文件模式
        if int(CONFIG.preload_model):
            self.load_model_common_object.preload_model_file(self.policy_model_wrapper_maps)

        """
        用于做超时控制的, key为aisrv --> actor的message_id, value为发送时间
        1. 发送时, 将message_id和发送时间放在map里
        2. 当响应包回来, 则当前时间 - 发送时间, 即耗时
        3. 如果在一定时间里没有响应包回来, 则开始删除map里的key, 并且记录ERROR日志
        """
        self.timeout_map = {}
        self.time_cost_map = {}

        # 进程空转了N次就主动让出CPU, 避免CPU空转100%
        self.process_run_idle_count = 0

        # 设置公共的预测类, 便于每次预测时调用
        self.predict_common_object = PredictCommon(self.policy_name, self.monitor_proxy, self.logger)
        self.predict_common_object.set_policy_model_wrapper_maps(self.policy_model_wrapper_maps)
        self.predict_common_object.set_model_file_sync_wrapper(self.model_file_sync_wrapper)
        self.predict_common_object.set_policy_conf(self.policy_conf)
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            self.predict_common_object.set_current_sync_model_version_from_learner(
                self.current_sync_model_version_from_learner
            )

        # 设置公共的aisrv/actor朝aisrv回包的处理类, 便于每次处理回包时调用
        self.actor_to_aisrv_response_common_object = ActorToAisrvResponseCommon(self.logger)
        self.actor_to_aisrv_response_common_object.set_slots(self.slots)
        self.actor_to_aisrv_response_common_object.set_slot_group_name(self.slot_group_name)
        self.actor_to_aisrv_response_common_object.set_zmq_server_ip(self.get_zmq_server_ip())

        """
        model_file_sync_wrapper, actor和learner之间的Model文件同步, 采用单独的进程
        如果是on-policy算法则需要保存下来learner同步过来最新的model文件ID, 如果是off-policy则不需要
        为了编程方便, 都统一设置下
        """
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            self.model_file_sync_wrapper.ckpt_sync_warper.make_model_dirs(self.logger)

            """
            注意这里的zmq端口是不能出现重复的, 如果出现重复了会导致在同一个容器里无法启动问题, 故引入了self.index变量, 每个预测进程启动1个zmq_server对象
            由于是不同的policy下都需要调用故需要区分policy
            """
            sorted_items = sorted(self.policy_model_wrapper_maps.items(), key=lambda item: item[0])
            sorted_keys = [key for key, value in sorted_items]
            key_index = sorted_keys.index(self.policy_name)
            zmq_server_port = int(CONFIG.zmq_server_port) + (self.index + 1) * 100 + key_index

            self.zmq_server = ZmqServer(CONFIG.aisrv_ip_address, zmq_server_port)
            self.zmq_server.bind()
            self.logger.info(
                f"predict zmq server on-policy bind at {CONFIG.aisrv_ip_address}:{zmq_server_port} for learner"
            )

            # 下面是统计告警指标
            self.on_policy_pull_from_modelpool_error_cnt = 0
            self.on_policy_pull_from_modelpool_success_cnt = 0
            self.actor_change_model_version_error_count = 0
            self.actor_change_model_version_success_count = 0

        # 在before run最后打印启动成功日志
        self.logger.info(
            (
                f"predict policy_name: {self.policy_name}, start success at pid {self.current_pid}, "
                f"on-policy/off-policy is {CONFIG.algorithm_on_policy_or_off_policy}, "
                f"actor_receive_cost_time_ms: {CONFIG.actor_receive_cost_time_ms}, "
                f"predict_batch_size: {CONFIG.predict_batch_size}"
            ),
            g_not_server_label,
        )

        return True

    def get_predict_request_data_by_direct(self):
        """
        aisrv需要预测的数据, 采用队列形式返回
        """
        with TimeIt() as ti:
            msgs = []

            # 按照时间间隔和批处理大小收包
            start_time = time.time()
            while len(msgs) < int(CONFIG.predict_batch_size):
                if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
                    msg = self.get_data_from_predict_data_queue()
                elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
                    msg = self.standard_get_data_from_predict_data_queue()
                else:
                    msg = None

                if msg:
                    msgs.append(copy.deepcopy(msg))
                    self.send_to_actor_succ_cnt += 1

                # 收包超时时强制退出, 平滑处理
                if (time.time() - start_time) * 1000 > int(CONFIG.actor_receive_cost_time_ms):
                    break

        msgs_length = len(msgs)
        if not msgs_length:
            return msgs

        # 获取采集周期里的最大值
        if self.actor_from_zmq_queue_size < msgs_length:
            self.actor_from_zmq_queue_size = msgs_length

        if self.actor_from_zmq_queue_cost_time_ms < ti.interval * 1000:
            self.actor_from_zmq_queue_cost_time_ms = ti.interval * 1000

        return msgs

    def standard_get_data_from_predict_data_queue(self):
        """
        这里的分2种情况:
        1. 如果是on-polciy的, 注意需要采用非阻塞的, 否则阻塞了predict主流程
        2. 如果是非on-polciy的, 注意采用阻塞的, 性能要好于非阻塞的, 但是对于battlesrv数量远远小于了batch_size则需要设置超时时间
        """
        tmp_data = None
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            try:
                tmp_data = self.msg_queue.get_nowait()
            except Exception as e:
                pass
        else:
            tmp_data = self.msg_queue.get()

        if not tmp_data:
            return None

        if isinstance(tmp_data, tuple):
            if CONFIG.aisrv_actor_protocol == KaiwuDRLDefine.PROTOCOL_PICKLE:
                compose_id, data = tmp_data

                self.compose_id_buf[self.cur_buf_size] = compose_id

                # 所有的预测数据作为一个整体
                self.buffer_data = data

                self.cur_buf_size += 1

                is_predict_request = data.get(KaiwuDRLDefine.MESSAGE_TYPE) == KaiwuDRLDefine.MESSAGE_PREDICT

                """
                注意:
                1. 获取当前时间放入timeout_map, 第一帧耗时比较大, 不做统计
                2. slot_id, agent_id, message_id作为存放时延的key, 不能加入model_version, 该值可能被actor返回的值修改掉
                3. 如果是管理流请求不能放入, 因为没有回包
                """
                # 获取当前时间放入timeout_map, 第一帧耗时比较大, 不做统计
                [
                    slot_id,
                    agent_id,
                    message_id,
                    model_version,
                    agent_main_id,
                ] = compose_id

                if message_id != 1 and is_predict_request:
                    self.timeout_map[(slot_id, agent_id, message_id)] = int(round(time.time() * 1000))

                if CONFIG.distributed_tracing:
                    self.logger.info(
                        (
                            f"predict distributed_tracing compose_id {compose_id} "
                            f"will send to actor {self.get_zmq_server_ip()}"
                        ),
                        g_not_server_label,
                    )

                msg = self.standard_serialize_buffer_data()

                self.cur_buf_size = 0

            return msg

    def get_zmq_server_ip(self):
        """
        返回zmq server的IP和端口
        """

        if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ:
            return f"{self.actor_address}:{CONFIG.zmq_server_port}"
        elif CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
            return f"{self.actor_address}:{CONFIG.zmq_server_op_port}"
        else:
            return None

    def get_data_from_predict_data_queue(self):
        """
        这里的分2种情况:
        1. 如果是on-polciy的, 注意需要采用非阻塞的, 否则阻塞了predict主流程
        2. 如果是非on-polciy的, 注意采用阻塞的, 性能要好于非阻塞的, 但是对于battlesrv数量远远小于了batch_size则需要设置超时时间
        """
        tmp_data = None
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            try:
                tmp_data = self.msg_queue.get_nowait()
            except Exception as e:
                pass
        else:
            tmp_data = self.msg_queue.get()

        if not tmp_data:
            return None

        # 单帧单agent处理逻辑
        if isinstance(tmp_data, tuple):

            if CONFIG.aisrv_actor_protocol == KaiwuDRLDefine.PROTOCOL_PROTOBUF:

                compose_id, data = tmp_data

                [
                    slot_id,
                    agent_id,
                    message_id,
                    model_version,
                    agent_main_id,
                ] = compose_id
                if message_id != 1:
                    self.timeout_map[(slot_id, agent_id, message_id)] = int(round(time.time() * 1000))

                kaiwu_server_request = aisrv_actor_req_resp_pb2.AisrvActorRequest()
                kaiwu_server_request.client_id = self.client_id_buf[0]
                kaiwu_server_request.sample_size = 1
                # 形如[[0, 1, 2]]
                kaiwu_server_request.compose_id.extend(np.asarray(compose_id).astype(np.int32).flatten().tolist())

                # 接入数据前处理过程
                kaiwu_server_request.observation.extend(data["observation"].flatten().tolist())
                kaiwu_server_request.lstm_hidden.extend(data["lstm_hidden"].flatten().tolist())
                kaiwu_server_request.lstm_cell.extend(data["lstm_cell"].flatten().tolist())
                kaiwu_server_request.legal_action.extend(data["legal_action"].flatten().tolist())
                kaiwu_server_request.sub_action_mask.extend(data["sub_action_mask"].flatten().tolist())

                msg = kaiwu_server_request

                msg = self.compress_request_data(msg)

            elif CONFIG.aisrv_actor_protocol == KaiwuDRLDefine.PROTOCOL_PICKLE:

                compose_id, data = tmp_data

                self.compose_id_buf[self.cur_buf_size] = compose_id

                """
                如果是标准化直接按照整个buff传输, 如果是非标准化按照不同的key传输
                """
                if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
                    for key in self.state_keys:
                        self.buffer_data[key][self.cur_buf_size] = data[key]

                elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
                    self.buffer_data = data

                else:

                    pass

                self.cur_buf_size += 1

                """
                注意:
                1. 获取当前时间放入timeout_map, 第一帧耗时比较大, 不做统计
                2. slot_id, agent_id, message_id作为存放时延的key, 不能加入model_version, 该值可能被actor返回的值修改掉
                """
                # 获取当前时间放入timeout_map, 第一帧耗时比较大, 不做统计
                [
                    slot_id,
                    agent_id,
                    message_id,
                    model_version,
                    agent_main_id,
                ] = compose_id
                if message_id != 1:
                    self.timeout_map[(slot_id, agent_id, message_id)] = int(round(time.time() * 1000))

                if CONFIG.distributed_tracing:
                    self.logger.info(
                        (
                            f"predict distributed_tracing compose_id {compose_id} "
                            f"will send to actor {self.get_zmq_server_ip()}"
                        ),
                        g_not_server_label,
                    )

                msg = self.serialize_buffer_data()

                self.cur_buf_size = 0

            # 未来扩展
            else:
                pass

        else:
            compose_id_0, data_0 = tmp_data[0]
            compose_id_1, data_1 = tmp_data[1]

            [slot_id, agent_id, message_id, model_version, agent_main_id] = compose_id_0
            if message_id != 1:
                self.timeout_map[(slot_id, agent_id, message_id)] = int(round(time.time() * 1000))

            kaiwu_server_request = aisrv_actor_req_resp_pb2.AisrvActorRequest()
            kaiwu_server_request.client_id = self.client_id_buf[0]
            kaiwu_server_request.sample_size = 2
            # 形如[[0, 1, 2]]

            kaiwu_server_request.compose_id.extend(
                np.asarray(compose_id_0).astype(np.int32).flatten().tolist()
                + np.asarray(compose_id_1).astype(np.int32).flatten().tolist()
            )

            # 接入数据前处理过程
            inputs = []
            for data in (data_0, data_1):
                tmp = np.concatenate(
                    [
                        np.array(data["observation"]),
                        np.array(data["lstm_cell"]),
                        np.array(data["lstm_hidden"]),
                    ],
                    axis=1,
                ).reshape(-1)
                inputs.append(tmp)
            input_array = np.concatenate(inputs)

            kaiwu_server_request.feature.extend(input_array.tolist())
            msg = kaiwu_server_request

        return msg

    # 周期性的操作
    def periodic_operations(self):
        # 记录发送给actor成功失败数目, 包括发出去和收回来的请求
        # schedule.run_pending()

        now = time.time()
        if now - self.last_predict_stat_time > int(CONFIG.prometheus_stat_per_minutes) * 60:
            self.predict_stat()
            self.last_predict_stat_time = now

        # 周期性加载model文件
        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_TRAIN:
            # on-policy条件下是在actor同步到model文件后开始加载, 其他条件下是周期性加载
            if CONFIG.algorithm_on_policy_or_off_policy != KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                """
                由于支持多个agent的加载, 故标准化里模型文件加载的操作就需要使用者去调用, 否则需要配置比较复杂的任务, KaiwuDRL才能执行加载, 比如:
                1. agent1加载最新模型
                2. agent2加载次新模型
                3. ......
                """
                if now - self.last_load_last_new_model_time > int(CONFIG.model_file_load_per_minutes) * 60:
                    if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
                        self.load_model_common_object.load_last_new_model(
                            self.policy_name,
                        )
                    else:
                        pass

                    self.last_load_last_new_model_time = now

    def handle_message_timeouts(self, compose_id_indexs):
        """
        处理响应回包里超时情况, 处理步骤如下:
        1. 按照message_id计算出耗时, 放入time_cost_map
        2. 删除timeout_map对应的message_id项

        主要去掉第一帧耗时大的
        """
        for index in compose_id_indexs:
            if index in self.timeout_map:
                now = int(round(time.time() * 1000))
                cost_time = now - self.timeout_map.get(index)
                self.time_cost_map[index] = cost_time

                del self.timeout_map[index]

    # actor重新从modelpool获取文件, 因为是learner才push到modelpool, 这里加上重试机制
    def actor_get_model_from_modelpool(self):
        all_pull_model_success = False
        current_available_model_files = []
        retry_count = 0

        while not all_pull_model_success and retry_count < int(CONFIG.on_policy_error_retry_count_when_modelpool):
            (
                pull_model_success,
                current_available_model_files,
            ) = self.model_file_sync_wrapper.ckpt_sync_warper.pull_checkpoint_from_model_pool_by_on_policy(
                self.logger, self.model_file_sync_wrapper_lock
            )
            if not pull_model_success:
                # 如果本次失败, 则sleep下再重试, 这里重试的间隔设置大些
                time.sleep(CONFIG.idle_sleep_second * 1000)
            else:
                all_pull_model_success = True
                self.logger.info(f"predict actor pull_checkpoint_from_model_pool success")
                break

            retry_count += 1

        return all_pull_model_success, current_available_model_files

    def aisrv_predict_on_policy_process_master(self, model_version):
        """
        aisrv上单个predict进程的处理逻辑
        1. 拉取最新的model文件,如果成功, 继续剩余流程; 否则失败退出
        2. 加载最新model文件
        3. 等待其他从进程加载最新model文件的响应
        4. 回复learner的on-policy流程成功
        """

        """
        aisrv重新从modelpool获取文件, 因为是learner才push到modelpool, 这里加上重试机制
        """
        aisrv_get_model_file_success = False
        for i in range(int(CONFIG.on_policy_error_max_retry_rounds)):
            success, current_available_model_files = self.actor_get_model_from_modelpool()
            if success:
                aisrv_get_model_file_success = True
                break

        """
        根据aisrv从modelpool拉取model文件执行下面流程:
        1. 成功, aisrv加载最新model文件, 更新当前self.current_sync_model_version_from_learner值, 回复learner响应
        2. 失败, aisrv告警指标++, 回复learner响应
        """
        aisrv_execute_on_policy_success = False
        # 存在返回aisrv_get_model_file_success为True, 但是current_available_model_files为空的情况
        if not aisrv_get_model_file_success or not current_available_model_files:
            self.logger.error(
                f"predict actor pull_checkpoint_from_model_pool failed, so return, "
                f"not change model_version: {model_version}"
            )
            self.on_policy_pull_from_modelpool_error_cnt += 1
        else:
            self.on_policy_pull_from_modelpool_success_cnt += 1

            # actor的主predictor进程加载最新model文件
            if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
                self.load_model_common_object.load_last_new_model(CONFIG.policy_name)
            elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
                self.load_model_common_object.standard_load_last_new_model_by_framework(
                    policy_name=CONFIG.policy_name, models_path=current_available_model_files[-1]
                )
            else:
                pass

            aisrv_execute_on_policy_success = True

        return aisrv_execute_on_policy_success

    def aisrv_predict_on_policy_process_detail(self):
        """
        aisrv上的单个predict进程的on-policy的处理流程:
        1. 同步model_version请求
        1.1 获取来自learner的 model文件同步请求
        1.2 aisrv预测进程重新从modelpool获取文件
        1.2.1 如果成功则继续剩余流程
        1.2.2 失败则返回learner的明确失败的结果, learner根据情况决定是否让aisrv执行更新model_version操作, aisrv预测进程等待下一次model_version改变再走该流程
        1.2.2.1 如果aisrv预测进程返回给learner执行model_version失败, 则learner不能让aisrv执行修改model_version操作
        1.2.2.2 如果aisrv预测进程返回给learner执行model_version成功, 则learner让aisrv执行修改model_version操作
        1.3 aisrv预测进程加载最新model文件
        1.4 朝learner发送model文件同步响应
        2. 心跳请求
        2.1 心跳响应
        """

        try:
            # 获取来自learner的 model文件同步请求
            client_id, message = self.zmq_server.recv(block=False, binary=False)
            if message:
                if (
                    message[KaiwuDRLDefine.MESSAGE_TYPE]
                    == KaiwuDRLDefine.ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_REQUEST
                ):

                    """
                    predictor主进程走on-policy的流程
                    """
                    model_version = message[KaiwuDRLDefine.MESSAGE_VALUE]
                    aisrv_execute_on_policy_success = self.aisrv_predict_on_policy_process_master(model_version)
                    if not aisrv_execute_on_policy_success:
                        # 接入告警统计
                        self.logger.error(f"predict learner ask actor to set model_version: {model_version} failed")
                        self.actor_change_model_version_error_count += 1
                    else:
                        self.current_sync_model_version_from_learner = model_version

                        # 更新下predict_common里的model_version
                        self.predict_common_object.set_current_sync_model_version_from_learner(
                            self.current_sync_model_version_from_learner
                        )
                        self.actor_change_model_version_success_count += 1

                        self.logger.info(f"predict learner ask actor to set model_version: {model_version} success")

                    # actor朝learner发送model文件同步响应
                    send_data = {
                        KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_RESPONSE,
                        KaiwuDRLDefine.MESSAGE_VALUE: aisrv_execute_on_policy_success,
                    }

                    self.zmq_server.send(str(client_id), send_data, binary=False)
                    self.logger.info(f"predict learner ask actor to {message[KaiwuDRLDefine.MESSAGE_TYPE]} success")

                elif message[KaiwuDRLDefine.MESSAGE_TYPE] == KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_REQUEST:

                    # 心跳采用最简单方式即可
                    send_data = {
                        KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_RESPONSE,
                        KaiwuDRLDefine.MESSAGE_VALUE: True,
                    }

                    self.zmq_server.send(str(client_id), send_data, binary=False)
                    self.logger.info(
                        f"predict learner ask actor to {message[KaiwuDRLDefine.MESSAGE_TYPE]} success, result is {aisrv_execute_on_policy_success}"
                    )

                else:
                    self.logger.error(
                        f"predict learner learner_model_sync_req not support "
                        f"message_type {message[KaiwuDRLDefine.MESSAGE_TYPE]}, so return"
                    )
                    return

        except Exception as e:
            pass

    def aisrv_predict_on_policy_process(self):
        """
        aisrv预测进程上执行on-policy的逻辑
        1. 拉取最新的model文件
        2. 加载最新的model文件
        """
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            self.aisrv_predict_on_policy_process_detail()

    def run_once_by_direct(self) -> None:
        """
        单次run_once, 采用串行操作
        """

        # 周期性的操作, 放在最前面, 规避因为没有请求而阻塞
        self.periodic_operations()

        # aisrv预测进程上执行on-policy流程
        self.aisrv_predict_on_policy_process()

        # 读取预测请求并且预测
        with TimeIt() as ti:
            msgs = self.get_predict_request_data_by_direct()
            if msgs:

                # 执行预测
                size, pred = self.predict_common_object.predict(msgs)

            self.process_run_idle_count += 1

        # 获取采集周期里的最大值
        if self.get_and_predict_cost_time_ms < ti.interval * 1000:
            self.get_and_predict_cost_time_ms = ti.interval * 1000

        with TimeIt() as it:
            if msgs:
                compose_id_indexs = []

                # 发送预测响应
                if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
                    compose_id_indexs = (
                        self.actor_to_aisrv_response_common_object.send_response_to_aisrv_simple_fast_by_aisrv(
                            size, pred
                        )
                    )

                elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
                    compose_id_indexs = (
                        self.actor_to_aisrv_response_common_object.standard_send_response_to_aisrv_simple_fast_by_aisrv(
                            size, pred
                        )
                    )
                else:

                    # 未来扩展
                    pass

                # 处理响应的回包里超时控制
                if compose_id_indexs:
                    self.handle_message_timeouts(compose_id_indexs)

        if self.send_cost_time_ms < it.interval * 1000:
            self.send_cost_time_ms = it.interval * 1000

    def run(self):
        if not self.before_run():
            self.logger.error(f"predict before_run failed, so return", g_not_server_label)
            return

        # 无论多个policy还是单个policy, 第1个policy是获取得到的
        model_wrapper = next(iter(self.policy_model_wrapper_maps.values()))
        while not model_wrapper.should_stop():
            try:
                self.run_once_by_direct()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题, 减少CPU损耗
                if self.process_run_idle_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_idle_count = 0

            except ValueError as e:
                self.logger.exception(
                    f"predict run error: {str(e)}",
                    g_not_server_label,
                )
            except Exception as e:
                self.logger.exception(
                    f"predict run error: {str(e)}",
                    g_not_server_label,
                )

        for policy, model_wrapper in self.policy_model_wrapper_maps.items():
            model_wrapper.close()
            self.logger.info(f"predict {policy} model_wrapper.close success")
