#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @file on_policy_predictor.py
# @brief
# @author kaiwu
# @date 2023-11-28


import time
import multiprocessing
from multiprocessing import Value
from kaiwudrl.common.checkpoint.model_file_save import ModelFileSave

# 按照需要导入
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.choose_deep_learning_frameworks import *

import numpy as np
import os
import schedule
import datetime
from kaiwudrl.server.actor.predictor import Predictor
from kaiwudrl.common.utils.common_func import (
    TimeIt,
    set_schedule_event,
    make_single_dir,
    actor_learner_aisrv_count,
    get_host_ip,
    decompress_data,
    decompress_data_parallel,
)

if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
    from kaiwudrl.common.pybind11.zmq_ops.zmq_ops import ZMQPullSocket

from kaiwudrl.common.alloc.alloc_proxy import AllocProxy
from kaiwudrl.common.ipc.zmq_util import ZmqServer
from kaiwudrl.common.logging.kaiwu_logger import KaiwuLogger
from kaiwudrl.common.protocol import aisrv_actor_req_resp_pb2
from kaiwudrl.common.algorithms.model_wrapper_common import (
    create_standard_model_wrapper,
    create_normal_model_wrapper,
)
from kaiwudrl.server.common.predict_common import PredictCommon
from kaiwudrl.server.common.actor_to_aisrv_response_common import (
    ActorToAisrvResponseCommon,
)
from kaiwudrl.server.common.load_model_common import LoadModelCommon


class OnPolicyPredictor(Predictor, multiprocessing.Process):
    def __init__(self, send_server, recv_server, name):
        super().__init__(send_server, recv_server, name)

        if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
            self.names, self.dtypes, self.shapes = self.tensor_spec()

        # 进程启动的序号
        self.index = -1

        """
        actor采用批处理从zmq_server获取, 故记录了此时队列长度, 从队列里获取的耗时,
        为了减少损耗, 只是记录统计周期最后一次的值
        1. actor从zmq-server获取的队列长度, 最大为配置值, 需要查看平时是多少
        2. 从zmq-server的队列里获取数据时批处理耗时
        3. actor批处理预测耗时
        4. actor将预测结果发送给aisrv的批处理耗时
        5. actor加载最新的Model文件耗时
        """
        self.actor_from_zmq_queue_size = 0
        self.actor_from_zmq_queue_cost_time_ms = 0

        self.max_decompress_time = 0

        """
        从actor_server获取的需要预测的数据, 每次处理完成需要清空
        因为存在每次按照batch_size或者按照超时时间来读取, 那这里采用单独的线程来读取数据, 规避超时时间的限制

        """
        if CONFIG.pipeline_process_sync:
            self.predict_request_queue = multiprocessing.Queue(CONFIG.queue_size)
            self.predict_result_queue = multiprocessing.Queue(CONFIG.queue_size)

        if CONFIG.actor_server_predict_server_different_queue:
            self.predict_request_queue_from_actor_server = None

        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:

            # 主要用于actor上predictor的从进程和主进程之间通信
            self.master_conn = []
            self.slave_conn = None

        # 进程pid
        self.current_pid = 0

        # policy和model对象的map, 为了支持多agent
        self.policy_model_wrapper_maps = {}

        self.model_file_sync_wrapper = None

        # 设置公共的预测类, 便于每次预测时调用
        self.predict_common_object = None

        # actor_predict_count, actor的predict进程数目
        if CONFIG.predict_local_or_remote == KaiwuDRLDefine.PREDICT_AS_LOCAL:
            if CONFIG.self_play:
                self.actor_predict_count = 2
            else:
                self.actor_predict_count = 1
        elif CONFIG.predict_local_or_remote == KaiwuDRLDefine.PREDICT_AS_REMOTE:
            self.actor_predict_count = CONFIG.actor_predict_process_num
        else:
            pass

        self.monitor_proxy = None

    # 返回predict_request_queue
    def get_predict_request_queue(self):
        if CONFIG.pipeline_process_sync or CONFIG.actor_server_predict_server_different_queue:
            return self.predict_request_queue

        return None

    # 返回predict_result_queue
    def get_predict_result_queue(self):
        if CONFIG.pipeline_process_sync or CONFIG.actor_server_predict_server_different_queue:
            return self.predict_result_queue

        return None

    def tensor_spec(self):
        names, dtypes, shapes = [], [], []
        for name, array_spec in self.policy_conf[CONFIG.policy_name].state.state_space().items():
            names.append(name)
            dtypes.append(tf.as_dtype(array_spec.dtype))
            shapes.append(tf.TensorShape((None,) + array_spec.shape))

        names.extend([KaiwuDRLDefine.CLIENT_ID_TENSOR, KaiwuDRLDefine.COMPOSE_ID_TENSOR])
        dtypes.extend([tf.int32, tf.int32])

        # 注意这里COMPOSE_ID_TENSOR的修改需要同步修改这里
        shapes.extend([tf.TensorShape((None,)), tf.TensorShape((None, 3))])
        return names, dtypes, shapes

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

    # actor周期性的加载七彩石修改配置, 主要包括进程独有的和公共的
    def rainbow_activate(self):
        self.rainbow_wrapper.rainbow_activate_single_process(KaiwuDRLDefine.SERVER_MAIN, self.logger)
        self.rainbow_wrapper.rainbow_activate_single_process(CONFIG.svr_name, self.logger)

    def predict_stat_reset(self):
        self.predict_common_object.set_actor_batch_predict_cost_time_ms(0)
        self.actor_from_zmq_queue_cost_time_ms = 0
        self.actor_from_zmq_queue_size = 0
        if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
            self.predict_common_object.set_actor_load_last_model_cost_ms(0)
        else:
            self.load_model_common_object.set_actor_load_last_model_cost_ms(0)

        self.max_decompress_time = 0
        self.actor_to_aisrv_response_common_object.set_max_compress_size(0)
        self.actor_to_aisrv_response_common_object.set_max_compress_time(0)

    # 这里增加predict的统计项
    def predict_stat(self):

        predict_count = self.predict_common_object.predict_stat()

        if int(CONFIG.use_prometheus) and CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE:

            predict_request_queue_size = 0
            predict_result_queue_size = 0
            try:
                predict_request_queue_size = self.predict_request_queue.qsize()
                predict_result_queue_size = self.predict_result_queue.qsize()
            except Exception as e:
                pass

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
            max_compress_time = self.actor_to_aisrv_response_common_object.get_max_compress_time()
            max_compress_size = self.actor_to_aisrv_response_common_object.get_max_compress_size()
            monitor_data = {
                KaiwuDRLDefine.MONITOR_ACTOR_PREDICT_SUCC_CNT: predict_count,
                KaiwuDRLDefine.MONITOR_ACTOR_FROM_ZMQ_QUEUE_SIZE: self.actor_from_zmq_queue_size,
                KaiwuDRLDefine.MONITOR_ACTOR_FROM_ZMQ_QUEUE_COST_TIME_MS: self.actor_from_zmq_queue_cost_time_ms,
                KaiwuDRLDefine.MONITOR_ACTOR_BATCH_PREDICT_COST_TIME_MS: actor_batch_predict_cost_time_ms,
                KaiwuDRLDefine.ACTOR_TCP_AISRV: actor_learner_aisrv_count(self.host, CONFIG.svr_name),
                KaiwuDRLDefine.ACTOR_LOAD_LAST_MODEL_COST_MS: actor_load_last_model_cost_ms,
                KaiwuDRLDefine.ACTOR_LOAD_LAST_MODEL_SUCC_CNT: actor_load_last_model_succ_cnt,
                KaiwuDRLDefine.ACTOR_LOAD_LAST_MODEL_ERROR_CNT: actor_load_last_model_error_cnt,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_DECOMPRESS_TIME: self.max_decompress_time,
                KaiwuDRLDefine.MONITOR_ACTOR_PREDICT_REQUEST_QUEUE_SIZE: predict_request_queue_size,
                KaiwuDRLDefine.MONITOR_ACTOR_PREDICT_RESULT_QUEUE_SIZE: predict_result_queue_size,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_TIME: max_compress_time,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_SIZE: max_compress_size,
            }

            # on-policy情况下actor的主进程进行上报下面指标操作
            if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY and not self.index:
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

            if CONFIG.use_prometheus:
                self.monitor_proxy.put_data({self.current_pid: monitor_data})

        # 指标复原, 计算的是周期性的上报指标
        self.predict_stat_reset()

        self.logger.info(f"predict now predict count is {predict_count}")

    def start_actor_process_by_type(self):
        """
        根据不同的启动方式进行处理:
        1. 正常启动, 无需做任何操作, tensorflow会加载容器里的空的model文件启动
        2. 加载配置文件启动, 需要从COS拉取model文件再启动, tensorflow会加载容器里的model文件启动
        """

        # 按照需要引入ModelFileSave
        self.model_file_saver = ModelFileSave()
        self.model_file_saver.start_actor_process_by_type(self.logger)

    # 外界设置下model_file_sync_wrapper
    def set_model_file_sync_wrapper(self, model_file_sync_wrapper):
        self.model_file_sync_wrapper = model_file_sync_wrapper

    # 框架运行前创建必要的文件目录
    def make_dirs(self):
        make_single_dir(CONFIG.log_dir)
        make_single_dir(f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}")

    def before_run(self):

        self.logger = KaiwuLogger()
        self.current_pid = os.getpid()
        self.logger.setLoggerFormat(
            f"/{CONFIG.svr_name}/predict_pid{self.current_pid}_log_"
            f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
        )

        self.make_dirs()

        # 支持间隔N分钟, 动态修改配置文件
        if int(CONFIG.use_rainbow):
            from kaiwudrl.common.utils.rainbow_wrapper import RainbowWrapper

            self.rainbow_wrapper = RainbowWrapper(self.logger)

            # 第一次配置主动从七彩石拉取, 后再设置为周期性拉取
            self.rainbow_activate()
            set_schedule_event(CONFIG.rainbow_activate_per_minutes, self.rainbow_activate)

        # 根据不同启动方式来进行处理
        self.start_actor_process_by_type()

        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EXAM:
            self.process_pid_list = []

        # 在标准化接入中, 需要引入业务自定义的workflow, 即while True循环
        if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
            self.workflow = None

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

        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EXAM:
            if CONFIG.actor_server_async:
                self.process_pid_list.append(self.send_server.pid)
                self.process_pid_list.append(self.recv_server.pid)
            else:
                self.process_pid_list.append(self.send_server.pid)

        """
        如果actor采用tensorrt, 则需要启动下面进程来进行并行化处理:
        1. CPU到GPU拷贝进程
        2. GPU到CPU拷贝进程
        """

        """
        if  KaiwuDRLDefine.MODEL_TENSORRT == CONFIG.use_which_deep_learning_framework:
            self.actor_tensorrt_cpu2gpu = ActorTensorRTCPU2GPU(self.server)
            self.actor_tensorrt_cpu2gpu.start()

            self.actor_tensort_gpu2cpu = ActorTensorRTGPU2CPU(self.server)
            self.actor_tensort_gpu2cpu.start()

        """

        """
        model_file_sync_wrapper, actor和learner之间的Model文件同步, 采用单独的进程
        如果是on-policy算法则需要保存下来learner同步过来最新的model文件ID, 如果是off-policy则不需要
        为了编程方便, 都统一设置下
        """
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            self.model_file_sync_wrapper.ckpt_sync_warper.make_model_dirs(self.logger)

            self.current_sync_model_version_from_learner = 0

            # 只有第一个主进程才需要启动zmq_server，从进程不需要
            if not self.index:
                zmq_server_port = int(CONFIG.zmq_server_port) + 100
                self.zmq_server = ZmqServer(CONFIG.ip_address, zmq_server_port)
                self.zmq_server.bind()
                self.logger.info(
                    f"predict zmq server on-policy bind at {CONFIG.ip_address}:{zmq_server_port} for learner"
                )

                # 下面是统计告警指标
                self.on_policy_pull_from_modelpool_error_cnt = 0
                self.on_policy_pull_from_modelpool_success_cnt = 0
                self.actor_change_model_version_error_count = 0
                self.actor_change_model_version_success_count = 0

        else:
            pass

        # 启动独立的进程, 负责actor与alloc交互
        if int(CONFIG.use_alloc) and self.index == 0:
            self.alloc_proxy = AllocProxy()
            self.alloc_proxy.start()

            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EXAM:
                self.process_pid_list.append(self.alloc_proxy.pid)

        # 注册定时器任务
        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.predict_stat)

        # 设置公共的加载文件类, 便于每次加载文件时调用
        self.load_model_common_object = LoadModelCommon(self.logger)
        self.load_model_common_object.set_model_file_sync_wrapper(self.model_file_sync_wrapper)
        self.load_model_common_object.set_policy_model_wrapper_maps(self.policy_model_wrapper_maps)

        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_TRAIN:
            # on-policy条件下是在actor同步到model文件后开始加载, 其他条件下是周期性加载
            if CONFIG.algorithm_on_policy_or_off_policy != KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
                    set_schedule_event(
                        CONFIG.model_file_load_per_minutes,
                        lambda: self.load_model_common_object.load_last_new_model(CONFIG.policy_name),
                        op_gap="minutes",
                    )
                else:
                    pass

        elif CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EXAM:
            if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
                self.load_model_common_object.load_last_new_model(CONFIG.policy_name)

            elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
                self.load_model_common_object.standard_load_last_new_model_by_framework(CONFIG.policy_name)

            else:
                pass

        else:
            pass

        # 预先加载模型文件模式
        if int(CONFIG.preload_model):
            self.load_model_common_object.preload_model_file(self.policy_model_wrapper_maps)

        # 获取本机IP
        self.host = get_host_ip()

        # 进程空转了N次就主动让出CPU, 避免CPU空转100%
        self.process_run_idle_count = 0

        # 设置公共的预测类, 便于每次预测时调用
        self.predict_common_object = PredictCommon(CONFIG.policy_name, self.monitor_proxy, self.logger)
        self.predict_common_object.set_policy_model_wrapper_maps(self.policy_model_wrapper_maps)
        self.predict_common_object.set_model_file_sync_wrapper(self.model_file_sync_wrapper)
        self.predict_common_object.set_policy_conf(self.policy_conf)

        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            self.predict_common_object.set_current_sync_model_version_from_learner(
                self.current_sync_model_version_from_learner
            )

        # 设置公共的aisrv/actor朝aisrv回包的处理类, 便于每次处理回包时调用
        self.actor_to_aisrv_response_common_object = ActorToAisrvResponseCommon(self.logger)
        self.actor_to_aisrv_response_common_object.set_zmq_send_server(self.send_server)
        if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
            self.actor_to_aisrv_response_common_object.set_zmq_server(self.zmq_server)

        # 在before run最后打印启动成功日志
        self.logger.info(
            f"predict process start success at pid is {self.current_pid}, "
            f"on-policy/off-policy is {CONFIG.algorithm_on_policy_or_off_policy}, "
            f"actor_receive_cost_time_ms: {CONFIG.actor_receive_cost_time_ms}, "
            f"predict_batch_size: {CONFIG.predict_batch_size}"
        )

        return True

    def predict_tensorrt_direct(self):
        """
        流程:
        判断GPU队列里是否为空:
        1. 队列为空, 等待下次操作
        2. 队列非空, 开始处理预测请求, 并且返回actor_server预测响应
        """
        size = 0
        pred = None

        # 处理actor --> aisrv的回包
        self.send_server.put_predict_result_data([size, pred])

    # 从actor_server进程提供的队列收集预测数据, 以线程形式, 暂时不用
    def get_predict_data_from_actor_server_by_threading(self):
        while True:
            self.get_predict_data_from_actor_server()

    # actor_server获取的预测请求数据放入到on_policy_predictor里
    def put_to_predict_queue(self, predict_data):
        if not predict_data:
            return

        if self.predict_request_queue.full():
            return

        self.predict_request_queue.put(predict_data)

    # on_policy_predictor的预测结果数据放入到本地后, actor_server从本地拿走
    def get_predict_result_data(self):
        return self.predict_result_queue.get()

    def get_predict_data_from_actor_server(self):
        """
        从actor_server进程提供的队列收集预测数据, 以函数形式
        1. 如果是pipeline_process_sync为False则从actor_server队列里获取
        2. 如果是pipeline_process_sync为True则从本地队列里获取
        控制条件依据pipeline_process_sync的值:
        1. 如果是False:
            1.1 单次批处理predict_batch_size
            1.2 设置的超时时间
        2. 如果是True:
            2.1 尽最大努力获取数据
            2.2 超过predict_batch_size跳出, 平滑操作
        """

        datas = []

        with TimeIt() as it:
            if not CONFIG.pipeline_process_sync:

                # 按照时间间隔和批处理大小收包
                start_time = time.time()
                while len(datas) < int(CONFIG.predict_batch_size):

                    # 区分从哪里获取数据
                    data = None
                    if not CONFIG.actor_server_predict_server_different_queue:
                        data = self.recv_server.get_from_to_predict_queue()
                    else:
                        try:
                            data = self.predict_request_queue_from_actor_server.get()
                        except Exception as e:
                            pass

                    if data:
                        # 增加压缩和解压缩耗时
                        with TimeIt() as ti:
                            decompressed_data = decompress_data(data)

                            if CONFIG.aisrv_actor_protocol == KaiwuDRLDefine.PROTOCOL_PROTOBUF:
                                kaiwu_server_request = aisrv_actor_req_resp_pb2.AisrvActorRequest()
                                kaiwu_server_request.ParseFromString(decompressed_data)
                                client_id = kaiwu_server_request.client_id
                                compose_id = list(kaiwu_server_request.compose_id)
                                sample_size = kaiwu_server_request.sample_size
                                observation = list(kaiwu_server_request.observation)
                                legal_action = list(kaiwu_server_request.legal_action)
                                sub_action_mask = list(kaiwu_server_request.sub_action_mask)
                                lstm_hidden = list(kaiwu_server_request.lstm_hidden)
                                lstm_cell = list(kaiwu_server_request.lstm_cell)

                                decompressed_data = [
                                    np.array([observation]),
                                    np.array([legal_action]),
                                    np.array([sub_action_mask]),
                                    np.array([lstm_hidden]),
                                    np.array([lstm_cell]),
                                    np.array([client_id]),
                                    np.array([compose_id]),
                                ]

                        if self.max_decompress_time < ti.interval:
                            self.max_decompress_time = ti.interval

                        datas.append(decompressed_data)

                    # 收包超时时强制退出, 平滑处理
                    if (time.time() - start_time) * 1000 > int(CONFIG.actor_receive_cost_time_ms):
                        break

            else:

                # 最大限度收包
                while not self.predict_request_queue.empty():
                    datas.append(self.predict_request_queue.get())

                    # 最大predict_batch_size的跳出去, 平滑处理
                    if len(datas) > int(CONFIG.predict_batch_size):
                        break

        # 如果本次没有数据, 提前返回, 不需要进行处理
        datas_length = len(datas)
        if not datas_length:
            self.process_run_idle_count += 1
            return datas

        if CONFIG.distributed_tracing:
            self.logger.info(f"predict distributed_tracing get_predict_data_from_actor_server end")

        # 获取采集周期里的最大值
        if self.actor_from_zmq_queue_size < datas_length:
            self.actor_from_zmq_queue_size = datas_length

        if self.actor_from_zmq_queue_cost_time_ms < it.interval * 1000:
            self.actor_from_zmq_queue_cost_time_ms = it.interval * 1000

        return datas

    def get_predict_data_from_actor_server_parallel(self):
        """
        从actor_server进程提供的队列收集预测数据, 以函数形式, 并行处理
        1. 如果是pipeline_process_sync为False则从actor_server队列里获取
        2. 如果是pipeline_process_sync为True则从本地队列里获取
        控制条件依据pipeline_process_sync的值:
        1. 如果是False:
            1.1 单次批处理predict_batch_size
            1.2 设置的超时时间
        2. 如果是True:
            2.1 尽最大努力获取数据
            2.2 超过predict_batch_size跳出, 平滑操作
        """

        datas = []

        with TimeIt() as it:
            if not CONFIG.pipeline_process_sync:

                # 按照时间间隔和批处理大小收包
                start_time = time.time()
                data_from_queues = []
                while len(data_from_queues) < int(CONFIG.predict_batch_size):

                    # 区分从哪里获取数据
                    data = None
                    if not CONFIG.actor_server_predict_server_different_queue:
                        data = self.recv_server.get_from_to_predict_queue()
                    else:
                        try:
                            data = self.predict_request_queue_from_actor_server.get()
                        except Exception as e:
                            pass

                    if data:
                        data_from_queues.append(data)

                    # 收包超时时强制退出, 平滑处理
                    if (time.time() - start_time) * 1000 > int(CONFIG.actor_receive_cost_time_ms):
                        break

                # 批量处理数据
                if data_from_queues:
                    with TimeIt() as ti:
                        decompressed_data = decompress_data_parallel(data_from_queues)
                        if CONFIG.aisrv_actor_protocol == KaiwuDRLDefine.PROTOCOL_PROTOBUF:
                            kaiwu_server_request = aisrv_actor_req_resp_pb2.AisrvActorRequest()
                            kaiwu_server_request.ParseFromString(decompressed_data)
                            client_id = kaiwu_server_request.client_id
                            compose_id = list(kaiwu_server_request.compose_id)
                            sample_size = kaiwu_server_request.sample_size
                            observation = list(kaiwu_server_request.observation)
                            legal_action = list(kaiwu_server_request.legal_action)
                            sub_action_mask = list(kaiwu_server_request.sub_action_mask)
                            lstm_hidden = list(kaiwu_server_request.lstm_hidden)
                            lstm_cell = list(kaiwu_server_request.lstm_cell)

                            decompressed_data = [
                                np.array([observation]),
                                np.array([legal_action]),
                                np.array([sub_action_mask]),
                                np.array([lstm_hidden]),
                                np.array([lstm_cell]),
                                np.array([client_id]),
                                np.array([compose_id]),
                            ]

                    # 增加压缩和解压缩耗时
                    if self.max_decompress_time < ti.interval:
                        self.max_decompress_time = ti.interval

                    datas = decompressed_data

            else:

                # 最大限度收包
                while not self.predict_request_queue.empty():
                    datas.append(self.predict_request_queue.get())

                    # 最大predict_batch_size的跳出去, 平滑处理
                    if len(datas) > int(CONFIG.predict_batch_size):
                        break

        # 如果本次没有数据, 提前返回, 不需要进行处理
        datas_length = len(datas)
        if not datas_length:
            self.process_run_idle_count += 1
            return datas

        if CONFIG.distributed_tracing:
            self.logger.info(f"predict distributed_tracing get_predict_data_from_actor_server end")

        # 获取采集周期里的最大值
        if self.actor_from_zmq_queue_size < datas_length:
            self.actor_from_zmq_queue_size = datas_length

        if self.actor_from_zmq_queue_cost_time_ms < it.interval * 1000:
            self.actor_from_zmq_queue_cost_time_ms = it.interval * 1000

        return datas

    # actor采用tensorrt前提下流水线处理
    def run_once_tesnorrt(self):
        # 步骤1, 定时器里执行记录统计信息
        schedule.run_pending()

        # 步骤2, 进行预测, 并且获取预测响应
        self.predict_tensorrt_direct()

    def run_once(self):

        # 步骤1, 启动定时器操作, 定时器里执行记录统计信息
        schedule.run_pending()

        # 步骤2, actor预测进程上执行on-policy流程
        self.actor_predict_on_policy_process()

        # 步骤3, 从zmq/zmq-ops上获取data/tensor进行预测, 这里按照批处理获取数据, 尽最大努力去拿取队列里的数据, 如果没有则跳出该循环
        datas = self.get_predict_data_from_actor_server()
        if datas:

            # 步骤4, 预测
            size, pred = self.predict_common_object.predict(datas)

            # 步骤5, actor朝aisrv回包
            if CONFIG.distributed_tracing:
                self.logger.info(f"predict distributed_tracing predict put actor_server predict result start")

            """
            处理actor->aisrv的响应回包
            """
            if not CONFIG.pipeline_process_sync:
                if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
                    self.actor_to_aisrv_response_common_object.send_response_to_aisrv_by_actor(size, pred)
                else:
                    if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
                        self.actor_to_aisrv_response_common_object.standard_send_response_to_aisrv_simple_fast_by_actor(
                            size, pred
                        )
                    else:
                        self.actor_to_aisrv_response_common_object.send_response_to_aisrv_simple_fast_by_actor(
                            size, pred
                        )

            else:
                self.predict_result_queue.put([size, pred])

            if CONFIG.distributed_tracing:
                self.logger.info(f"predict distributed_tracing predict put actor_server predict result end")

        # Model文件同步操作, learner --> actor, 采用单独的进程处理

    def actor_predict_on_policy_process(self):
        """
        actor上执行on-policy流程
        1. actor的predict的0号进程作为预测主进程, 处理on-policy的事务
        2. actor的predict的其他进程作为预测从进程, 只是处理模型加载事务
        """
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            if not self.index:
                self.actor_predict_on_policy_process_detail()
            else:
                self.actor_predict_on_policy_process_slave()

    def actor_predict_on_policy_process_slave(self):
        """
        actor的非预测主进程, 进行下面操作:
        1. 轮询是否需要加载model文件
        1.1 如果需要加载即从预测主进程发送的管道里有model_change_version值即代表需要加载则加载
        1.1.1 加载成功, 返回预测主进程成功信息
        1.1.2 加载失败, 返回预测主进程失败信息
        1.2 如果不需要加载, 本次不做操作
        """
        if self.slave_conn and self.slave_conn.poll(0):
            model_change_version = self.slave_conn.recv()
            if model_change_version:
                # 需要区分是非标准化和标准化场景下的
                if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
                    self.load_model_common_object.load_last_new_model(CONFIG.policy_name)
                elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
                    self.load_model_common_object.standard_load_last_new_model_by_framework(CONFIG.policy_name)
                else:
                    pass

                self.current_sync_model_version_from_learner = model_change_version

                # 更新下predict_common里的model_version
                self.predict_common_object.set_current_sync_model_version_from_learner(
                    self.current_sync_model_version_from_learner
                )

                # 返回该预测从进程加载model文件成功的消息
                self.slave_conn.send(True)

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

    # 从的predictor进程需要设置model_version
    def set_model_version(self, model_version):
        self.current_sync_model_version_from_learner = model_version

    def append_predictor_master_conn(self, master_conn):
        self.master_conn.append(master_conn)

    def set_predictor_slave_conn(self, slave_conn):
        self.slave_conn = slave_conn

    def actor_predict_on_policy_process_master(self, model_version):
        """
        actor上单个predict进程的处理逻辑, 注意是有多个predict进程的场景, 见actor配置actor_predict_process_num
        1. 第0个进程作为主进程, 主进程的工作:
        1.1 拉取最新的model文件,如果成功, 继续剩余流程; 否则失败退出
        1.2 加载最新model文件
        1.3 等待其他从进程加载最新model文件的响应
        1.4 回复learner的on-policy流程成功
        2. 剩余的进程作为从进程
        2.1 加载最新model文件
        2.2 给预测主进程返回响应
        """

        """
        actor重新从modelpool获取文件, 因为是learner才push到modelpool, 这里加上重试机制
        """
        actor_get_model_file_success = False
        for i in range(int(CONFIG.on_policy_error_max_retry_rounds)):
            success, current_available_model_files = self.actor_get_model_from_modelpool()
            if success:
                actor_get_model_file_success = True
                break

        """
        根据actor从modelpool拉取model文件执行下面流程:
        1. 成功, actor加载最新model文件, 更新当前self.current_sync_model_version_from_learner值, 回复learner响应
        2. 失败, actor告警指标++, 回复learner响应
        """
        actor_execute_on_policy_success = False
        # 存在返回aisrv_get_model_file_success为True, 但是current_available_model_files为空的情况
        if not actor_get_model_file_success or not current_available_model_files:
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

            # 等待其他predictor进程的更新model文件和model_version, 采用异步方式
            all_predictor_on_policy_success = True
            for conn in self.master_conn:
                conn.send(model_version)

            # 等待所有的从进程确认加载最新model文件完成
            for conn in self.master_conn:
                retries = 0
                while retries < CONFIG.on_policy_error_max_retry_rounds:
                    change_model_version = conn.recv()
                    if change_model_version:
                        break
                    else:
                        time.sleep(CONFIG.idle_sleep_second * 1000)
                        retries += 1

                # 达到超时条件跳出整个循环, 说明此时某个预测从进程是异常的情况
                if retries == CONFIG.on_policy_error_max_retry_rounds:
                    all_predictor_on_policy_success = False
                    break

            if not all_predictor_on_policy_success:
                actor_execute_on_policy_success = False
            else:
                actor_execute_on_policy_success = True

        return actor_execute_on_policy_success

    def actor_predict_on_policy_process_detail(self):
        """
        actor上的单个predict进程的on-policy的处理流程:
        1. 同步model_version请求
        1.1 获取来自learner的 model文件同步请求
        1.2 actor重新从modelpool获取文件
        1.2.1 如果成功则继续剩余流程
        1.2.2 失败则返回learner的明确失败的结果, learner根据情况决定是否让aisrv执行更新model_version操作, actor等待下一次model_version改变再走该流程
        1.2.2.1 如果actor返回给learner执行model_version失败, 则learner不能让aisrv执行修改model_version操作
        1.2.2.2 如果actor返回给learner执行model_version成功, 则learner让aisrv执行修改model_version操作
        1.3 actor加载最新model文件
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
                    actor_execute_on_policy_success = self.actor_predict_on_policy_process_master(model_version)
                    if not actor_execute_on_policy_success:
                        # 接入告警统计
                        self.logger.error(f"predict learner ask actor to set model_version: {model_version} failed")
                        self.actor_change_model_version_error_count += 1
                    else:
                        self.current_sync_model_version_from_learner = model_version
                        self.actor_change_model_version_success_count += 1

                        self.logger.info(f"predict learner ask actor to set model_version: {model_version} success")

                    # actor朝learner发送model文件同步响应
                    send_data = {
                        KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_RESPONSE,
                        KaiwuDRLDefine.MESSAGE_VALUE: actor_execute_on_policy_success,
                    }

                    self.zmq_server.send(str(client_id), send_data, binary=False)
                    self.logger.info(
                        f"predict learner ask actor to {message[KaiwuDRLDefine.MESSAGE_TYPE]} success, result is {aisrv_execute_on_policy_success}"
                    )

                elif message[KaiwuDRLDefine.MESSAGE_TYPE] == KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_REQUEST:

                    # 心跳采用最简单方式即可
                    send_data = {
                        KaiwuDRLDefine.MESSAGE_TYPE: KaiwuDRLDefine.ON_POLICY_MESSAGE_HEARTBEAT_RESPONSE,
                        KaiwuDRLDefine.MESSAGE_VALUE: True,
                    }

                    self.zmq_server.send(str(client_id), send_data, binary=False)
                    self.logger.debug(f"predict learner ask actor to {message[KaiwuDRLDefine.MESSAGE_TYPE]} success")

                else:
                    self.logger.error(
                        f"predict learner learner_model_sync_req not support "
                        f"message_type {message[KaiwuDRLDefine.MESSAGE_TYPE]}, so return"
                    )
                    return

        except Exception as e:
            pass

    def set_index(self, index):
        self.index = index

    def set_monitor_proxy(self, monitor_proxy):
        self.monitor_proxy = monitor_proxy

    def set_predict_request_queue_from_actor_server(self, predict_request_queue_from_actor_server):
        if not predict_request_queue_from_actor_server:
            return

        self.predict_request_queue_from_actor_server = predict_request_queue_from_actor_server

    def run(self):
        if not self.before_run():
            self.logger.error(f"before_run failed, so break")
            return

        # 无论多个policy还是单个policy, 第1个policy是获取得到的
        model_wrapper = next(iter(self.policy_model_wrapper_maps.values()))
        while not model_wrapper.should_stop():
            try:
                self.run_once()

                # 因为在pipeline_process_sync模式下一直从本地收包容易导致CPU100%, 而在非pipeline_process_sync模式下有收包超时时间反而不容易发生
                if CONFIG.pipeline_process_sync:
                    # 短暂sleep, 规避容器里进程CPU使用率100%问题, 由于存在actor的按照时间间隔去预测, 故这里不休眠, 后期修改为事件机制
                    if self.process_run_idle_count % CONFIG.idle_sleep_count == 0:
                        time.sleep(CONFIG.idle_sleep_second)

                        # process_run_count置0, 规避溢出
                        self.process_run_idle_count = 0

            except Exception as e:
                self.logger.exception(f"predict failed to run {self.name} predict. exit. Error is: {e}, ")

        if CONFIG.actor_server_async:
            self.send_server.stop()
            self.recv_server.stop()
            self.logger.info("predict self.send_server.stop self.recv_server.stop success")
        else:
            self.send_server.stop()
            self.logger.info("predict self.send_server.stop success")

        for policy, model_wrapper in self.policy_model_wrapper_maps.items():
            model_wrapper.close()
            self.logger.info(f"predict {policy} model_wrapper.close success")

        self.model_file_sync_wrapper.stop()
        self.logger.info("predict self.model_file_sync_wrapper.stop success")
