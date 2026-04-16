#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file actor_server_sync_sender.py
# @brief
# @author kaiwu
# @date 2023-11-28


import os
import multiprocessing
import datetime
import schedule
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.slots import Slots
from kaiwudrl.common.pybind11.zmq_ops import *
from kaiwudrl.common.utils.common_func import (
    TimeIt,
    Context,
    set_schedule_event,
)
from kaiwudrl.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.monitor.monitor_proxy_process import MonitorProxy
from kaiwudrl.server.common.actor_to_aisrv_response_common import (
    ActorToAisrvResponseCommon,
)


class ActorServerSyncSender(multiprocessing.Process):
    """
    该类主要用于actor <--> aisrv之间的消息处理:
    1. aisrv <-- actor方向, 采用self.zmq_client
    2. aisrv --> actor方向
    2.1 如果是框架定义session, 采用self.zmq_ops_client
    2.2 如果是业务定义session, 采用self.zmq_client
    """

    def __init__(self, zmq_server) -> None:
        super(ActorServerSyncSender, self).__init__()

        self.zmq_server = zmq_server

        # 单个zmq server处理多个zmq client的请求和响应
        self.client_id = ""

        # 停止标志位
        self.exit_flag = multiprocessing.Value("b", False)

        # 上下文Context
        self.context = Context()
        self.context.slots = Slots(CONFIG.max_tcp_count)
        self.context.slot_group_name = "actor_server"
        self.context.slots.register_group(self.context.slot_group_name)

        # 使用队列模式, 获取actor的预测响应, 发给aisrv
        self.predict_result_queue = multiprocessing.Queue(CONFIG.queue_size)

        """
        if int(CONFIG.use_pipeline_predict):
            self.predict_result_queue = multiprocessing.Queue(CONFIG.queue_size)

        else:
            self.input_pipe, self.output_pipe = multiprocessing.Pipe(duplex=False)
        """

        # 采用队列模式, actor接收来自aisrv的zmq请求
        if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ:
            self.to_predict_queue = multiprocessing.Queue(CONFIG.queue_size)

        # 统计指标
        self.send_to_aisrv_succ_cnt = 0
        self.send_to_aisrv_error_cnt = 0

    # actor --> aisrv发送预测请求的响应
    def put_predict_result_data(self, predict_result_data):
        if predict_result_data:
            # 采用pickle序列化, 因为采用pipe可以去掉序列化和反序列化步骤
            # predict_result_data = pickle.dumps(predict_result_data)

            if self.predict_result_queue.full():
                # 队列满时报错
                return
            else:
                self.predict_result_queue.put(predict_result_data)

    # 返回类的名字, 便于确认调用关系
    def get_class_name(self):
        return self.__class__.__name__

    def before_run(self):

        # 日志处理
        self.logger = KaiwuLogger()
        self.current_pid = os.getpid()
        self.logger.setLoggerFormat(
            f"/{CONFIG.svr_name}/actor_server_sender_pid{self.current_pid}_log_"
            f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
            CONFIG.svr_name,
        )

        self.logger.info(
            f"actor_server zmq server bind at {CONFIG.ip_address} : {CONFIG.zmq_server_port}",
            g_not_server_label,
        )
        self.logger.info(
            f"actor_server process pid is {self.current_pid}, class name is {self.get_class_name()}",
            g_not_server_label,
        )

        # 启动记录发送成功失败的数目的定时器
        self.send_and_recv_zmq_stat()

        # 进程空转了N次就主动让出CPU, 避免CPU空转100%
        self.process_run_idle_count = 0

        """
        # 启动类内线程, 处理预测请求/响应收发, 但是效果不明显
        self.actor_receiver = threading.Thread(target=ActorServerSync.actor_receive_msg, args=(self,))
        self.actor_receiver.start()
        self.actor_sender = threading.Thread(target=ActorServerSync.actor_send_msg, args=(self,))
        self.actor_sender.start()
        """

        # 访问普罗米修斯的类
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = MonitorProxy()
            self.monitor_proxy.start()

        # actor --> aisrv发送时批量处理耗时
        self.actor_send_to_aisrv_batch_cost_time_ms = 0

        # 设置公共的aisrv/actor朝aisrv回包的处理类, 便于每次处理回包时调用
        self.actor_to_aisrv_response_common_object = ActorToAisrvResponseCommon(self.logger)
        self.actor_to_aisrv_response_common_object.set_zmq_send_server(self.zmq_server)
        if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
            self.actor_to_aisrv_response_common_object.set_zmq_server(self.zmq_server)

        self.logger.info(
            f"actor_server zmq server bind at {CONFIG.ip_address} : {CONFIG.zmq_server_port}",
            g_not_server_label,
        )
        self.logger.info(
            f"actor_server process pid is {self.current_pid}, class name is {self.get_class_name()}",
            g_not_server_label,
        )

        return True

    # 定时器采用schedule, need pip install schedule
    def send_and_recv_zmq_stat(self):

        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.zmq_stat)

    def zmq_stat_reset(self):

        self.send_to_aisrv_succ_cnt = 0
        self.send_to_aisrv_error_cnt = 0

        self.actor_send_to_aisrv_batch_cost_time_ms = 0
        self.actor_to_aisrv_response_common_object.set_max_compress_size(0)
        self.actor_to_aisrv_response_common_object.set_max_compress_time(0)

    def zmq_stat(self):

        # 针对zmq_server的统计
        if int(CONFIG.use_prometheus):
            actor_send_to_aisrv_batch_cost_time_ms = self.actor_send_to_aisrv_batch_cost_time_ms
            max_compress_time = self.actor_to_aisrv_response_common_object.get_max_compress_time()
            max_compress_size = self.actor_to_aisrv_response_common_object.get_max_compress_size()
            monitor_data = {
                KaiwuDRLDefine.MONITOR_ACTOR_SENDTO_AISRV_SUCC_CNT: self.send_to_aisrv_succ_cnt,
                KaiwuDRLDefine.MONITOR_ACTOR_SENDTO_AISRV_ERROR_CNT: self.send_to_aisrv_error_cnt,
                KaiwuDRLDefine.MONITOR_ACTOR_SENDTO_AISRV_BATCH_COST_TIME_MS: actor_send_to_aisrv_batch_cost_time_ms,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_TIME: max_compress_time,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_SIZE: max_compress_size,
            }
            self.monitor_proxy.put_data({self.current_pid: monitor_data})

        """
        self.logger.debug(f'actor_server zmq stat, send_succ_cnt is {self.send_to_aisrv_succ_cnt}, \
                          send_error_cnt is {self.send_to_aisrv_error_cnt} \
                          recv_succ cnt is {self.recv_from_aisrv_succ_cnt} \
                          recv_error_cnt is {self.recv_from_aisrv_error_cnt}', g_not_server_label)
        """

        # 指标复原, 计算的是周期性的上报指标
        self.zmq_stat_reset()

    # 如果本次无预测请求, 返回None; 否则返回数据, 注意需要处理异常, 规避队列get时阻塞时间
    def get_from_to_predict_queue(self):
        try:
            return self.to_predict_queue.get(block=False)
        except Exception as e:
            return None

    # 类中线程, 处理actor --> aisrv方向预测请求
    def actor_send_msg(self):

        # 判断队列为空self.msg_queue.empty()时, 可能出现报错Connection reset by peer, 需要使用try-except形式
        try:
            if not self.predict_result_queue.empty():
                data = self.predict_result_queue.get()
            else:
                self.process_run_idle_count += 1

                return

        except Exception as e:
            self.process_run_idle_count += 1

            return

        # 采用pickle反序列化, 因为采用pipe可以去掉序列化和反序列化步骤
        # size, pred = pickle.loads(data, encoding='bytes')

        size, pred = data
        if not size or not pred:
            self.process_run_idle_count += 1

            return

        # 获取client_ids和compose_ids
        with TimeIt() as ti:
            if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
                self.actor_to_aisrv_response_common_object.send_response_to_aisrv_by_actor(size, pred)
            else:
                self.actor_to_aisrv_response_common_object.send_response_to_aisrv_simple_fast_by_actor(size, pred)

        # 获取采集周期里的最大值
        if self.actor_send_to_aisrv_batch_cost_time_ms < ti.interval * 1000:
            self.actor_send_to_aisrv_batch_cost_time_ms = ti.interval * 1000

    def run_once(self):

        # 进行预测请求/响应的发送
        self.actor_send_msg()

        # 记录发送给aisrv成功失败数目, 包括发出去和收回来的请求
        schedule.run_pending()

    def run(self):
        if not self.before_run():
            self.logger.error(f"actor_server before_run failed", g_not_server_label)
            return

        while not self.exit_flag.value:
            try:
                self.run_once()

                """
                # 短暂sleep, 规避容器里进程CPU使用率100%问题, 由于actor的zmq_server是比较忙碌的, 这里暂时不做人为休眠, 后期修改为事件提醒机制
                if self.process_run_idle_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_idle_count = 0
                """

            except Exception as e:
                self.logger.exception(
                    f"actor_server ActorServer run error: {str(e)}",
                    g_not_server_label,
                )

    # 停止进程
    def stop(self):
        self.exit_flag.value = True
