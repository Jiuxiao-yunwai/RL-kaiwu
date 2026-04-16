#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @file actor_server_postdata.py
# @brief
# @author kaiwu
# @date 2023-11-28


import os

import multiprocessing
import datetime
import schedule
import time
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.pybind11.zmq_ops import *
from kaiwudrl.common.utils.common_func import (
    set_schedule_event,
)
from kaiwudrl.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.monitor.monitor_proxy_process import MonitorProxy
from kaiwudrl.server.common.actor_to_aisrv_response_common import (
    ActorToAisrvResponseCommon,
)


class ActorServerPostData(multiprocessing.Process):
    """
    该类主要用于actor_server --> on_policy_predictor之间的消息处理:
    1. 从数据actor_server的收包方向里的队列数据读出
    2. 将数据放入on_policy_predictor的收包方向的队列
    """

    def __init__(self, zmq_send_server, on_policy_predictor) -> None:
        super(ActorServerPostData, self).__init__()

        self.zmq_send_server = zmq_send_server
        self.on_policy_predictor = on_policy_predictor

        # 停止标志位
        self.exit_flag = multiprocessing.Value("b", False)

    # 返回类的名字, 便于确认调用关系
    def get_class_name(self):
        return self.__class__.__name__

    def before_run(self):

        # 日志处理
        self.logger = KaiwuLogger()
        self.current_pid = os.getpid()
        self.logger.setLoggerFormat(
            f"/{CONFIG.svr_name}/actor_server_postdata_pid{self.current_pid}_log_"
            f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
            CONFIG.svr_name,
        )

        # 启动记录发送成功失败的数目的定时器
        self.send_and_recv_zmq_stat()

        # 进程空转了N次就主动让出CPU, 避免CPU空转100%
        self.process_run_idle_count = 0

        # 访问普罗米修斯的类
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = MonitorProxy()
            self.monitor_proxy.start()

        # 设置公共的aisrv/actor朝aisrv回包的处理类, 便于每次处理回包时调用
        self.actor_to_aisrv_response_common_object = ActorToAisrvResponseCommon(self.logger)
        self.actor_to_aisrv_response_common_object.set_zmq_send_server(self.zmq_send_server)
        if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
            self.actor_to_aisrv_response_common_object.set_zmq_server(self.zmq_send_server)

        self.logger.info(
            f"actor_server process pid is {self.current_pid}, class name is {self.get_class_name()}",
            g_not_server_label,
        )

        return True

    # 定时器采用schedule, need pip install schedule
    def send_and_recv_zmq_stat(self):

        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.zmq_stat)

    def zmq_stat_reset(self):
        self.actor_to_aisrv_response_common_object.set_max_compress_size(0)
        self.actor_to_aisrv_response_common_object.set_max_compress_time(0)

    def zmq_stat(self):

        # 针对zmq_server的统计
        if int(CONFIG.use_prometheus):
            max_compress_time = self.actor_to_aisrv_response_common_object.get_max_compress_time()
            max_compress_size = self.actor_to_aisrv_response_common_object.get_max_compress_size()
            monitor_data = {
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_TIME: max_compress_time,
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_COMPRESS_SIZE: max_compress_size,
            }

            self.monitor_proxy.put_data({self.current_pid: monitor_data})

        # 指标复原, 计算的是周期性的上报指标
        self.zmq_stat_reset()

    # 操作数据
    def actor_server_postdata(self):
        data = self.on_policy_predictor.get_predict_result_data()
        if data:
            size, pred = data

            if CONFIG.aisrv_actor_communication_way == KaiwuDRLDefine.COMMUNICATION_WAY_ZMQ_OPS:
                self.actor_to_aisrv_response_common_object.send_response_to_aisrv_by_actor(size, pred)
            else:
                self.actor_to_aisrv_response_common_object.send_response_to_aisrv_simple_fast_by_actor(size, pred)
        else:
            self.process_run_idle_count += 1

    def run_once(self):

        # 进行预测请求/响应的发送
        self.actor_server_postdata()

        # 记录发送给aisrv成功失败数目, 包括发出去和收回来的请求
        schedule.run_pending()

    def run(self):
        if not self.before_run():
            self.logger.error(f"actor_server before_run failed, so return", g_not_server_label)
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
