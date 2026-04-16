#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @file actor_server_predata.py
# @brief
# @author kaiwu
# @date 2023-11-28


import os
import multiprocessing
import datetime
import schedule
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.pybind11.zmq_ops import *
from kaiwudrl.common.utils.common_func import (
    TimeIt,
    set_schedule_event,
    decompress_data,
)
from kaiwudrl.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.monitor.monitor_proxy_process import MonitorProxy


class ActorServerPreData(multiprocessing.Process):
    """
    该类主要用于actor_server --> on_policy_predictor之间的消息处理:
    1. 从数据actor_server的收包方向里的队列数据读出
    2. 将数据放入on_policy_predictor的收包方向的队列
    """

    def __init__(self, zmq_receive_server, on_policy_predictor) -> None:
        super(ActorServerPreData, self).__init__()

        self.zmq_receive_server = zmq_receive_server
        self.on_policy_predictor = on_policy_predictor

        # 停止标志位
        self.exit_flag = multiprocessing.Value("b", False)

        # 统计数字
        self.max_decompress_time = 0

    # 返回类的名字, 便于确认调用关系
    def get_class_name(self):
        return self.__class__.__name__

    def before_run(self):

        # 日志处理
        self.logger = KaiwuLogger()
        self.current_pid = os.getpid()
        self.logger.setLoggerFormat(
            f"/{CONFIG.svr_name}/actor_server_predata_pid{self.current_pid}_log_"
            f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
            CONFIG.svr_name,
        )

        self.logger.info(
            f"actor_server process pid is {self.current_pid}, class name is {self.get_class_name()}",
            g_not_server_label,
        )

        # 启动记录发送成功失败的数目的定时器
        self.send_and_recv_zmq_stat()

        # 进程空转了N次就主动让出CPU, 避免CPU空转100%
        self.process_run_idle_count = 0

        # 访问普罗米修斯的类
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = MonitorProxy()
            self.monitor_proxy.start()

        return True

    # 定时器采用schedule, need pip install schedule
    def send_and_recv_zmq_stat(self):

        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.zmq_stat)

    def zmq_stat_reset(self):
        self.max_decompress_time = 0

    def zmq_stat(self):

        # 针对zmq_server的统计
        if int(CONFIG.use_prometheus):
            monitor_data = {
                KaiwuDRLDefine.MONITOR_ACTOR_MAX_DECOMPRESS_TIME: self.max_decompress_time,
            }

            self.monitor_proxy.put_data({self.current_pid: monitor_data})

        # 指标复原, 计算的是周期性的上报指标
        self.zmq_stat_reset()

    # 操作数据
    def actor_server_predata(self):
        data = self.zmq_receive_server.get_from_to_predict_queue()
        if data:
            # 增加压缩和解压缩耗时
            with TimeIt() as ti:
                decompressed_data = decompress_data(data)
            if self.max_decompress_time < ti.interval:
                self.max_decompress_time = ti.interval

            self.on_policy_predictor.put_to_predict_queue(decompressed_data)
        else:
            self.process_run_idle_count += 1

    def run_once(self):

        # 进行预测请求/响应的发送
        self.actor_server_predata()

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
