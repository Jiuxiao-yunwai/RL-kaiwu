#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file monitor_proxy.py
# @brief
# @author kaiwu
# @date 2023-11-28


import multiprocessing
import queue
import threading
import os
import signal
import time
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.monitor.prometheus_utils import PrometheusUtils
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


class MonitorProxy(threading.Thread):
    """
    此类用于aisrv, actor, learner进程与监控产品(当前是普罗米修斯, 后期可以按照需要调整)
    独立出进程, 减少核心路径消耗
    """

    def __init__(self, logger) -> None:
        super(MonitorProxy, self).__init__()

        # 进程是否退出, 用于在异常条件下主动退出进程
        self.exit_flag = multiprocessing.Value("b", False)

        # 目前由于单个进程内有多个进程可能用到监控统计信息上报, 故日志需要区分下
        self.logger = logger
        self.logger.info(f"ppid is {threading.currentThread().ident}")

        """
        该队列使用场景:
        1. aisrv,actor,learner主进程向该队列放置监控的数据
        2. monitor_proxy从该队列拿出需要监控的数据, 发往普罗米修斯
        """
        self.msg_queue = queue.Queue(CONFIG.queue_size)

        # 记录最后上报监控的时间
        self.last_report_monitor_time = 0
        self.monitor_data_count_per_minutes = 0

    def before_run(self):

        # PrometheusUtils 工具类, 与普罗米修斯交互操作
        self.prometheus_utils = PrometheusUtils(self.logger)

        self.process_run_count = 0

        # pull模式需要启动server
        if CONFIG.use_prometheus_way == KaiwuDRLDefine.USE_PROMETHEUS_WAY_PULL:
            self.prometheus_utils.prometheus_start_http_server(CONFIG.prometheus_server_port)

        return True

    def put_data(self, monitor_data):
        """
        monitor_data采用map形式, 即key/value格式, 监控指标/监控值
        """
        if not monitor_data:
            return False

        if self.msg_queue.full():
            self.logger.error(f"queue is full, return")
            return False
        else:
            self.msg_queue.put(monitor_data)
            return True

    def get_data(self):
        """
        采用queue.Queue类的get方法, 减少CPU损耗
        """
        return self.msg_queue.get()

    def send_to_prometheus(self, monitor_data):
        if not monitor_data:
            return

        if not isinstance(monitor_data, dict):
            self.logger.error(f"monitor_data is not dict, return")
            return

        for monitor_name, montor_value in monitor_data.items():
            if isinstance(montor_value, list):
                for i in range(len(montor_value)):
                    self.prometheus_utils.gauge_use(CONFIG.svr_name, monitor_name, monitor_name, montor_value[i])
            else:
                self.prometheus_utils.gauge_use(CONFIG.svr_name, monitor_name, monitor_name, montor_value)

        # push 模式需要主动推送
        if CONFIG.use_prometheus_way == KaiwuDRLDefine.USE_PROMETHEUS_WAY_PUSH:
            self.prometheus_utils.push_to_prometheus_gateway()

        # self.logger.debug(f'monitor_proxy push_to_prometheus_gateway success')

    def run_once(self):

        # 获取需要监控的数据
        monitor_data = self.get_data()
        if monitor_data:
            now = time.time()
            if now - self.last_report_monitor_time >= CONFIG.prometheus_stat_per_minutes * 60:
                self.monitor_data_count_per_minutes = 0
                self.last_report_monitor_time = now

            # 满足大于最小的CONFIG.min_report_monitor_seconds即开始上报避免普罗米修斯服务雪崩, 否则这期间的监控数据被丢弃并且打印日志
            if self.monitor_data_count_per_minutes < CONFIG.max_report_monitor_count_per_minutes:
                self.send_to_prometheus(monitor_data)
                self.monitor_data_count_per_minutes += 1
            else:
                self.logger.error(
                    f"monitor_proxy, monitor_data_count_per_minutes {self.monitor_data_count_per_minutes} "
                    f">= CONFIG.max_report_monitor_count_per_minutes {CONFIG.max_report_monitor_count_per_minutes}, "
                    f"so monitor_data {monitor_data} will drop"
                )
        else:
            # 如果本次为空, 则self.process_run_count += 1, 尽快获得休息时间, 减少CPU损耗
            self.process_run_count += 1

    # 进程停止函数
    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info("monitor_proxy MonitorProxy stop success")

    def run(self) -> None:
        if not self.before_run():
            self.logger.error(f"monitor_proxy before_run failed, so return")
            return

        while not self.exit_flag.value:
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                time.sleep(CONFIG.idle_sleep_second)

            except Exception as e:
                self.logger.exception(f"monitor_proxy run error: {str(e)}")
