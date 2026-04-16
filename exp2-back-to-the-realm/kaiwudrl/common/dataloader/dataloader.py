#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file dataloader.py
# @brief
# @author kaiwu
# @date 2023-11-28


import multiprocessing
import datetime
import os
import time
import json
import schedule
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from kaiwudrl.common.utils.common_func import (
    is_list_eq,
    set_schedule_event,
    python_exec_shell,
)
from kaiwudrl.common.monitor.prometheus_utils import PrometheusUtils
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


class DataLoader(multiprocessing.Process):
    """
    该类主要是负责数据导入, 业务来定义数据格式
    """

    def __init__(self) -> None:
        super(DataLoader, self).__init__()

        # 进程是否退出, 用于在异常条件下主动退出进程
        self.exit_flag = multiprocessing.Value("b", False)

    def before_run(self):

        # 日志处理
        self.logger = KaiwuLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(
            f"/{CONFIG.svr_name}/dataloader_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
            "dataloader",
        )

        # 访问普罗米修斯的类
        self.prometheus_utils = PrometheusUtils(self.logger)

        self.process_run_count = 0

        # 在before run最后打印启动成功日志
        self.logger.info(f"dataloader start success at pid {pid}", g_not_server_label)

        return True

    def run_once(self):

        # 启动定时器
        schedule.run_pending()

    # 进程停止函数
    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info("dataloader DataLoader stop success", g_not_server_label)

    def run(self) -> None:
        if not self.before_run():
            self.logger.error(f"dataloader before_run failed, so return", g_not_server_label)
            return

        while not self.exit_flag.value:
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0

            except Exception as e:
                self.logger.exception(
                    f"dataloader run error: {str(e)}",
                    g_not_server_label,
                )
