#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file monitor_build.py
# @brief
# @author kaiwu
# @date 2023-11-28


from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.monitor.monitor_proxy import MonitorProxy


class MonitorBuilder:
    """
    主要是考虑到有多个进程调用, 但是只是需要初始化monitor_proxy一次
    """

    def __init__(self) -> None:
        self.monitor_proxy = None

    def build(self):
        pass
