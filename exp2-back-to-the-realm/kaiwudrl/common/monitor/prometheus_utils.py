#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file prometheus_utils.py
# @brief
# @author kaiwu
# @date 2023-11-28


"""
普罗米修斯的官网见:https://prometheus.io/
需要安装prometheus_client, 采用pip install prometheus_client, 见: https://github.com/prometheus/client_python
"""

"""
下面是KaiwuDRL上报到普罗米修斯的指标, 容器方面的采集指标复用k8s自带的

采用Counter、Gauge、Histogram、Summary

aisrv
1. aisrv --> actor --> aisrv的平均时延, 最大时延
2. aisrv的QPS
3. aisrv进程的CPU, 内存占用

actor
1. actor的单次预测平均时延, 最大时延
2. actor的GPU使用率

learner
1. learner的GPU使用率
2. learner的单次预测平均时延, 最大时延

"""

import os
import hashlib

MONITOR_ITEMS = {
    "aisrv": [],
    "actor": [],
    "learner": [],
}

import random
from prometheus_client import (
    Counter,
    Histogram,
    Summary,
    Gauge,
    push_to_gateway,
    CollectorRegistry,
    start_http_server,
)
import time
from kaiwudrl.common.config.config_control import CONFIG
from prometheus_client.exposition import basic_auth_handler
from kaiwudrl.common.utils.common_func import get_host_ip
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.http_utils import http_utils_request, http_utils_delete


# 普罗米修斯监控方面
class PrometheusUtils(object):
    def __init__(self, logger, should_clear_data=False) -> None:

        # 下面是参数配置
        self.prometheus_pwd = CONFIG.prometheus_pwd
        self.prometheus_user = CONFIG.prometheus_user
        self.prometheus_pushgateway = CONFIG.prometheus_pushgateway
        self.prometheus_instance = CONFIG.prometheus_instance
        self.prometheus_db = CONFIG.prometheus_db

        # 同一个容器里可能会启动多个进程, 故采用CONTAINER_INDEX区分
        self.container_index = os.getenv("CONTAINER_INDEX", "0")

        # 本机IP名
        self.host = get_host_ip()

        # task_id
        self.task_id = CONFIG.task_id

        # 上报进程的pid
        self.current_pid = os.getpid()

        # job名, 格式固定, 采用多label形式
        self.job = f"kaiwu_tasks_{self.current_pid}"

        self.logger = logger

        # 注意每次push后复用问题
        self.registry = CollectorRegistry()

        # 注意每次定义时不能重复, 格式是{srv_name}_{item_name}, 确保每一项指标有对应的数据结构, 故采用map形式
        self.g_maps = {}
        self.c_maps = {}
        self.h_maps = {}
        self.s_maps = {}

        """
        label的设计:
        1. 如果是多进程里调用, 比如aisrv, learner, 则需要带上pid, 即host_index_pid
        2. 如果是单进程里调用, 比如kaiwu_env, 则不需要带上pid, 即host_index

        默认的labels, 需要区分出来各个进程的情况, task_id, instance规避不同进程间的数量覆盖的问题
        """
        self.default_labels = ["task_id", "app", "instance"]
        self.label_values = {
            "task_id": self.task_id,
            "app": CONFIG.app,
            "instance": f"{self.host}_{self.container_index}_{self.current_pid}",
        }

        """
        设计时只是learner在清理数据, 其余进程不需要
        """
        self.should_clear_data = should_clear_data

        # 最后探测push_gateway是否健康的时间
        self.last_check_push_gateway_available_time = 0
        self.last_check_result = True
        self.last_clear_push_gateway_data_time = time.time()

    def prometheus_start_http_server(self, port):
        """
        机器上启动普罗米修斯服务器
        """
        start_http_server(port)

    # 检测进程名是否属于范围内
    def check_server_name(self, server_name):
        if server_name not in [
            KaiwuDRLDefine.SERVER_AISRV,
            KaiwuDRLDefine.SERVER_ACTOR,
            KaiwuDRLDefine.SERVER_LEARNER,
            KaiwuDRLDefine.SERVER_CLIENT,
        ]:
            self.logger.error(f"server_name {server_name} is not valid")
            return False

        return True

    # 认证
    def auth_handler(self, url, method, timeout, headers, data):
        return basic_auth_handler(
            url,
            method,
            timeout,
            headers,
            data,
            self.prometheus_user,
            self.prometheus_pwd,
        )

    # Counter使用, 只是增加不减少
    def counter_use(self, server_name, item_name, item_help, value, pid=None):
        if not self.check_server_name(server_name):
            return

        if pid is not None:
            self.label_values["instance"] = f"{self.host}_{self.container_index}_{pid}"

        metric_key = f"{server_name}_{item_name}"
        if metric_key not in self.c_maps:
            self.c_maps[metric_key] = Counter(
                item_name,
                item_help,
                registry=self.registry,
                labelnames=self.default_labels,
            )

        self.c_maps[metric_key].labels(**self.label_values).inc(value)

    # Histogram使用, 直方图
    def histogram_use(self, server_name, item_name, item_help, value, pid=None):
        if not self.check_server_name(server_name):
            return

        if pid is not None:
            self.label_values["instance"] = f"{self.host}_{self.container_index}_{pid}"

        metric_key = f"{server_name}_{item_name}"
        if metric_key not in self.h_maps:
            self.h_maps[metric_key] = Histogram(
                item_name,
                item_help,
                registry=self.registry,
                labelnames=self.default_labels,
            )

        self.h_maps[metric_key].labels(**self.label_values).observe(value)

    # Summary使用
    def summary_use(self, server_name, item_name, item_help, value, pid=None):
        if not self.check_server_name(server_name):
            return

        if pid is not None:
            self.label_values["instance"] = f"{self.host}_{self.container_index}_{pid}"

        metric_key = f"{server_name}_{item_name}"
        if metric_key not in self.s_maps:
            self.s_maps[metric_key] = Summary(
                item_name,
                item_help,
                registry=self.registry,
                labelnames=self.default_labels,
            )

        self.s_maps[metric_key].labels(**self.label_values).observe(value)

    # Gauge使用, 可增可减
    def gauge_use(self, server_name, item_name, item_help, item_value, pid=None):
        if not self.check_server_name(server_name):
            return

        if pid is not None:
            self.label_values["instance"] = f"{self.host}_{self.container_index}_{pid}"

        # 需要保证是调用第一次来定义Gauge, 并且lablenames不能带上item_name
        metric_key = f"{server_name}_{item_name}"
        if metric_key not in self.g_maps:
            self.g_maps[metric_key] = Gauge(
                item_name,
                item_help,
                registry=self.registry,
                labelnames=self.default_labels,
            )

        self.g_maps[metric_key].labels(**self.label_values).set(item_value)

    def is_push_gateway_healthy(self):
        """
        探测push_gate_way是否是健康
        1. 如果不需要探测是否健康, 则直接返回True
        2. 如果没有达到需要探测健康的时间间隔, 则直接返回True
        3. 如果需要探测是否健康
        3.1 健康则返回True
        3.2 不健康则返回False
        """

        if not CONFIG.check_prometheus_way_availability:
            return True

        current_time = time.time()
        if (
            current_time - self.last_check_push_gateway_available_time
            < CONFIG.check_prometheus_way_availability_per_seconds
        ):
            return self.last_check_result

        try:
            self.last_check_push_gateway_available_time = current_time

            # 针对url的设置
            url = self.prometheus_pushgateway
            if not url.startswith("http://") and not url.startswith("https://"):
                url = f"http://{url}"
            url = f"{url}/-/healthy"

            resp = http_utils_request(url, print_error_msg=False)
            if resp == "OK":
                self.last_check_result = True
            else:
                self.last_check_result = False
        except Exception as e:
            self.last_check_result = False

        return self.last_check_result

    def push_to_prometheus_gateway(self):
        """
        1. 如果是push模式则需要调用:
            由于每次push_to_gateway需要和网络调用,
            故需要调用N次gauge_use或者summary_use或者histogram_use或者counter_use后再调用push_to_prometheus_gateway, 减少网络耗时
            不能每次就调用push_to_gateway

        2. 如果是pull模式不需要调用
        """
        try:
            if self.is_push_gateway_healthy():
                push_to_gateway(
                    self.prometheus_pushgateway,
                    job=self.job,
                    registry=self.registry,
                    handler=self.auth_handler,
                )
            else:
                # 此时可能存在情况是push_gate_way已经不正常了, 打印日志即可
                self.logger.info(
                    f"push_to_gateway is not healthy, prometheus_pushgateway is {self.prometheus_pushgateway}"
                )
        except Exception as e:
            self.logger.info(
                f"push_to_gateway failed, error is {str(e)}, prometheus_pushgateway is {self.prometheus_pushgateway}"
            )
        finally:
            # 确保无论如何都执行清理
            self.clear_data()

    def clear_data(self):
        """
        清理的数据包括:
        1. 客户端上报时的CollectorRegistry
        2. pushgateway上一段时间的旧数据
        """
        if not self.should_clear_data:
            return

        now = time.time()
        if now - self.last_clear_push_gateway_data_time >= 10 * CONFIG.prometheus_stat_per_minutes * 60:
            delete_url = f"{self.prometheus_pushgateway}/metrics/job/{self.job}"
            if not delete_url.startswith("http://") and not delete_url.startswith("https://"):
                delete_url = f"http://{delete_url}"
            status, success = http_utils_delete(
                url=delete_url, auth=(self.prometheus_user, self.prometheus_pwd), print_error_msg=False
            )
            if success:
                self.logger.info(f"Cleaned metrics for task {self.task_id}, url is {delete_url}, status is {status}")
            else:
                self.logger.info(
                    f"Failed to clean metrics for task {self.task_id}, url is {delete_url}, status is {status}"
                )

            self.last_clear_push_gateway_data_time = now
