#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
from kaiwudrl.common.utils.http_utils import http_utils_request
from kaiwudrl.common.config.config_control import CONFIG


def get_training_metrics():
    url = f"http://{CONFIG.prometheus_pushgateway}/api/v1/metrics"
    resp = http_utils_request(url)
    if not resp:
        return False

    datas = resp.get("data", [])

    # 初始化统计变量, 严格对应在监控面板展示的指标
    metrics_dict = {
        "basic": {
            "train_global_step": "sum",
            "actor_predict_succ_cnt": "sum",
            "sample_production_and_consumption_ratio": "avg",
            "episode_cnt": "sum",
            "actor_load_last_model_succ_cnt": "sum",
            "sample_receive_cnt": "sum",
        },
        "algorithm": {
            "reward": "avg",
            "q_value": "avg",
            "value_loss": "avg",
        },
        "env": {
            "total_score": "avg",
            "treasure_score": "avg",
            "max_steps": "avg",
            "finished_steps": "avg",
            "treasure_random": "avg",
            "total_treasures": "avg",
            "collected_treasures": "avg",
            "skill_cnt": "avg",
            "buff_cnt": "avg",
        },
        "diy": {
            "diy_1": "avg",
            "diy_2": "avg",
            "diy_3": "avg",
            "diy_4": "avg",
            "diy_5": "avg",
        },
    }

    # 初始化统计变量
    metrics_sum = {category: {metric: 0 for metric in metrics.keys()} for category, metrics in metrics_dict.items()}
    metrics_count = {category: {metric: 0 for metric in metrics.keys()} for category, metrics in metrics_dict.items()}

    # 遍历数据
    for data in datas:
        for category, metrics in metrics_dict.items():
            for metric, method in metrics.items():
                if metric in data:
                    metric_list = data[metric]["metrics"]
                    for metric_data in metric_list:
                        metrics_sum[category][metric] += float(metric_data["value"])
                        metrics_count[category][metric] += 1

    # 计算平均值和总和
    metrics_result = {}
    for category, metrics in metrics_dict.items():
        metrics_result[category] = {}
        for metric, method in metrics.items():
            if method == "avg":
                if metrics_count[category][metric] > 0:
                    metrics_result[category][metric] = round(
                        metrics_sum[category][metric] / metrics_count[category][metric], 2
                    )
                else:
                    metrics_result[category][metric] = 0
            elif method == "sum":
                metrics_result[category][metric] = round(metrics_sum[category][metric], 2)
        # 删除空的类别
        if not metrics_result[category]:
            del metrics_result[category]

    # 修改特定的键值
    if "basic" in metrics_result and "actor_predict_succ_cnt" in metrics_result["basic"]:
        metrics_result["basic"]["predict_succ_cnt"] = metrics_result["basic"].pop("actor_predict_succ_cnt")
    if "basic" in metrics_result and "actor_load_last_model_succ_cnt" in metrics_result["basic"]:
        metrics_result["basic"]["load_model_succ_cnt"] = metrics_result["basic"].pop("actor_load_last_model_succ_cnt")

    return metrics_result
