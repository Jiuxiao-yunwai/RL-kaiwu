#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file standard_predictor.py
# @brief
# @author kaiwu
# @date 2023-11-28


from kaiwudrl.server.actor.on_policy_predictor import OnPolicyPredictor
from kaiwudrl.common.config.config_control import CONFIG


class StandardPredictor(OnPolicyPredictor):
    """
    定义了预测网络的input_tensors的结构
    """

    def __init__(self, send_server, recv_server):
        super().__init__(send_server, recv_server, CONFIG.algo)
