#!/usr/bin/env python
# -*- coding: utf-8 -*-


# @file network_builder.py
# @brief
# @author kaiwu
# @date 2023-11-28


from kaiwudrl.common.utils.tf_utils import *


class NetworkBuilder:
    """
    对NetWork类的封装
    """

    def __init__(self, network):
        self.network = network

    def build(self, input_tensors):
        with tf.compat.v1.variable_scope("network"):
            self.network.build_network(input_tensors)

            logits_p = self.network.as_p()
            logits_v = self.network.as_v()
            extra_tensors = self.network.extra_tensors()

        return logits_p, logits_v, extra_tensors

    # 从已经有的模型文件加载
    def build_from_exist_model(self, input_tensors, model_path):
        pass
