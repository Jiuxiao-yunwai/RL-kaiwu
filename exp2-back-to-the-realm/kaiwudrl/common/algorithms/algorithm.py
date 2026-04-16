#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file algorithm.py
# @brief
# @author kaiwu
# @date 2023-11-28


class Algorithm(object):
    def __init__(self, model):
        self.model = model

    def build_graph(self, datas, update):
        self.update = update
        self.model.inference(datas)
        self._calculate_loss()

    def get_optimizer(self):
        raise NotImplementedError("get optimizer: not implemented")

    def _calculate_loss(self):
        raise NotImplementedError("calculate loss: not implemented")
