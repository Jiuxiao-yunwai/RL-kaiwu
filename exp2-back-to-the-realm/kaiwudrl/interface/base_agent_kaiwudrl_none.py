#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file base_agent_kaiwudrl_none.py
# @brief 该文件主要是在标准化过程中使用的, 无模式
# @author kaiwu
# @date 2023-11-28


"""
下面函数的调用参数顺序是位置参数, 默认参数, 关键字参数
"""


def learn_wrapper(func):
    def wrapper(agent, g_data, *args, **kargs):
        """
        无模式下的处理: 返回None
        """
        return None

    return wrapper


def predict_wrapper(func):
    def wrapper(agent, list_obs_data, *args, **kargs):
        """
        无模式下的处理: 返回None
        """
        return None

    return wrapper


def exploit_wrapper(func):
    def wrapper(agent, list_obs_data, *args, **kargs):
        """
        无模式下的处理: 返回None
        """
        return None

    return wrapper


def save_model_wrapper(func):
    def wrapper(agent, skip_decorator=False, *args, **kargs):
        """
        无模式下的处理: 返回None
        """
        return None

    return wrapper


def load_model_wrapper(func):
    def wrapper(agent, *args, **kargs):
        """
        无模式下的处理: 返回None
        """
        return None

    return wrapper
