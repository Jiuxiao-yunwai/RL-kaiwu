#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file base_agent_kaiwudrl_local.py
# @brief 该文件主要是在标准化过程中使用的, 单机模式
# @author kaiwu
# @date 2023-11-28

from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


"""
下面函数的调用参数顺序是位置参数, 默认参数, 关键字参数
"""


def learn_wrapper(func):
    def wrapper(agent, g_data, *args, **kargs):
        """
        单机模式下aisrv的处理:
        1. 如果是需要在aisrv上训练, 则直接训练
        单机模式下的learner的处理:
        1. 如果是需要在learner上训练, 则直接训练
        """
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:

            if g_data is None:
                return None

            # 增加train关键字是为了解决循环调用问题
            if "train" in kargs and kargs["train"]:
                del kargs["train"]

                return func(agent, g_data, *args, **kargs)

            else:
                return agent.framework_handler.train_local(agent, g_data, *args, **kargs)

        elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
            if g_data is None:
                return None

            return func(agent, g_data, *args, **kargs)
        else:
            return None

    return wrapper


def predict_wrapper(func):
    def wrapper(agent, list_obs_data, *args, **kargs):
        """
        单机模式下aisrv的处理:
        1. 如果是需要在aisrv上预测, 则直接预测
        单机模式下的actor的处理:
        1. 如果是需要在actor上预测, 则直接预测
        """
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:

            # 增加predict关键字是为了解决循环调用问题
            if "predict" in kargs and kargs["predict"]:
                del kargs["predict"]

                return func(agent, list_obs_data, *args, **kargs)

            else:
                return agent.framework_handler.predict_local(agent, list_obs_data, *args, **kargs)

        elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR:
            return func(agent, list_obs_data, *args, **kargs)
        else:
            return None

    return wrapper


def exploit_wrapper(func):
    def wrapper(agent, list_obs_data, *args, **kargs):
        """
        单机模式下aisrv的处理:
        1. 如果是需要在aisrv上预测, 则直接预测
        单机模式下的actor的处理:
        1. 如果是需要在actor上预测, 则直接预测
        """
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:

            # 增加predict关键字是为了解决循环调用问题
            if "predict" in kargs and kargs["predict"]:
                del kargs["predict"]

                return func(agent, list_obs_data, *args, **kargs)

            else:
                return agent.framework_handler.exploit_local(agent, list_obs_data, *args, **kargs)

        elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR:
            return func(agent, list_obs_data, *args, **kargs)
        else:
            return None

    return wrapper


def save_model_wrapper(func):
    def wrapper(agent, *args, **kargs):
        """
        单机模式下aisrv/learner的处理:
        1. 调用KaiwuDRL保存模型文件
        """
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV or CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
            # 直接传递func是为了解决循环调用问题
            return agent.framework_handler.save_param(agent, func, *args, **kargs)
        else:
            return None

    return wrapper


def load_model_wrapper(func):
    def wrapper(agent, *args, **kargs):
        """
        单机模式下aisrv/actor的处理:
        1. 调用业务加载模型文件
        单机模式下learner的处理:
        1. 调用业务加载模型文件
        """
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV or CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR:
            if "load_model" in kargs and kargs["load_model"]:
                del kargs["load_model"]

                return func(agent, *args, **kargs)
            else:
                # 直接传递func是为了解决循环调用问题
                return agent.framework_handler.standard_load_last_new_model(agent, func, *args, **kargs)
        elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
            return func(agent, *args, **kargs)
        else:
            return None

    return wrapper


def get_training_metrics_wrapper(func):
    def wrapper(agent, *args, **kargs):
        """
        单机模式下的处理:
        1. learner, FIXME: 暂时认为没有learner上调用的情况
        2. aisrv, 直接zmq发送到learner上获取
        """
        # if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
        #     return agent.framework_handler.get_training_metrics(agent, func, *args, **kargs)
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
            return agent.framework_handler.get_training_metrics_local(agent, *args, **kargs)
        else:
            return None

    return wrapper
