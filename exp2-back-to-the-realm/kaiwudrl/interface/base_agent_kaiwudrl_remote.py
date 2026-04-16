#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file base_agent_kaiwudrl_remote.py
# @brief 该文件主要是在标准化过程中使用的, 集群模式
# @author kaiwu
# @date 2023-11-28


from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwu_agent.agent.protocol.protocol import (
    SampleData2NumpyData,
    NumpyData2SampleData,
)


"""
下面函数的调用参数顺序是位置参数, 默认参数, 关键字参数
"""


def learn_wrapper(func):
    def wrapper(agent, g_data, *args, **kargs):
        """
        集群模式下aisrv的处理:
        1. 如果是需要在aisrv上训练, 则直接训练
        2. 如果是需要在learner上训练, 则aisrv发往reverb
        集群模式下的learner的处理:
        1. 如果是需要在learner上训练, 则learner从reverb读取样本训练
        """
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
            if g_data is None:
                return None

            # 如果是aisrv上训练则直接训练, 增加train关键字是为了解决循环调用问题
            if "train" in kargs and kargs["train"]:
                del kargs["train"]

                # 因为learner需要的是类SampleData, 故这里转换下
                datas = []
                for data in g_data[0]:
                    datas.append(NumpyData2SampleData(data))

                return func(agent, datas, *args, **kargs)

            else:
                # 因为reverb处理的是numpy数组, 故这里转换下
                datas = []
                # 如果没有设置优先级则默认优先级为1
                train_data_prioritized = []
                for data in g_data:
                    datas.append(SampleData2NumpyData(data))
                    train_data_prioritized.append(1)

                # 从aisrv传递给learner
                return agent.framework_handler.send_train_data(agent, datas, train_data_prioritized)

        elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
            if g_data is None:
                return None

            # 因为learner需要的是类SampleData, 故这里转换下
            datas = []
            for data in g_data[0]:
                datas.append(NumpyData2SampleData(data))

            # 增加train关键字是为了解决循环调用问题
            if "train" in kargs and kargs["train"]:
                del kargs["train"]

            return func(agent, datas, *args, **kargs)
        else:
            return None

    return wrapper


def predict_wrapper(func):
    def wrapper(agent, list_obs_data, *args, **kargs):
        """
        集群模式下aisrv的处理:
        1. 如果是需要在aisrv上预测, 则直接预测
        2. 如果是需要在actor上预测, 则aisrv发往actor
        集群模式下的actor的处理:
        1. 如果是需要在actor上预测, 则actor进行预测
        """
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:

            # 因为在aisrv上存在的场景有朝队列去传递数据, 进行预测, 故这里采用关键字里带predict的进行预测, 不带predict关键字的即放入队列
            if "predict" in kargs and kargs["predict"]:
                del kargs["predict"]

                return func(agent, list_obs_data, *args, **kargs)
            else:
                return agent.framework_handler.predict(agent, list_obs_data)
        elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR:
            # 增加predict关键字是为了解决循环调用问题
            if "predict" in kargs and kargs["predict"]:
                del kargs["predict"]

            return func(agent, list_obs_data, *args, **kargs)
        else:
            return None

    return wrapper


def exploit_wrapper(func):
    def wrapper(agent, list_obs_data, *args, **kargs):
        """
        集群模式下aisrv的处理:
        1. 如果是需要在aisrv上预测, 则直接预测
        2. 如果是需要在actor上预测, 则aisrv发往actor
        集群模式下的actor的处理:
        1. 如果是需要在actor上预测, 则actor进行预测
        """
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:

            # 因为在aisrv上存在的场景有朝队列去传递数据, 进行预测, 故这里采用关键字里带predict的进行预测, 不带predict关键字的即放入队列
            if "predict" in kargs and kargs["predict"]:
                del kargs["predict"]

                return func(agent, list_obs_data, *args, **kargs)
            else:
                return agent.framework_handler.exploit(agent, list_obs_data)
        elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR:
            # 增加predict关键字是为了解决循环调用问题
            if "predict" in kargs and kargs["predict"]:
                del kargs["predict"]

            return func(agent, list_obs_data, *args, **kargs)
        else:
            return None

    return wrapper


def save_model_wrapper(func):
    def wrapper(agent, *args, **kargs):
        """
        集群模式下的处理:
        1. learner, 调用KaiwuDRL保存模型文件
        2. aisrv, 采用队列发送到learner上执行
        """
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
            # 直接传递func是为了解决循环调用问题
            return agent.framework_handler.save_param(agent, func, *args, **kargs)
        elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
            return agent.framework_handler.send_save_model_file_data(agent, *args, **kargs)
        else:
            return None

    return wrapper


def load_model_wrapper(func):
    def wrapper(agent, *args, **kargs):
        """
        集群模式下actor/aisrv的处理:
        1. aisrv
        1.1 如果是actor则预测
        1.1.1 如果带有load_model字样, 直接调用业务load_model
        1.1.2 如果没有带有load_model字样, 将该请求采用zmq队列发送到actor侧
        1.2 如果是aisrv_proxy_local预测, 则直接将请求采用zmq队列发送到actor侧
        1.2.1 如果带有load_model字样, 直接调用业务的load_model
        1.2.2 如果没有带有load_model字样, 将该请求采用zmq队列发送到aisrv_proxy_local侧
        2. actor
        2.1 如果是带有load_model字样, 直接调用业务的load_model
        2.2 其他情况不做处理
        3. learner
        3.1 直接调用业务的load_model
        """
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR:

            # 增加load_model关键字是为了解决循环调用问题
            if "load_model" in kargs and kargs["load_model"]:
                del kargs["load_model"]

                return func(agent, *args, **kargs)
            else:
                return None

        elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
            return func(agent, *args, **kargs)

        elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
            if "load_model" in kargs and kargs["load_model"]:
                del kargs["load_model"]

                return func(agent, *args, **kargs)
            else:
                return agent.framework_handler.send_load_model_file_data(agent, *args, **kargs)

        else:
            return None

    return wrapper


def get_training_metrics_wrapper(func):
    def wrapper(agent, *args, **kargs):
        """
        集群模式下的处理:
        1. learner, FIXME: 暂时认为没有learner上调用的情况
        2. aisrv, 直接zmq发送到learner上获取
        """
        # if CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
        #     return agent.framework_handler.get_training_metrics(agent, func, *args, **kargs)
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
            return agent.framework_handler.get_training_metrics(agent, *args, **kargs)
        else:
            return None

    return wrapper
