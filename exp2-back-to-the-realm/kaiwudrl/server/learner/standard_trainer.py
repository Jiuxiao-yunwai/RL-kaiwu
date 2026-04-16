#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file standard_trainer.py
# @brief
# @author kaiwu
# @date 2023-11-28

# 按照需要导入
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.choose_deep_learning_frameworks import *
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.server.learner.on_policy_trainer import OnPolicyTrainer


class StandardTrainer(OnPolicyTrainer):
    def __init__(self):
        super(StandardTrainer, self).__init__(name=CONFIG.algo)

    def init(self):
        super().init()

    @property
    def tensor_names(self):
        """
        目前sgame的设置一个属性input_datas
        设置reverb表中数据的key
        发送样本时, 每条样本为一个dict, key为tensor_names
        """
        names = []
        names.append("input_datas")
        return names

    @property
    def tensor_dtypes(self):
        """设置样本的类型"""
        dtypes = []

        if KaiwuDRLDefine.MODEL_PYTORCH == CONFIG.use_which_deep_learning_framework:
            dtypes.append(torch.float32)
        else:
            dtypes.append(tf.float32)

        return dtypes

    @property
    def tensor_shapes(self):
        """设置样本的shape"""
        shapes = []

        """
        获取样本维度的规则:
        1. 如果用户在算法名/config.py里面的类Config是存在的, 并且存在字段sample_dim则获取该字段
        2. 如果1不存在则默认获取框架的sample_dim字段
        """
        try:
            # 假设模块路径为 '{CONFIG.algo}.config'
            module_path = f"{CONFIG.algo}.config"
            user_config_module = __import__(module_path, fromlist=["Config"])
            Config = getattr(user_config_module, "Config")

            # 检查 Config 类是否有 SAMPLE_DIM 属性
            if hasattr(Config, "SAMPLE_DIM"):
                sample_dim = Config.SAMPLE_DIM
            else:
                # 如果用户的 Config 类没有 SAMPLE_DIM 属性，则使用框架的默认值
                sample_dim = CONFIG.sample_dim

            # 确认后修改CONFIG.sample_dim的值
            CONFIG.sample_dim = sample_dim
        except (ModuleNotFoundError, AttributeError) as e:
            # 如果用户的配置文件不存在或者没有 Config 类，则使用框架的默认值
            sample_dim = CONFIG.sample_dim

        if KaiwuDRLDefine.MODEL_PYTORCH == CONFIG.use_which_deep_learning_framework:
            shapes.append(torch.Size([1, sample_dim]))
        else:
            shapes.append(tf.TensorShape((1, sample_dim)))

        return shapes
