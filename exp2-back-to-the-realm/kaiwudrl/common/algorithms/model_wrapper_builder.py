#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file model_wrapper_builder.py
# @brief
# @author kaiwu
# @date 2023-11-28


from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


class ModelWrapperBuilder:
    """
    对ModelWrapper类的封装, 目前存在多种使用场景, 待在算法同事实际交付过程中再沉淀, 目前支持多种预测方案
    1. tensorflow, 框架加载业务类, 定义graph, session, 业务使用
    2. tensorflow, 框架加载业务类, 业务定义graph, session, 业务使用
    3. 采用pytorch类
    4. 采用tcnn类
    5. 采用其他方式
    """

    def __init__(self) -> None:
        pass

    def create_model_wrapper(self, model, logger, server=None):
        """
        根据配置项加载不同的ModelWrapper
        """

        # 根据配置项选择不同的ModelWrapper类进行实例化并返回
        if KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework:
            from kaiwudrl.common.algorithms.model_wrapper_tensorflow_simple import (
                ModelWrapperTensorflowSimple,
            )

            return ModelWrapperTensorflowSimple(model, logger, server)

        elif KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework:
            from kaiwudrl.common.algorithms.model_wrapper_tensorflow_complex import (
                ModelWrapperTensorflowComplex,
            )

            return ModelWrapperTensorflowComplex(model, logger, server)

        elif KaiwuDRLDefine.MODEL_PYTORCH == CONFIG.use_which_deep_learning_framework:
            from kaiwudrl.common.algorithms.model_wrapper_pytorch import ModelWrapperPytorch

            return ModelWrapperPytorch(model, logger, server)

        elif KaiwuDRLDefine.MODEL_TCNN == CONFIG.use_which_deep_learning_framework:
            from kaiwudrl.common.algorithms.model_wrapper_tcnn import ModelWrapperTcnn

            return ModelWrapperTcnn(model, logger, server)

        elif KaiwuDRLDefine.MODEL_TENSORRT == CONFIG.use_which_deep_learning_framework:
            from kaiwudrl.common.algorithms.model_wrapper_tensorrt import ModelWrapperTensorRT

            return ModelWrapperTensorRT(model, logger, server)

        else:
            # 如果配置项不匹配任何已知的ModelWrapper类，则返回None
            return None
