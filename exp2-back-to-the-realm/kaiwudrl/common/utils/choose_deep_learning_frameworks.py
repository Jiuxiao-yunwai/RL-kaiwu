#!/usr/bin/env python
# -*- coding: utf-8 -*-


# @file choose_deep_learning_frameworks.py
# @brief
# @author kaiwu
# @date 2023-11-28


from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


"""
针对不同的文件加载不同的框架, 采用的方法:
1. 第一次解析时必须要在配置项加载后开始调用, 否则因为CONFIG.use_which_deep_learning_framework配置不正确会出错
2. 根据具体的配置加载不同的框架
"""
if (
    KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework
    or KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework
):
    from kaiwudrl.common.utils.tf_utils import *
elif KaiwuDRLDefine.MODEL_PYTORCH == CONFIG.use_which_deep_learning_framework:
    from kaiwudrl.common.utils.torch_utils import *
else:
    raise ValueError("Unsupported deep learning framework specified.")
