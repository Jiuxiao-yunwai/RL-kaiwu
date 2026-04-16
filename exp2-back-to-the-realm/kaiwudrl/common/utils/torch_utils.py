#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file torch_utils.py
# @brief
# @author kaiwu
# @date 2023-11-28


"""
与PyTorch相关的公共函数
"""


# 如果是pytorch的框架, 则需要设置下环境变量, 规避出现CPU满载情况
import os
from kaiwudrl.common.config.config_control import CONFIG

os.environ["OMP_NUM_THREADS"] = f"{CONFIG.torch_num_threads}"
os.environ["MKL_NUM_THREADS"] = f"{CONFIG.torch_num_threads}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{CONFIG.torch_num_threads}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{CONFIG.torch_num_threads}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{CONFIG.torch_num_threads}"

# torch设置线程运行数目
import torch

# 用户可能设置故KaiuwDRL不做强制设置
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine

if getattr(CONFIG, f"{CONFIG.svr_name}_device_type") == KaiwuDRLDefine.MACHINE_DEVICE_NPU:
    import torch_npu

    torch.npu.set_compile_mode(jit_compile=False)


# 判断机器上GPU是否安装成功
def torch_is_gpu_available():
    return torch.cuda.is_available()


# 判断机器上是否安装NPU
def torch_is_npu_available():
    return torch_npu.npu.is_available()


# 设置运行的GPU卡
def set_runtime_gpu():
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


# 根据业务配置来确定device的情况
def get_machine_device_by_config(process_name):
    if not process_name:
        return None

    if getattr(CONFIG, f"{process_name}_device_type") == KaiwuDRLDefine.MACHINE_DEVICE_NPU:
        device = "npu:0"
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return device


# 获取GPU的型号
def get_gpu_info():
    return torch.cuda.get_device_name()


# 释放pytorch占用的显存
def release_cache():
    torch.cuda.empty_cache()


# 判断某个model文件是合理的
def pytorch_model_file_valid(model_path):
    if not model_path:
        return False
    try:
        # 加载模型
        model = torch.load(model_path)
        return True
    except Exception as e:
        return False


# 编译Torch脚本为本机代码
def torch_compile_func(func_name):
    if not func_name:
        return None

    compiled_func_name = torch.jit.compile(func_name)
    return compiled_func_name


# 打印2个model文件之间的差异
def difference_between_models(model1_file, model2_file):
    if not model1_file or not model2_file:
        print("模型文件路径不能为空")
        return

    if not os.path.exists(model1_file) or not os.path.exists(model2_file):
        print("一个或两个模型文件不存在")
        return

    try:
        # 加载模型
        model1 = torch.load(model1_file)
        model2 = torch.load(model2_file)
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return

    # 比较参数
    for name1, param1 in model1.items():
        if name1 in model2:
            param2 = model2[name1]
            if not torch.equal(param1, param2):
                print(f"参数 {name1} 在两个模型中不同。")
                # 可以进一步打印出差异，例如计算差异的范数或直接显示差异值
                print("差异:", torch.norm(param1 - param2))
        else:
            print(f"参数 {name1} 在第二个模型中不存在。")
