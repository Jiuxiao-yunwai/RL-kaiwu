#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file aisrv.py
# @brief
# @author kaiwu
# @date 2022-04-26


import os
import sys
import faulthandler
import signal
import io
import os
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.cmd_argparser import cmd_args_parse
from kaiwudrl.common.config.app_conf import AppConf
from kaiwudrl.common.config.algo_conf import AlgoConf
from kaiwudrl.common.utils.common_func import make_single_dir
from kaiwudrl.common.utils.common_func import machine_device_check


def proc_flags(configure_file):
    # 解析aisrv进程的配置
    CONFIG.set_configure_file(configure_file)
    CONFIG.parse_aisrv_configure()

    # aisrv上需要配置加载pybind11
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(base_dir + "/../../common/pybind11/zmq_ops")
    sys.path.append(base_dir + "/../../../")

    # 解析业务app的配置
    AppConf.load_conf(CONFIG.app_conf, CONFIG.svr_name)

    # 加载配置文件conf/algo_conf.json
    AlgoConf.load_conf(CONFIG.algo_conf)

    # 确保框架需要的文件目录存在
    make_single_dir(CONFIG.log_dir)

    if (
        KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE == CONFIG.use_which_deep_learning_framework
        or KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX == CONFIG.use_which_deep_learning_framework
    ):
        # 设置TensorFlow日志级别
        from kaiwudrl.common.utils.tf_utils import set_tensorflow_log_level

        set_tensorflow_log_level()

        # actor需要设置在GPU机器上运行
        if "GPU" == CONFIG.actor_device_type:
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            os.environ["CUDA_VISIBLE_DEVICES"] = str(get_local_rank())

    elif KaiwuDRLDefine.MODEL_PYTORCH == CONFIG.use_which_deep_learning_framework:
        from kaiwudrl.common.utils.torch_utils import torch_is_gpu_available

    else:
        pass


def register_signal():
    try:
        faulthandler.register(signal.SIGUSR1)
    except io.UnsupportedOperation:
        pass


def app_check_param():
    """
    下面是目前业务的正确配置项, 如果配置错误, 则强制进行修正
    sgame_1v1:
    1. aisrv_actor_protocol = pickle
    2. aisrv_actor_protocol = protobuf

    sgame_5v5:
    1. aisrv_actor_protocol = protobuf

    gym:
    1. aisrv_actor_protocol = pickle
    """

    if CONFIG.app == KaiwuDRLDefine.APP_SGAME_1V1:
        pass

    elif CONFIG.app == KaiwuDRLDefine.APP_SGAME_5V5:
        if CONFIG.aisrv_actor_protocol != KaiwuDRLDefine.PROTOCOL_PROTOBUF:
            CONFIG.aisrv_actor_protocol = KaiwuDRLDefine.PROTOCOL_PROTOBUF

    elif CONFIG.app == KaiwuDRLDefine.APP_GYM:
        pass

    else:
        pass

    return True


def check_param():
    """
    在进程启动前进行检测参数合理性
    """
    app_check_param_result = app_check_param()
    machine_device_check_result = machine_device_check(CONFIG.svr_name)

    return app_check_param_result and machine_device_check_result


def main():
    """
    启动命令样例: python3 kaiwudrl/server/aisrv/aisrv.py --conf=conf/kaiwudrl/aisrv.toml
    """

    os.chdir(CONFIG.project_root)

    # 步骤1, 按照命令行来解析参数
    args = cmd_args_parse(KaiwuDRLDefine.SERVER_AISRV)

    # 步骤2, 解析参数, 包括业务级别和算法级别
    proc_flags(args.conf)

    # 步骤3, 检测输入参数正确性
    if not check_param():
        print("conf param error, please check")
        return

    # 步骤4, 处理信号
    register_signal()

    # 步骤5, 启动进程
    if KaiwuDRLDefine.AISRV_FRAMEWORK_SOCKETSERVER == CONFIG.aisrv_framework:
        # python版本
        from kaiwudrl.server.aisrv.aisrv_socketserver import AiSrv, AiSrvHandle

        server = AiSrv((CONFIG.aisrv_ip_address, CONFIG.aisrv_server_port), AiSrvHandle)
        server.serve_forever()

    elif KaiwuDRLDefine.AISRV_FRAMEWORK_KAIWUDRL == CONFIG.aisrv_framework:
        # C++版本
        from kaiwudrl.server.aisrv.aisrv_server import AiServer

        server = AiServer()
        server.run()

    elif KaiwuDRLDefine.AISRV_FRAMEWORK_KAIWU_ENV == CONFIG.aisrv_framework:
        # python版本
        from kaiwudrl.server.aisrv.aisrv_server_standard import AiServer

        server = AiServer()
        server.run()

    else:
        print(
            f"not support {CONFIG.aisrv_framework}, only support "
            f"{KaiwuDRLDefine.AISRV_FRAMEWORK_SOCKETSERVER} or "
            f"{KaiwuDRLDefine.AISRV_FRAMEWORK_KAIWUDRL}"
        )


if __name__ == "__main__":
    sys.exit(main())
