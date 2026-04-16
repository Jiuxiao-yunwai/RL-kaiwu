#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file learner.py
# @brief
# @author kaiwu
# @date 2022-04-26

import faulthandler
import signal
import os
import io
import sys
import time
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine

# 某些机器环境没有安装tensorflow的, 故需要按需安装
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.cmd_argparser import cmd_args_parse
from kaiwudrl.common.config.algo_conf import AlgoConf
from kaiwudrl.common.config.app_conf import AppConf
from kaiwudrl.common.utils.common_func import get_local_rank
from kaiwudrl.common.utils.common_func import machine_device_check


def proc_flags(configure_file):
    CONFIG.set_configure_file(configure_file)
    CONFIG.parse_learner_configure()

    # actor上需要配置加载pybind11
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(base_dir + "/../../common/pybind11/zmq_ops")

    # 加载配置文件conf/algo_conf.json
    AlgoConf.load_conf(CONFIG.algo_conf)

    # 加载配置文件conf/app_conf.json
    AppConf.load_conf(CONFIG.app_conf)
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


def register_signal():
    try:
        faulthandler.register(signal.SIGUSR1)
    except io.UnsupportedOperation:
        pass


def app_check_param():
    """
    下面是目前业务的正确配置项, 如果配置错误, 则强制进行修正
    sgame_1v1:
    1. train_batch_size = 1024

    sgame_5v5:
    1. train_batch_size = 512

    gym:
    1.
    """
    # learner的批处理大小需要小于等于replay_buff的capacity
    if CONFIG.train_batch_size > CONFIG.replay_buffer_capacity:
        print(f"train_batch_size {CONFIG.train_batch_size} > replay_buffer_capacity {CONFIG.replay_buffer_capacity}")
        return False

    return True


# 在进程启动前进行检测参数合理性
def check_param():

    app_check_param_result = app_check_param()
    machine_device_check_result = machine_device_check(CONFIG.svr_name)

    return app_check_param_result and machine_device_check_result


def start_learner_server(replay_buffer_wrapper):
    # 启动learner_server, 包括learner_server_reverb和learner_server_zmq, 主要用于没有reverb而使用zmq的场景里
    if CONFIG.use_learner_server:
        from kaiwudrl.server.learner.learner_server import LearnerServer

        # 人为的增加sleep时间
        time.sleep(CONFIG.start_python_daemon_sleep_after_cpp_daemon_sec)

        learner_server = LearnerServer(replay_buffer_wrapper)
        learner_server.start()

    else:
        pass


def train_loop():

    train = None
    if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
        train = AlgoConf[CONFIG.algo].trainer()
    elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
        from kaiwudrl.server.learner.standard_trainer import StandardTrainer

        train = StandardTrainer()
    else:
        print("can not get trainer, please check")

        return

    if not train:
        print("can not get trainer, please check")

        return

    replay_buffer_wrapper = train.get_replay_buffer_object()
    # 进程之间有时序关系
    start_learner_server(replay_buffer_wrapper)

    # 先启动训练进程
    train.loop()


def main():
    """
    启动命令样例: python3 kaiwudrl/server/learner/learner.py --conf=conf/kaiwudrl/learner.toml
    """

    # 步骤1, 按照命令行来解析参数
    args = cmd_args_parse(KaiwuDRLDefine.SERVER_LEARNER)

    # 步骤2, 解析参数, 包括业务级别和算法级别
    proc_flags(args.conf)

    # 步骤3, 检测输入参数正确性
    if not check_param():
        print("conf param error, please check")
        return

    # 步骤4, 处理信号
    register_signal()

    # 步骤5, 开始轮训处理
    train_loop()


if __name__ == "__main__":
    sys.exit(main())
