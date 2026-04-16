#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file actor.py
# @brief
# @author kaiwu
# @date 2022-04-26


import os
import sys

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir + "/../../common/pybind11/zmq_ops")
import faulthandler
import signal
import io
import multiprocessing
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.cmd_argparser import cmd_args_parse
from kaiwudrl.common.config.app_conf import AppConf
from kaiwudrl.common.config.algo_conf import AlgoConf
from kaiwudrl.common.utils.common_func import get_local_rank, get_gpu_machine_type
from kaiwudrl.common.checkpoint.model_file_sync_wrapper import ModelFileSyncWrapper
from kaiwudrl.common.utils.common_func import machine_device_check


def proc_flags(configure_file):
    CONFIG.set_configure_file(configure_file)
    CONFIG.parse_actor_configure()

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
    1. use_pipeline_predict = False
    2. pipeline_process_sync = True
    3. actor_server_async = False
    4. python_cpp_daemon = False
    5. cpp_daemon_send_recv_zmq_data = False
    6. aisrv_actor_protocol = pickle
    7. use_which_deep_learning_framework = tensorflow_simple

    sgame_5v5:
    1. use_pipeline_predict = True
    2. pipeline_process_sync = False
    3. actor_server_async = False
    4. python_cpp_daemon = True
    5. cpp_daemon_send_recv_zmq_data = True
    6. aisrv_actor_protocol = protobuf
    7. use_which_deep_learning_framework = tensorrt

    gym:
    1.
    """
    if CONFIG.app == KaiwuDRLDefine.APP_SGAME_1V1:
        if CONFIG.use_pipeline_predict:
            CONFIG.use_pipeline_predict = False
        if CONFIG.python_cpp_daemon:
            CONFIG.python_cpp_daemon = False
        if CONFIG.cpp_daemon_send_recv_zmq_data:
            CONFIG.cpp_daemon_send_recv_zmq_data = False
        if CONFIG.use_which_deep_learning_framework != KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE:
            CONFIG.use_which_deep_learning_framework = KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE
    elif CONFIG.app == KaiwuDRLDefine.APP_SGAME_5V5:
        if not CONFIG.use_pipeline_predict:
            CONFIG.use_pipeline_predict = True
        if not CONFIG.python_cpp_daemon:
            CONFIG.python_cpp_daemon = True
        if not CONFIG.cpp_daemon_send_recv_zmq_data:
            CONFIG.cpp_daemon_send_recv_zmq_data = True
        if CONFIG.aisrv_actor_protocol != KaiwuDRLDefine.PROTOCOL_PROTOBUF:
            CONFIG.aisrv_actor_protocol = KaiwuDRLDefine.PROTOCOL_PROTOBUF
        if CONFIG.use_which_deep_learning_framework != KaiwuDRLDefine.MODEL_TENSORRT:
            CONFIG.use_which_deep_learning_framework = KaiwuDRLDefine.MODEL_TENSORRT
    elif CONFIG.app == KaiwuDRLDefine.APP_GYM:
        pass
    else:
        pass

    return True


def model_file_check():
    """
    如果是eval模式, 需要验证加载的model文件是否正常, 包括:
    1. model文件是否存在
    2. model文件是否加载正常
    """
    if CONFIG.run_mode != KaiwuDRLDefine.RUN_MODEL_EVAL:
        return True

    if not CONFIG.eval_model_dir:
        print(f"eval_model_dir {CONFIG.eval_model_dir} is empty")
        return False

    # 如果是tensorflow是目录, 如果是pytorch是文件
    if (
        CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE
        or CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX
        or CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_TENSORRT
    ):
        # 因为tensorflow的需要加载图后才能判断是否正确, 故放到on_policy_predictor.py里实现
        if not os.path.exists(CONFIG.eval_model_dir + ".meta"):
            print(f"eval_model_dir {CONFIG.eval_model_dir} is not exist")
            return False
        return True
        # return tensorflow_model_file_valid(CONFIG.eval_model_dir)
    elif CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_PYTORCH:

        # return pytorch_model_file_valid(CONFIG.eval_model_dir)
        return True
    elif CONFIG.use_which_deep_learning_framework == KaiwuDRLDefine.MODEL_TCNN:
        pass
    else:
        pass

    return False


def check_param():
    """
    在进程启动前进行检测参数合理性, 按照业务来区分
    """

    app_check_result = app_check_param()

    model_file_check_result = model_file_check()

    machine_device_check_result = machine_device_check(CONFIG.svr_name)

    return app_check_result and model_file_check_result and machine_device_check_result


def predictor_loop(actor_send_server, actor_recv_server, monitor_proxy):

    if CONFIG.actor_server_predict_server_different_queue:
        predictor_queues = []

    # actor上开启的predict进程列表
    predictor_process_objects = []

    """
    如果在大规模场景下, 因为model_file_sync进程只需要启动1个, 而actor的predict进程是多个的, 故这里需要采用下面步骤:
    1. model_file_sync进程先启动
    2. 将model_file_sync进程的对象句柄传入到actor的predict进程里进行使用
    """

    model_file_sync_wrapper = ModelFileSyncWrapper()
    model_file_sync_wrapper.init()

    # 根据配置文件conf/actor_conf.json找到本次使用的predictor类
    for i in range(CONFIG.actor_predict_process_num):
        if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
            predictor = AlgoConf[CONFIG.algo].predictor(actor_send_server, actor_recv_server)
        elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
            from kaiwudrl.server.actor.standard_predictor import StandardPredictor

            predictor = StandardPredictor(actor_send_server, actor_recv_server)
        else:
            # 如果是错误的情况下, 则进程直接退出
            print(f"can not get predictor, please check")
            sys.exit(1)

        predictor.set_index(i)
        predictor_process_objects.append(predictor)

        # predictor进程将queue注册到actor_server去
        if CONFIG.actor_server_predict_server_different_queue:

            """
            管道引起性能下降的原因是管道长度操作系统确定64KB, 且无法修改, 如果收方不及时的取走数据, 则发送方阻塞
            # 读方, 写方
            predict_conn, actor_server_conn = multiprocessing.Pipe(duplex=False)
            """

            predict_request_queue = multiprocessing.Queue(CONFIG.queue_size)

            predictor_queues.append(predict_request_queue)

            predictor.set_predict_request_queue_from_actor_server(predict_request_queue)

        if CONFIG.use_prometheus:
            predictor.set_monitor_proxy(monitor_proxy)

        predictor.set_model_file_sync_wrapper(model_file_sync_wrapper)

        # 是否采用流水线方式
        if CONFIG.pipeline_process_sync:

            # 启动actor_server和on_policy_predictor之间的predata和postdata进程
            from kaiwudrl.server.actor.actor_server_predata import ActorServerPreData

            actor_server_predata = ActorServerPreData(actor_recv_server, predictor)
            actor_server_predata.start()

            from kaiwudrl.server.actor.actor_server_postdata import ActorServerPostData

            actor_server_post_data = ActorServerPostData(actor_send_server, predictor)
            actor_server_post_data.start()

    """
    下面的逻辑主要是为在on-policy实现设计的, 因为单个actor只有1个端口由zmq使用, 且有多个预测进程,
    故这里不能以端口作为唯一标志, 而是需要采用预测进程内通信方式
    需要倒序输出, 此时第0号主进程作为主进程, 然后主进程是需要通知其他进程消息的
    """
    for i in range(CONFIG.actor_predict_process_num - 1, -1, -1):
        if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
            if i:
                # 采用pipe通信, 读方, 写方, 因为是同步的场景, 故设置为双向管道
                slave_conn, master_conn = multiprocessing.Pipe(duplex=True)
                predictor_process_objects[0].append_predictor_master_conn(master_conn)
                predictor_process_objects[i].set_predictor_slave_conn(slave_conn)

        # 启动预测进程
        predictor_process_objects[i].start()

    if CONFIG.actor_server_predict_server_different_queue:
        actor_recv_server.set_predict_request_queues(predictor_queues)

    if CONFIG.actor_server_async:
        actor_send_server.start()
        actor_recv_server.start()

    else:
        actor_send_server.start()


def gpu_machine_engine():
    """
    流程如下:
    1. 判定当前GPU机器类型
    2. 拷贝相关文件到对应目录
    """

    gpu_machine_type = get_gpu_machine_type()
    if gpu_machine_type is None:
        return True, gpu_machine_type

    # 因为有存在在不是GPU机器上运行的情况, 故这里不做强判断
    print(f"current gpu machine is {gpu_machine_type}")

    """
    代码里不能调用cp tensorrt的大文件, 容易出现异常, 故这里解决方案
    1. 打镜像时, 采用shell将相关的文件拷贝
    2. 在进程启动前, 采用shell将相关的文件拷贝
    """

    return True, gpu_machine_type


def main():
    """
    启动命令样例: python3 kaiwudrl/server/actor/actor.py --conf=conf/kaiwudrl/actor.toml
    """

    os.chdir(CONFIG.project_root)

    # 步骤1, 按照命令行来解析参数
    args = cmd_args_parse(KaiwuDRLDefine.SERVER_ACTOR)

    # 步骤2, 解析参数, 包括业务级别和算法级别
    proc_flags(args.conf)

    # 步骤3, 检测输入参数正确性
    if not check_param():
        print("conf param error, please check")
        return

    # 步骤4, 支持异构GPU, 主要是tensorrt
    ret, gpu_machine_type = gpu_machine_engine()
    if not ret:
        print(f"unsupport gpu_machine_type or error {gpu_machine_type} , please check")
        return

    # 步骤5, 处理信号
    register_signal()

    monitor_proxy = None
    if CONFIG.use_prometheus:
        from kaiwudrl.common.monitor.monitor_proxy_process import MonitorProxy

        monitor_proxy = MonitorProxy()
        monitor_proxy.start()

    # 步骤6, 启动ActorServer, libzmqops.so和interface.so的protoc版本不兼容, 需要在这里import

    """
    如果采用的是异步方式: actor_server从zmq_server的收发进程是2个独立的
    如果采用的是同步方式: actor_server从zmq_server的收发进程是1个
    """
    if CONFIG.actor_server_async:
        from kaiwudrl.server.actor.actor_server_async import ActorServerASync

        actor_server_async = ActorServerASync()
        actor_send_server = actor_server_async.get_actor_send_server()
        actor_recv_server = actor_server_async.get_actor_recv_server()
        if CONFIG.use_prometheus:
            actor_send_server.set_monitor_proxy(monitor_proxy)
            actor_recv_server.set_monitor_proxy(monitor_proxy)

    else:
        from kaiwudrl.server.actor.actor_server_sync import ActorServerSync

        actor_send_server = ActorServerSync()
        actor_recv_server = actor_send_server
        if CONFIG.use_prometheus:
            actor_send_server.set_monitor_proxy(monitor_proxy)

    # 步骤7, 开始预测
    predictor_loop(actor_send_server, actor_recv_server, monitor_proxy)


if __name__ == "__main__":
    sys.exit(main())
