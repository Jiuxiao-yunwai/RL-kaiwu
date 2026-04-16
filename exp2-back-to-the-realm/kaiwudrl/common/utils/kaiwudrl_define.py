#!/usr/bin/env python
# -*- coding: utf-8 -*-


# @file kaiwudrl_define.py
# @brief
# @author kaiwu
# @date 2023-11-28


class KaiwuDRLDefine(object):
    """
    KaiwuDRL的定义文件, 包含所有定义
    """

    # 下面是aisrv、actor、learner的扩缩容标志位
    PROCESS_ADD = "add"
    PROCESS_REDUCE = "reduce"

    # 下面是aisrv <--> actor之间的消息格式定义
    COMPOSE_ID_SIZE = 5
    CLIENT_ID_SIZE = 1

    # 进程服务名字, 其中main是为了便于在七彩石上管理, 不是实际存在的进程名
    SERVER_AISRV = "aisrv"
    SERVER_ACTOR = "actor"
    SERVER_LEARNER = "learner"
    SERVER_BATTLE = "client"
    SERVER_KAIWU_ENV = "kaiwu_env"
    SERVER_MAIN = "main"
    SERVER_MODELPOOL = "modelpool"
    SERVER_MODELPOOL_PROXY = "modelpool_proxy"
    SERVER_PYTHON = "python3"
    SERVER_CLIENT = "client"
    SERVER_BATTLE_SRV = "battlesrv"
    TRAIN_TEST_CMDLINE = "train_test.py"

    # 进程名字, 区分下面的
    PROCESS_AISRV = "aisrv"
    PROCESS_ACTOR_PROXY_LOCAL = "actor_proxy_local"

    # 记录aisrv <--> actor之间的zmq连接, 如果后期优化到了C++层面, 则可以去掉
    CLIENT_ID_TENSOR = "KAIWU_CLIENT_ID"
    # 标志单个aisrv上不同的agent id, slot id, message_id的请求
    COMPOSE_ID_TENSOR = "KAIWU_COMPOSE_ID"

    # checkpoint文件
    CHECK_POINT_FILE = "checkpoint"
    KAIWU_CHECK_POINT_FILE = "kaiwu_checkpoint"
    KAIWU_MODEL_CKPT = "model.ckpt"
    KAIWU_MODEL_ID_LIST = "id_list"
    KAIWU_MODEDL_WIGHT = "model.wight"
    KAIWU_ONNX_FILE = "onnx"
    KAIWU_PB_FILE = "model.pb"

    # KaiwuDRL支持的不同modelwraper
    MODEL_TENSORFLOW_SIMPLE = "tensorflow_simple"
    MODEL_TENSORFLOW_COMPLEX = "tensorflow_complex"
    MODEL_PYTORCH = "pytorch"
    MODEL_TCNN = "tcnn"
    MODEL_TENSORRT = "tensorrt"
    MODEL_NO_DEEP_LEARNING_FRAMEWORK = "local"

    # KaiwuDRL支持的统计指标, 需要对齐指标名字

    # actor
    MONITOR_ACTOR_PREDICT_SUCC_CNT = "actor_predict_succ_cnt"
    MONITOR_ACTOR_FROM_ZMQ_QUEUE_SIZE = "actor_from_zmq_queue_size"
    MONITOR_TENSORRT_REFIT_SUC_CNT = "tensorrt_refit_suc_cnt"
    MONITOR_TENSORRT_REFIT_ERR_CNT = "tensorrt_refit_err_cnt"
    MONITOR_ACTOR_FROM_ZMQ_QUEUE_COST_TIME_MS = "actor_from_zmq_queue_cost_time_ms"
    MONITOR_ACTOR_BATCH_PREDICT_COST_TIME_MS = "actor_batch_predict_cost_time_ms"
    MONITOR_PUSH_TO_CUDA_QUEUE_COST_TIME_MS = "push_to_cuda_queue_cost_time_ms"
    MONITOR_ACTOR_SENDTO_AISRV_SUCC_CNT = "send_to_aisrv_suc_cnt"
    MONITOR_ACTOR_SENDTO_AISRV_ERROR_CNT = "send_to_aisrv_err_cnt"
    MONITOR_ACTOR_RECEIVEFROM_AISRV_SUCC_CNT = "recv_from_aisrv_suc_cnt"
    MONITOR_ACTOR_RECEIVEFROM_AISRV_ERROR_CNT = "recv_from_aisrv_err_cnt"
    MONITOR_ACTOR_SENDTO_AISRV_BATCH_COST_TIME_MS = "actor_send_to_aisrv_batch_cost_time_ms"
    PULL_FROM_MODEL_POOL_SUCC_CNT = "pull_from_model_pool_succ_cnt"
    PULL_FROM_MODEL_POOL_ERR_CNT = "pull_from_model_pool_err_cnt"
    ACTOR_TENSORRT_CPU2GPU_SUCC_CNT = "actor_tensorrt_cpu_send_to_gpu_succ_cnt"
    ACTOR_TENSORRT_CPU2GPU_ERR_CNT = "actor_tensorrt_cpu_send_to_gpu_error_cnt"
    ACTOR_TENSORRT_GPU2CPU_SUCC_CNT = "actor_tensorrt_gpu_send_to_cpu_succ_cnt"
    ACTOR_TENSORRT_GPU2CPU_ERR_CNT = "actor_tensorrt_gpu_send_to_cpu_error_cnt"
    MONITOR_ACTOR_SERVER_QUEUE_FULL_CNT = "actor_server_queue_full_cnt"
    MONITOR_ACTOR_MAX_COMPRESS_TIME = "actor_max_compress_time"
    MONITOR_ACTOR_MAX_DECOMPRESS_TIME = "actor_max_decompress_time"
    MONITOR_ACTOR_MAX_COMPRESS_SIZE = "actor_max_compress_size"
    MONITOR_ACTOR_PREDICT_REQUEST_QUEUE_SIZE = "predict_request_queue_size"
    MONITOR_ACTOR_PREDICT_RESULT_QUEUE_SIZE = "predict_result_queue_size"
    MONITOR_ACTOR_SERVER_REQUEST_QUEUE_SIZE = "actor_server_request_queue_size"
    MONITOR_ACTOR_SERVER_RESULT_QUEUE_SIZE = "actor_server_result_queue_size"
    MONITOR_ACTOR_GET_AND_PREDICT_COST_MS = "local_get_and_predict_cost_time_ms"
    # actor/actor上aisrv的TCP数目
    ACTOR_TCP_AISRV = "actor_tcp_aisrv"
    # 在使用TesnorFlow/TensorRT时, 可能会出现refit时大时延, 故获取最大值
    ACTOR_LOAD_LAST_MODEL_COST_MS = "actor_load_last_model_cost_ms"
    ACTOR_LOAD_LAST_MODEL_SUCC_CNT = "actor_load_last_model_succ_cnt"
    ACTOR_LOAD_LAST_MODEL_ERROR_CNT = "actor_load_last_model_error_cnt"
    # 下面是on-policy的actor统计告警指标
    ON_POLICY_PULL_FROM_MODELPOOL_ERROR_CNT = "on_policy_pull_from_modelpool_error_cnt"
    ON_POLICY_PULL_FROM_MODELPOOL_SUCCESS_CNT = "on_policy_pull_from_modelpool_success_cnt"
    ON_POLICY_ACTOR_CHANGE_MODEL_VERSION_ERROR_COUNT = "actor_change_model_version_error_count"
    ON_POLICY_ACTOR_CHANGE_MODEL_VERSION_SUCCESS_COUNT = "actor_change_model_version_success_count"

    # learner
    MONITOR_REVERB_READY_SIZE = "reverb_ready_size"
    MONITOR_TRAIN_SUCCESS_CNT = "train_success_cnt"
    MONITOR_TRAIN_GLOBAL_STEP = "train_global_step"
    MONITOR_BATCH_TRAIN_COST_TIME_MS = "batch_train_cost_time_ms"
    PUSH_TO_COS_SUCC_CNT = "push_to_cos_succ_cnt"
    PUSH_TO_COS_ERR_CNT = "push_to_cos_err_cnt"
    PUSH_TO_MODEL_POOL_SUCC_CNT = "push_to_model_pool_succ_cnt"
    PUSH_TO_MODEL_POOL_ERR_CNT = "push_to_model_pool_err_cnt"
    MONITOR_LEARNER_ZMQ_REVERB_QUEUE_LEN = "learner_zmq_reverb_queue_len"
    # actor/actor上aisrv的TCP数目
    LEARNER_TCP_AISRV = "learner_tcp_aisrv"
    # 下面是on-policy的learner统计告警指标
    ON_POLICY_PUSH_TO_MODELPOOL_ERROR_CNT = "on_policy_push_to_modelpool_error_cnt"
    ON_POLICY_PUSH_TO_MODELPOOL_SUCCESS_CNT = "on_policy_push_to_modelpool_success_cnt"
    ON_POLICY_LEARNER_RECV_AISRV_ERROR_CNT = "on_policy_learner_recv_aisrv_error_cnt"
    ON_POLICY_LEARNER_RECV_AISRV_SUCCESS_CNT = "on_policy_learner_recv_aisrv_success_cnt"
    ON_POLICY_LEARNER_RECV_ACTOR_ERROR_CNT = "on_policy_learner_recv_actor_error_cnt"
    ON_POLICY_LEARNER_RECV_ACTOR_SUCCESS_CNT = "on_policy_learner_recv_actor_success_cnt"

    # aisrv
    MONITOR_SENDTO_REVERB_SUCC_CNT = "send_to_reverb_succ_cnt"
    MONITOR_SENDTO_REVERB_ERR_CNT = "send_to_reverb_err_cnt"
    MONITOR_SEND_TO_LEARNER_PROXY_SUC_CNT = "send_to_learner_suc_cnt"
    MONITOR_SEND_TO_LEARNER_PROXY_ERR_CNT = "send_to_learner_err_cnt"
    MONITOR_MAX_SAMPLE_SIZE = "max_sample_size"
    MONITOR_AISRV_SENDTO_ACTOR_SUCC_CNT = "send_to_actor_suc_cnt"
    MONITOR_AISRV_SENDTO_ACTOR_ERROR_CNT = "send_to_actor_err_cnt"
    MONITOR_AISRV_RECVFROM_ACTOR_SUCC_CNT = "recv_from_actor_suc_cnt"
    MONITOR_AISRV_RECVFROM_ACTOR_ERROR_CNT = "recv_from_actor_err_cnt"
    MONITOR_AISRV_ACTOR_PROXY_QUEUE_LEN = "aisrv_actor_proxy_queue_len"
    MONITOR_AISRV_LEARNER_PROXY_QUEUE_LEN = "aisrv_learner_proxy_queue_len"
    MONITOR_AISRV_MAX_COMPRESS_TIME = "aisrv_max_compress_time"
    MONITOR_AISRV_MAX_DECOMPRESS_TIME = "aisrv_max_decompress_time"
    MONITOR_AISRV_MAX_COMPRESS_SIZE = "aisrv_max_compress_size"
    MONITOR_AISRV_ACTOR_MEAN_TIME_COST = "aisrv_actor_mean_time_cost"
    MONITOR_AISRV_ACTOR_MAX_TIME_COST = "aisrv_actor_max_time_cost"
    MONITOR_AISRV_ACTOR_TIMEOUT_GT = "aisrv_actor_timeout_gt_"
    # actor/actor上aisrv的TCP数目
    AISRV_TCP_BATTLESRV = "aisrv_tcp_battlesrv"
    MONITOR_AISRV_SEND_TO_BATTLESRV_SUC_CNT = "send_to_battlesrv_suc_cnt"
    MONITOR_AISRV_SEND_TO_BATTLESRV_ERR_CNT = "send_to_battlesrv_err_cnt"
    MONITOR_AISRV_RECV_FROM_BATTLESRV_SUC_CNT = "recv_from_battlesrv_suc_cnt"
    MONITOR_AISRV_RECV_FROM_BATTLESRV_ERR_CNT = "recv_from_battlesrv_err_cnt"
    MONITOR_AISRV_MAX_PROCESSING_TIME = "max_processing_time"
    # 下面是on-policy的aisrv统计告警指标
    MONITOR_AISRV_ON_POLICY_KAIWU_RL_HELPER_PAUSE_ERROR_COUNT = "kaiwu_rl_helper_pause_error_count"
    MONITOR_AISRV_ON_POLICY_KAIWU_RL_HELPER_PAUSE_SUCCESS_COUNT = "kaiwu_rl_helper_pause_success_count"
    MONITOR_AISRV_ON_POLICY_KAIWU_RL_HELPER_CONTINUE_ERROR_COUNT = "kaiwu_rl_helper_continue_error_count"
    MONITOR_AISRV_ON_POLICY_KAIWU_RL_HELPER_CONTINUE_SUCCESS_COUNT = "kaiwu_rl_helper_continue_success_count"
    MONITOR_AISRV_ON_POLICY_AISRV_CHANGE_MODEL_VERSION_ERROR_COUNT = "aisrv_change_model_version_error_count"
    MONITOR_AISRV_ON_POLICY_AISRV_CHANGE_MODEL_VERSION_SUCCESS_COUNT = "aisrv_change_model_version_success_count"

    # COS桶下的key名字
    COS_BUCKET_KEY = "kaiwu_drl_models/"

    # 从COS下载的最新的文件的名字
    COS_LAST_MODEL_FILE = "from_cos.tar.gz"
    TAR_GZ = "tar.gz"
    TAR = "tar"

    # 机器上关于Model的目录路径
    CKPT_DIR = "ckpt_dir"
    RESTORE_DIR = "restore_dir"
    SUMMARY_DIR = "summary_dir"
    PB_MODEL_DIR = "pb_model_dir"

    # aisrv和actor之间通信方式
    COMMUNICATION_WAY_ZMQ = "zmq"
    COMMUNICATION_WAY_ZMQ_OPS = "zmq-ops"

    # actor_server采用的方式
    RUN_AS_COROUTINE = "coroutine"
    RUN_AS_DIRECT = "direct"
    RUN_AS_THREAD = "thread"
    RUN_AS_GEVENT = "gevent"

    # 业务名称
    APP_GYM = "gym"
    APP_SGAME_1V1 = "sgame_1v1"
    APP_SGAME_5V5 = "sgame_5v5"

    # KaiwuDRL支持的GPU机器类型
    GPU_MACHINE_A100 = "A100"
    GPU_MACHINE_V100 = "V100"
    GPU_MACHINE_T4 = "T4"
    GPU_MACHINE_P100 = "P100"
    GPU_MACHINE_CPU = "CPU"

    # KaiwuDRL支持的压缩/解压缩算法
    COMPRESS_DECOMPRESS_ALGORITHMS_LZ4 = "lz4"
    COMPRESS_DECOMPRESS_ALGORITHMS_ZSTD = "zstd"

    # python和C++使用共享内存通信, 默认不需要修改
    SHMNAME_NAME = "G6SHMNAME"
    SHMNAME_NAME_VALUE = "KaiwuDRL"

    # actor在C++端生成的二进制名字
    ACTOR_CPP_SERVER = "actor_cpp_server"

    # 业务训练指标
    REWARD_VALUE = "reward_value"
    SAMPLE_PRODUCTION_AND_CONSUMPTION_RATIO = "sample_production_and_consumption_ratio"
    SAMPLE_PRODUCT_RATE = "sample_product_rate"
    SAMPLE_CONSUME_RATE = "sample_consume_rate"
    SAMPLE_RECEIVE_CNT = "sample_receive_cnt"

    # aisrv和actor之间采用的通信协议
    PROTOCOL_PICKLE = "pickle"
    PROTOCOL_PROTOBUF = "protobuf"

    # 文件结束标志的文件名
    FILE_FINISH_NAME = "FINISH"
    FILE_FINISH_OLD_NAME = "FINISH_OLD"

    # 运行模式, train, eval, exam
    RUN_MODEL_TRAIN = "train"
    RUN_MODEL_EVAL = "eval"
    RUN_MODEL_EXAM = "exam"

    # aisrv的运行框架, 包括socketserver, kaiwudrl, kaiwu_env, 其中socketserver, kaiwu_env是python的, kaiwudrl是C++的
    AISRV_FRAMEWORK_SOCKETSERVER = "socketserver"
    AISRV_FRAMEWORK_KAIWUDRL = "kaiwudrl"
    AISRV_FRAMEWORK_KAIWU_ENV = "kaiwu_env"

    # 字符串编码
    UTF_8 = "utf-8"
    GBK = "gbk"

    # configparser的默认配置
    CONFIG_DEFAULT_INT = 0
    CONFIG_DEFAULT_FLOAT = 0.0
    CONFIG_DEFAULT_BOOL = False
    CONFIG_DEFAULT_STRING = ""

    # 支持的算法on-policy, off-policy
    ALGORITHM_ON_POLICY = "on-policy"
    ALGORITHM_OFF_POLICY = "off-policy"

    # 支持on-policy的方式, step, episode, time_interval
    ALGORITHM_ON_POLICY_WAY_STEP = "step"
    ALGORITHM_ON_POLICY_WAY_EPISODE = "episode"
    ALGORITHM_ON_POLICY_WAY_TIME_INTERVAL = "time_interval"

    # 本机IP的字符串
    LOCAL_HOST_IP = "127.0.0.1"
    ALL_HOST_IP = "0.0.0.0"

    # 下面是on-policy里面的消息类型和值
    MESSAGE_TYPE = "message_type"
    MESSAGE_VALUE = "message_value"

    ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_REQUEST = "model_version_change_request"
    ON_POLICY_MESSAGE_MODEL_VERSION_CHANGE_RESPONSE = "model_version_change_response"
    ON_POLICY_MESSAGE_ASK_LEARNER_TO_EXECUTE_ON_POLICY_PROCESS_REQUEST = (
        "ask_learner_to_execute_on_policy_process_request"
    )
    ON_POLICY_MESSAGE_ASK_LEARNER_TO_EXECUTE_ON_POLICY_PROCESS_RESPONSE = (
        "ask_learner_to_execute_on_policy_process_response"
    )
    ON_POLICY_MESSAGE_HEARTBEAT_REQUEST = "heartbeat_request"
    ON_POLICY_MESSAGE_HEARTBEAT_RESPONSE = "heartbeat_response"

    # 下面是aisrv朝actor发送的协议, 包括数据流, 管理流
    MESSAGE_PREDICT = "predict"
    MESSAGE_EXPLOIT = "exploit"
    MESSAGE_LOAD_MODEL = "load_model"

    # 下面是aisrv朝learner发送的协议, 包括数据流, 管理流
    MESSAGE_SEND_SAMPLE = "send_sample"
    MESSAGE_TRAIN = "train"
    MESSAGE_SAVE_MODEL = "save_model"
    MESSAGE_PROCESS_STOP = "process_stop"
    MESSAGE_GET_TRAINING_METRICS = "get_training_metrics"

    # 加载model文件时ID的定义, 其中latest, random是特别的
    ID_LATEST = "latest"
    ID_RANDOM = "random"

    # 预测是放在actor远程还是aisrv本地, 小规模场景建议是aisrv本地, 大规模场景和小规模场景都可以使用的actor远程
    PREDICT_AS_LOCAL = "local"
    PREDICT_AS_REMOTE = "remote"

    # 普罗米修斯的push/pull模式
    USE_PROMETHEUS_WAY_PULL = "pull"
    USE_PROMETHEUS_WAY_PUSH = "push"

    # 解决actor的predict进程和model_file_sync进程之间采用的文件锁
    LOCK_READ_FILE = "lock_read_file"
    LOCK_WRITE_FILE = "lock_write_file"

    # 框架接入模式
    INTEGRATION_PATTERNS_STANDARD = "standard"
    INTEGRATION_PATTERNS_NORMAL = "normal"

    # wrapper三种模式
    WRAPPER_REMOTE = "remote"
    WRAPPER_LOCAL = "local"
    WRAPPER_NONE = "none"

    # KaiwuDRL的model文件的magic
    KAIWUDRL_MODEL_FILE_MAGIC = "tencent_kaiwu"
    KAIWUDRL_MODEL_FILE_JSON_FILE_NAME = "kaiwu"

    # 样本处理的replay_buffer组件, 支持reverb, tf_uniform, zmq等, 默认为reverb
    REPLAY_BUFFER_TYPE_REVERB = "reverb"
    REPLAY_BUFFER_TYPE_TF_UNIFORM = "tf_uniform"
    REPLAY_BUFFER_TYPE_ZMQ = "zmq"

    # actor/learner之间的model文件传输组件, 支持modelpool等
    CKPT_SYNC_WAY_MODELPOOL = "modelpool"

    # save_model和load_model会从不同的方式调用
    SAVE_OR_LOAD_MODEL_By_USER = "user"
    SAVE_OR_LOAD_MODEL_By_FRAMEWORK = "framework"
    SAVE_OR_LOAD_MODEL_BY_SIGTERM = "sigterm"

    # 机器上machine device类型
    MACHINE_DEVICE_CPU = "CPU"
    MACHINE_DEVICE_GPU = "GPU"
    MACHINE_DEVICE_NPU = "NPU"

    # 容器退出错误码
    DOCKER_EXIT_CODE_SUCCESS = 0
    DOCKER_EXIT_CODE_ERROR = 1
    # 目前设置的timeout和error场景一样
    DOCKER_EXIT_CODE_TIMEOUT = 1

    # 运行的部署环境, 单机客户端, 集群, 平台等
    DEPLOYMENT_PLATFORMS_CLIENT = "client"
    DEPLOYMENT_PLATFORMS_CLUSTER = "cluster"

    # 在aisrv处理多个agent时是进程模式还是其他模式
    MULTI_AGENT_PREDICT_SEQUENTIAL = "sequential"
    MULTI_AGENT_PREDICT_PARALLEL = "parallel"

    # 序列化和反序列化方法
    SERIALIZE_TYPE_MSGPACK = "msgpack"
    SERIALIZE_TYPE_DILL = "dill"
