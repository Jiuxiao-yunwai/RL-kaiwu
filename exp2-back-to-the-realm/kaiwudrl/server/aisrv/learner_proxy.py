#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file learner_proxy.py
# @brief
# @author kaiwu
# @date 2023-11-28


import os
import multiprocessing
import schedule
import datetime
import time
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from kaiwudrl.common.utils.common_func import (
    set_schedule_event,
    compress_data,
    get_uuid,
)
from kaiwudrl.common.ipc.zmq_util import ZmqClient
from kaiwudrl.common.monitor.monitor_proxy_process import MonitorProxy
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


class LearnerProxy(multiprocessing.Process):
    def __init__(self, policy_name, learner_addr, context) -> None:
        super(LearnerProxy, self).__init__()

        self.policy_name = policy_name

        """
        支持业务自定义和从alloc获取的情况
        1. 默认端口
        2. alloc服务下发的IP和端口
        3. 从配置文件读取的IP和端口
        """
        self.learner_addr = learner_addr[0]
        self.learner_port = learner_addr[1]

        """
        aisrv里主线程放进该Queue, learnproxy采用reverb client发送给reverb server
        """
        self.msg_queue = multiprocessing.Manager().Queue(CONFIG.queue_size)

        self.context = context

        # 根据不同的replay_buffer_type设置不同的对象
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            # reverb 工具类, aisrv上采用reverb client将数据发送给learn进程上的reverb server
            self.revervb_util = None
            self.reverb_table_names = None

        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            self.zmq_client = None
            self.client_id = None

            self.send_to_learner_err_cnt = 0
            self.send_to_learner_succ_cnt = 0

        else:
            pass

        # zmq_client, 某些管理流需要从aisrv传到learner上去执行, 故采用zmq, 因为与前面的self.zmq_client可能命名冲突, 然后定义其他名字
        self.client_id_for_learner = 0
        self.zmq_client_for_learner = None

        # 进程是否退出, 用于在对端异常条件下, 主动退出进程
        self.exit_flag = multiprocessing.Value("b", False)

    def put_data(self, slot_id, train_data):
        # 这里不需要指定是哪个battsvr和agent出现的数据, 故只是发送训练数据即可
        if self.msg_queue.full():
            return False

        self.msg_queue.put(train_data)
        return True

    # 返回参数是train_data
    def get_data(self):
        return self.msg_queue.get()

    # 返回reverb server的IP和端口
    def get_reverb_ip(self):
        return f"{self.learner_addr}:{self.learner_port}"

    def before_run(self):

        # 日志处理
        self.logger = KaiwuLogger()
        self.current_pid = os.getpid()
        self.logger.setLoggerFormat(
            f"/{CONFIG.svr_name}/learner_proxy_pid{self.current_pid}_log_"
            f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
            "learner_proxy",
        )

        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            from kaiwudrl.common.ipc.reverb_util import RevervbUtil

            # 必须放在这里赋值, 否则reverb client会卡住
            self.revervb_util = RevervbUtil(f"{self.learner_addr}:{self.learner_port}", self.logger)

            self.reverb_table_names = [
                "{}_{}".format(CONFIG.reverb_table_name, i) for i in range(int(CONFIG.reverb_table_size))
            ]
            self.logger.info(
                f"learner_proxy send reverb server use reverb, tables is {self.reverb_table_names}",
                g_not_server_label,
            )
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:

            """
            aisrv <--> learner之间, learner是支持多个aisrv的, 故learner需要知道各个aisrv的client_id, 故这里采用uuid方式
            """
            port = int(CONFIG.reverb_svr_port) - 1
            self.client_id = get_uuid()
            self.zmq_client = ZmqClient(
                str(self.client_id),
                self.learner_addr,
                str(port),
            )
            self.zmq_client.connect()

            self.logger.info(
                f"learner_proxy send reverb server use zmq, connect to {self.learner_addr}, "
                f"port is {port}, client_id is {self.client_id}",
                g_not_server_label,
            )
        else:
            pass

        # aisrv与learner之间的zmq管理流通信, 无论什么场景都需要使用
        self.client_id_for_learner = get_uuid()
        port = int(CONFIG.reverb_svr_port) - 2
        self.zmq_client_for_learner = ZmqClient(str(self.client_id_for_learner), self.learner_addr, port)
        self.zmq_client_for_learner.connect()
        self.logger.info(
            f"learner_proxy zmq client connect at {self.learner_addr}:{port} "
            f"with client_id {self.client_id_for_learner}",
            g_not_server_label,
        )

        # 访问普罗米修斯的类
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = MonitorProxy()
            self.monitor_proxy.start()

        self.send_to_sample_server_stat()

        self.process_run_count = 0

        # aisrv朝learner发送的最大样本大小
        self.max_sample_size = 0

        # 在before run最后打印启动成功日志
        self.logger.info(
            f"learner_proxy policy_name: {self.policy_name}, start success at pid {self.current_pid}",
            g_not_server_label,
        )

        return True

    def sample_server_stat(self):
        """
        获取发送样本的统计情况
        """
        if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
            succ_cnt, error_cnt = self.revervb_util.get_send_to_reverb_server_stat()
        elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
            succ_cnt, error_cnt = (
                self.send_to_learner_succ_cnt,
                self.send_to_learner_err_cnt,
            )
        else:
            succ_cnt, error_cnt = 0, 0

        if int(CONFIG.use_prometheus):

            # 注意msg_queue.qsize()可能出现异常报错, 故采用try-catch模式
            try:
                msg_queue_size = self.msg_queue.qsize()
            except Exception as e:
                msg_queue_size = 0

            monitor_data = {
                KaiwuDRLDefine.MONITOR_SENDTO_REVERB_SUCC_CNT: succ_cnt,
                KaiwuDRLDefine.MONITOR_SENDTO_REVERB_ERR_CNT: error_cnt,
                KaiwuDRLDefine.MONITOR_MAX_SAMPLE_SIZE: self.max_sample_size,
                KaiwuDRLDefine.MONITOR_AISRV_LEARNER_PROXY_QUEUE_LEN: msg_queue_size,
            }

            self.monitor_proxy.put_data({self.current_pid: monitor_data})

        self.logger.info(
            f"learner_proxy send reverb server stat, succ_cnt is {succ_cnt}, error_cnt is {error_cnt}",
            g_not_server_label,
        )

    # 定时器采用schedule, need pip install schedule
    def send_to_sample_server_stat(self):

        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.sample_server_stat)

    def run_once(self):

        # get sample data
        train_data = self.get_data()

        """
        根据不同的协议发送不同的操作:
        1. 如果是训练, 则按照样本发送的逻辑, 采用reverb, 如果不能采用reverb则采用zmq
        2. 如果是保留模型文件, 则按照发送模型文件的逻辑, 采用zmq
        """
        message_type = train_data.get(KaiwuDRLDefine.MESSAGE_TYPE)
        message_value = train_data.get(KaiwuDRLDefine.MESSAGE_VALUE)
        if message_type == KaiwuDRLDefine.MESSAGE_TRAIN:
            train_data = message_value.get("train_data")
            train_data_prioritized = message_value.get("train_data_prioritized")
            if train_data and train_data_prioritized:
                if CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_REVERB:
                    # use reverb client send sample data to reverb server
                    self.send_msg_use_reverb_client(train_data, train_data_prioritized)

                elif CONFIG.replay_buffer_type == KaiwuDRLDefine.REPLAY_BUFFER_TYPE_ZMQ:
                    self.send_msg_use_zmq_client(train_data)

                else:
                    pass

        elif message_type == KaiwuDRLDefine.MESSAGE_SAVE_MODEL:
            self.zmq_client_for_learner.send(train_data, binary=False)
            self.logger.info(
                f"learner_proxy send save_model data to learner",
                g_not_server_label,
            )

            result = self.zmq_client_for_learner.recv(binary=False)
            if (
                result
                and result.get(KaiwuDRLDefine.MESSAGE_TYPE) == KaiwuDRLDefine.MESSAGE_SAVE_MODEL
                and result.get(KaiwuDRLDefine.MESSAGE_VALUE)
            ):
                self.logger.info(
                    f"learner_proxy recv save_model data result from learner success",
                    g_not_server_label,
                )
            else:
                self.logger.error(
                    f"learner_proxy recv save_model data result from learner failed",
                    g_not_server_label,
                )

        elif message_type == KaiwuDRLDefine.MESSAGE_PROCESS_STOP:
            self.zmq_client_for_learner.send(train_data, binary=False)
            self.logger.info(
                f"learner_proxy send process_stop data to learner",
                g_not_server_label,
            )

            result = self.zmq_client_for_learner.recv(binary=False)
            if (
                result
                and result.get(KaiwuDRLDefine.MESSAGE_TYPE) == KaiwuDRLDefine.MESSAGE_PROCESS_STOP
                and result.get(KaiwuDRLDefine.MESSAGE_VALUE)
            ):
                self.logger.info(
                    f"learner_proxy recv process_stop data result from learner success",
                    g_not_server_label,
                )
            else:
                self.logger.error(
                    f"learner_proxy recv process_stop data result from learner failed",
                    g_not_server_label,
                )

        else:
            self.logger.error(
                f"learner_proxy recv un support message_type: {message_type}, please check",
                g_not_server_label,
            )

        # 启动记录发送成功失败的数目的定时器
        schedule.run_pending()

    # 进程停止函数
    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info("learner_proxy LearnerProxy stop success", g_not_server_label)

    def run(self) -> None:
        if not self.before_run():
            self.logger.error("learner_proxy before_run failed", g_not_server_label)
            return

        while not self.exit_flag.value:
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    time.sleep(CONFIG.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0

            except Exception as e:
                self.logger.exception(
                    f"learner_proxy run error: {str(e)}",
                    g_not_server_label,
                )

    # 发送样本时, 可以对样本进行预处理操作
    def before_send_train_data(self, train_data, train_data_prioritized):
        if not train_data:
            return

        # 暂时删除step维度
        if "s" in train_data.keys():
            del train_data["s"]

        # 增加lz4压缩
        # compress_train_data = lz4.block.compress(train_data, store_size=False)

    def before_send_train_data_simple(self, train_data, train_data_prioritized):
        """
        在发送样本开始时处理, 主要是压缩/解压缩, 主要是对train_data做检测, train_data_prioritized有些场景可能没有
        """
        if not train_data:
            return None

        # 增加lz4压缩
        compress_train_data = compress_data(train_data)
        return compress_train_data

    def after_send_train_data_simple(self, train_data):
        """
        在发送样本后的处理, 主要是做统计
        """
        if not train_data:
            return

        # 更新最大样本大小
        input_datas_list = train_data
        sample_size = 0
        for agent in input_datas_list:
            if CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_NORMAL:
                sample_size += agent["input_datas"].nbytes
            elif CONFIG.framework_integration_patterns == KaiwuDRLDefine.INTEGRATION_PATTERNS_STANDARD:
                sample_size += agent.nbytes
            else:
                pass

        # 转换成MB
        sample_size = round(sample_size / (1024 * 1024), 2)

        # 更新最大样本大小
        if sample_size > self.max_sample_size:
            self.max_sample_size = sample_size

    def send_msg_use_zmq_client(self, train_data):
        """
        采用zmq_client发送请求
        """
        if not train_data:
            return False

        train_data_size = len(train_data)

        try:
            data = self.before_send_train_data_simple(train_data, None)
            if data:
                self.zmq_client.send(data, binary=True)

                # 接收回包
                result = self.zmq_client.recv(binary=False)
                if (
                    result
                    and result.get(KaiwuDRLDefine.MESSAGE_TYPE) == KaiwuDRLDefine.MESSAGE_SEND_SAMPLE
                    and result.get(KaiwuDRLDefine.MESSAGE_VALUE)
                ):
                    self.send_to_learner_succ_cnt += train_data_size
                else:
                    self.send_to_learner_err_cnt += train_data_size
            else:
                self.send_to_learner_err_cnt += train_data_size

        except Exception as e:
            self.logger.exception(
                f"learner_proxy send to zmq_server {self.get_reverb_ip()} failed, "
                f"client_id is {self.client_id}, run error: {str(e)}, ",
                g_not_server_label,
            )
            self.send_to_learner_err_cnt += train_data_size

    # use reverb client send msq to reverb server
    def send_msg_use_reverb_client(self, train_data, train_data_prioritized):
        if not train_data:
            return

        # 发给reverb server, 没有进行样本发送前的处理是由于reverb暂时不支持lz4压缩/解压缩
        self.revervb_util.write_to_reverb_server_simple(self.reverb_table_names, train_data, train_data_prioritized)

        self.after_send_train_data_simple(train_data)

    # 注意，这里实际是在aisrv进程上的learner_proxy，而不是learner_proxy进程
    def get_training_metrics(self, msg):

        # 由于是多个进程可能调用, 故每个进程传递的client_id不一样的, client_id设置为self.client_id_for_learner + os.getpid()
        if self.zmq_client_for_learner is None:
            port = int(CONFIG.reverb_svr_port) - 2
            self.zmq_client_for_learner = ZmqClient(
                str(self.client_id_for_learner + os.getpid()), self.learner_addr, port
            )
            self.zmq_client_for_learner.connect()

        message_type = msg.get(KaiwuDRLDefine.MESSAGE_TYPE)
        message_value = msg.get(KaiwuDRLDefine.MESSAGE_VALUE)

        if message_type == KaiwuDRLDefine.MESSAGE_GET_TRAINING_METRICS:
            self.zmq_client_for_learner.send(msg, binary=False)

            result = self.zmq_client_for_learner.recv(binary=False)
            training_metrics = result.get(KaiwuDRLDefine.MESSAGE_VALUE)
        return training_metrics
