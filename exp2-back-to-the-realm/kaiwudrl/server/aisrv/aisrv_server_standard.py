#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file aisrv_server_standard.py
# @brief
# @author kaiwu
# @date 2023-11-28


import json
import multiprocessing
import datetime
import os
import schedule
import time
import copy
import flatbuffers
import collections
from kaiwudrl.common.monitor.monitor_proxy_process import MonitorProxy
from kaiwudrl.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.alloc.alloc_proxy import AllocProxy, AllocUtils
from kaiwudrl.common.utils.common_func import (
    is_list_eq,
    list_diff,
    set_schedule_event,
    actor_learner_aisrv_count,
    get_host_ip,
    Context,
    make_single_dir,
    python_exec_shell,
    register_sigterm_handler,
)
from kaiwudrl.common.ipc.zmq_util import ZmqServer
from kaiwudrl.common.utils.slots import Slots
from kaiwudrl.common.config.app_conf import AppConf
from kaiwudrl.server.aisrv.flatbuffer.kaiwu_msg_helper import KaiwuMsgHelper
from kaiwudrl.server.aisrv.msg_buff import MsgBuff
from kaiwudrl.server.aisrv.flatbuffer.kaiwu_msg import ReqMsg
from kaiwudrl.common.checkpoint.model_file_sync_wrapper import ModelFileSyncWrapper


class AiServer(multiprocessing.Process):
    def __init__(
        self,
    ) -> None:
        super(AiServer, self).__init__()

        # 设置日志Log配置
        self.logger = KaiwuLogger()
        self.current_pid = os.getpid()
        self.logger.setLoggerFormat(
            f"/{CONFIG.svr_name}/aiserver_pid{self.current_pid}_log_"
            f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
            "AiServer",
        )

        # 获取本机IP
        self.host = get_host_ip()

    # 业务的上报, aisrv主线程
    def aisrv_main_process_stat(self):
        if int(CONFIG.use_prometheus):
            monitor_data = {}

            # 无论什么场景下都需要上报的监控值
            monitor_data[KaiwuDRLDefine.AISRV_TCP_BATTLESRV] = actor_learner_aisrv_count(self.host, CONFIG.svr_name)

            self.monitor_proxy.put_data({self.current_pid: monitor_data})

    def check_param(self):
        """
        进程启动前配置参数检测
        1. 规则1, 如果是设置了self-play模式, 但是app文件里设置的policy是1个, 则报错
        2. 规则2, 如果是设置了非self-play模式, 但是app文件里设置的policy是2个, 则报错
        3. 规则3, 如果是设置了self-play模式, 但是aisrv.toml文件里设置的actor_addrs/learner_addrs的policy是1个, 则报错
        4. 规则4, 如果是设置了非self-play模式, 但是aisrv.toml文件里设置的actor_addrs/learner_addrs的policy是2个, 则报错
        5. 规则5, 如果是设置了on-policy模式, 但是同时是小规模模式, 则报错
        """

        actor_addrs = CONFIG.actor_addrs
        learner_addrs = CONFIG.learner_addrs

        if int(CONFIG.self_play):
            if len(AppConf[CONFIG.app].policies) == 1:
                self.logger.error(f"AiServer self-play模式, 但是配置的policy维度为1, 请修改配置后重启进程")
                return False

            if len(actor_addrs) == 1 or len(learner_addrs) == 1:
                self.logger.error(
                    f"AiServer self-play模式, 但是配置的aisrv.toml的actor_addrs/learner_addrs的policy维度为1, 请修改配置后重启进程"
                )
                return False

        else:
            if len(AppConf[CONFIG.app].policies) == 2:
                self.logger.error(f"AiServer 非self-play模式, 但是配置的policy维度为2, 请修改配置后重启进程")
                return False

            if len(actor_addrs) == 2 or len(learner_addrs) == 2:
                self.logger.error(
                    f"AiServer 非self-play模式, 但是配置的aisrv.toml的actor_addrs/learner_addrs的policy维度为2, 请修改配置后重启进程"
                )
                return False

        return True

    # aisrv在处理actor和learner的动态扩缩容逻辑
    def aisrv_with_new_actor_learner_change(self):
        if not CONFIG.actor_learner_expansion:
            return

    # aisrv从alloc服务获取kaiwu_env的IP地址
    def get_kaiwu_env_ip_from_alloc(self):
        # 重试CONFIG.socket_retry_times次, 每次sleep CONFIG.alloc_process_per_seconds获取actor和learner地址
        retry_num = 0
        kaiwu_env_address = []
        while retry_num < CONFIG.socket_retry_times:
            kaiwu_env_address = self.alloc_util.get_kaiwu_env_ip(CONFIG.set_name, CONFIG.self_play_set_name)
            if not kaiwu_env_address:
                time.sleep(int(CONFIG.socket_timeout))
                retry_num += 1
            else:
                break

        # 如果超过重试次数, 则放弃从alloc获取地址, 从本地配置文件启动
        if retry_num >= CONFIG.socket_retry_times:
            self.logger.error(
                f"AiServer server get kaiwu_env address retry times more than "
                f"{CONFIG.socket_retry_times}, will start with configure file"
            )
            return None

        return kaiwu_env_address

    # aisrv朝alloc服务的注册函数, 需要先注册才能拉取地址
    def aisrv_registry_to_alloc(self):
        # 需要先注册本地aisrv地址后, 再拉取actor, learner地址
        code, msg = self.alloc_util.registry()
        if code:
            self.logger.info(f"AiServer alloc interact registry success")
            return True
        else:
            self.logger.error(f"AiServer alloc interact registry fail, will retry next time, error_code is {msg}")
            return False

    def get_actor_learner_ip_from_alloc(self):
        """
        增加aisrv从alloc获取IP地址的逻辑, 为了和以前从配置文件加载的方式结合, 采用操作步骤如下:
        1. 每隔CONFIG.alloc_process_per_seconds拉取, 最大CONFIG.socket_retry_times次后报错, 当返回有具体的数据则跳出循环
        2. 针对返回的actor和learner地址, 修改内存和配置文件里的值
        """

        if CONFIG.predict_local_or_remote == KaiwuDRLDefine.PREDICT_AS_REMOTE:
            # 重试CONFIG.socket_retry_times次, 每次sleep CONFIG.alloc_process_per_seconds获取actor和learner地址
            retry_num = 0
            while retry_num < CONFIG.socket_retry_times:
                if not int(CONFIG.self_play):
                    (
                        actor_address,
                        learner_address,
                        _,
                        _,
                    ) = self.alloc_util.get_actor_learner_ip(CONFIG.set_name, None)
                    if not actor_address or not learner_address:
                        time.sleep(int(CONFIG.socket_timeout))
                        retry_num += 1
                    else:
                        break
                else:
                    # 对于self-play模式, self_play_set下的learner不是强要求的
                    (
                        self_play_actor_address,
                        self_play_old_actor_address,
                        self_play_learner_address,
                        self_play_old_learner_address,
                    ) = self.alloc_util.get_actor_learner_ip(CONFIG.set_name, CONFIG.self_play_set_name)

                    if not self_play_actor_address or not self_play_learner_address or not self_play_old_actor_address:
                        time.sleep(int(CONFIG.socket_timeout))
                        retry_num += 1
                    else:
                        break

            # 如果超过重试次数, 则放弃从alloc获取地址, 从本地配置文件启动
            if retry_num >= CONFIG.socket_retry_times:
                self.logger.error(
                    f"AiServer server get actor and learner address retry times more than "
                    f"{CONFIG.socket_retry_times}, will start with configure file"
                )
                return

            # 修改配置文件
            if not int(CONFIG.self_play):
                self.change_configure_content(actor_address, learner_address, None, None, None, None)
            else:
                self.change_configure_content(
                    None,
                    None,
                    self_play_actor_address,
                    self_play_learner_address,
                    self_play_old_actor_address,
                    self_play_old_learner_address,
                )
        else:
            # 重试CONFIG.socket_retry_times次, 每次sleep CONFIG.alloc_process_per_seconds获取learner地址
            retry_num = 0
            while retry_num < CONFIG.socket_retry_times:
                if not int(CONFIG.self_play):
                    learner_address, _ = self.alloc_util.get_learner_ip(CONFIG.set_name, None)
                    if not learner_address:
                        time.sleep(int(CONFIG.socket_timeout))
                        retry_num += 1
                    else:
                        break
                else:
                    # 对于self-play模式, self_play_set下的learner不是强要求的
                    (
                        self_play_learner_address,
                        self_play_old_learner_address,
                    ) = self.alloc_util.get_learner_ip(CONFIG.set_name, CONFIG.self_play_set_name)
                    if not self_play_learner_address:
                        time.sleep(int(CONFIG.socket_timeout))
                        retry_num += 1
                    else:
                        break

            # 如果超过重试次数, 则放弃从alloc获取地址, 从本地配置文件启动
            if retry_num >= CONFIG.socket_retry_times:
                self.logger.error(
                    f"AiServer server get actor and learner address retry times more than "
                    f"{CONFIG.socket_retry_times}, will start with configure file"
                )
                return

            # 修改配置文件
            if not int(CONFIG.self_play):
                # 此处需要针对设置值
                actor_address = [f"{KaiwuDRLDefine.LOCAL_HOST_IP}:{CONFIG.zmq_server_port}"]
                self.change_configure_content(actor_address, learner_address, None, None, None, None)
            else:
                self_play_actor_address = [f"{KaiwuDRLDefine.LOCAL_HOST_IP}:{CONFIG.zmq_server_port}"]
                self_play_old_actor_address = [f"{KaiwuDRLDefine.LOCAL_HOST_IP}:{CONFIG.zmq_server_port}"]
                self.change_configure_content(
                    None,
                    None,
                    self_play_actor_address,
                    self_play_learner_address,
                    self_play_old_actor_address,
                    self_play_old_learner_address,
                )

    # C++ 常驻进程进程配置文件修改
    def save_to_file(self, process_name, to_change_key_values):
        if not to_change_key_values or not process_name:
            return

        # 先删除actor_addrs,learner_addrs,self_play, actor_proxy_num, learner_proxy_num
        cmd = (
            f"sed -i '/actor_addrs/d' {CONFIG.cpp_aisrv_configure}; "
            f"sed -i '/learner_addrs/d' {CONFIG.cpp_aisrv_configure}; "
            f"sed -i '/self_play/d' {CONFIG.cpp_aisrv_configure}; "
            f"sed -i '/actor_proxy_num/d' {CONFIG.cpp_aisrv_configure}; "
            f"sed -i '/learner_proxy_num/d' {CONFIG.cpp_aisrv_configure};"
        )
        result_code, result_str = python_exec_shell(cmd)
        if result_code:
            self.logger.error(f"AiServer python_exec_shell failed, cmd is {cmd}, error msg is {result_str}")
            return

        # 由于self_play是在main里配置, 这里根据返回的actor_addrs和learner_addrs来决定其值
        actor_addrs_json = json.loads(to_change_key_values.get("actor_addrs"), strict=False)
        self_play = 0
        if len(actor_addrs_json) == 2:
            self_play = 1
        to_change_key_values["self_play"] = self_play

        # 去掉actor_proxy_num和learner_proxy_num参数
        del to_change_key_values["actor_proxy_num"]
        del to_change_key_values["learner_proxy_num"]

        # 追加文件写操作
        with open(CONFIG.cpp_aisrv_configure, "a", encoding=KaiwuDRLDefine.UTF_8) as f:
            for key, value in to_change_key_values.items():
                # gflags严格要求key=value形式, 不能留空格
                f.write(f"--{key}={value}\n")
                self.logger.info(f"AiServer {CONFIG.cpp_aisrv_configure} {key} {value}")

        self.logger.info(f"AiServer {CONFIG.cpp_aisrv_configure} CONFIG save_to_file success")

    def change_configure_content(
        self,
        actor_addrs,
        learner_addrs,
        self_play_actor_address,
        self_play_learner_address,
        self_play_old_actor_address,
        self_play_old_learner_address,
    ):
        """
        修改conf/system/aisrv_system.toml里的配置项目, 如下:
        1. actor_addrs
        2. actor_proxy_num
        3. learner_addrs
        4. learner_proxy_num
        5. self_play_actor_proxy_num
        6. self_play_old_actor_proxy_num
        7. self_play_learner_proxy_num
        8. self_play_old_learner_proxy_num
        """

        # 写回配置文件内容
        to_change_key_values = {}

        # 将当前的配置文件的内容读成json串, 内存修改后, 再写回json内容, 如果解析json串出错, 则提前报错返回
        try:
            old_actor_address_map = copy.deepcopy(CONFIG.actor_addrs)
            old_learner_address_map = copy.deepcopy(CONFIG.learner_addrs)

            # 如果是非self-play, 需要删除掉CONFIG.self_play_old_policy对应的数据
            if not int(CONFIG.self_play):
                if CONFIG.self_play_old_policy in old_actor_address_map:
                    del old_actor_address_map[CONFIG.self_play_old_policy]
                if CONFIG.self_play_old_policy in old_learner_address_map:
                    del old_learner_address_map[CONFIG.self_play_old_policy]

        except Exception as e:
            self.logger.error(
                f"AiServer get actor and learner address from conf failed, error is {str(e)}",
                g_not_server_label,
            )

            return

        """
        处理实例如下:
        actor_addrs = {"train_one": ["127.0.0.1:8001"], "train_two": ["127.0.0.1:8002"]}
        learner_addrs = {"train_one": ["127.0.0.1:9000"], "train_two": ["127.0.0.1:9001"]}
        """

        if not int(CONFIG.self_play):
            if not actor_addrs and not learner_addrs:
                return

            # 如果actor_addrs不空则处理, 否则跳过
            if actor_addrs:
                actor_proxy_num = len(actor_addrs)
                old_actor_address_map[CONFIG.policy_name] = actor_addrs
                to_change_key_values["actor_proxy_num"] = actor_proxy_num
                to_change_key_values["actor_addrs"] = old_actor_address_map

            # 如果learner_addrs不空则处理, 否则跳过
            if learner_addrs:
                learner_proxy_num = len(learner_addrs)
                old_learner_address_map[CONFIG.policy_name] = learner_addrs
                to_change_key_values["learner_proxy_num"] = learner_proxy_num
                to_change_key_values["learner_addrs"] = old_learner_address_map

            # 修改配置文件内容落地
            if actor_addrs or learner_addrs:
                if KaiwuDRLDefine.AISRV_FRAMEWORK_KAIWUDRL == CONFIG.aisrv_framework:
                    self.save_to_file(KaiwuDRLDefine.SERVER_AISRV, to_change_key_values)

                CONFIG.write_to_config(to_change_key_values)
                CONFIG.save_to_file(KaiwuDRLDefine.SERVER_AISRV, to_change_key_values)

                self.logger.info(f"AiServer {KaiwuDRLDefine.SERVER_AISRV} CONFIG save_to_file success")

        else:
            if not self_play_actor_address and not self_play_learner_address and not self_play_old_actor_address:
                return

            if self_play_actor_address:
                self_play_actor_proxy_num = len(self_play_actor_address)
                old_actor_address_map[CONFIG.self_play_policy] = self_play_actor_address
                to_change_key_values["self_play_actor_proxy_num"] = self_play_actor_proxy_num

            if self_play_old_actor_address:
                self_play_old_actor_proxy_num = len(self_play_old_actor_address)
                old_actor_address_map[CONFIG.self_play_old_policy] = self_play_old_actor_address
                to_change_key_values["self_play_old_actor_proxy_num"] = self_play_old_actor_proxy_num

            to_change_key_values["actor_addrs"] = old_actor_address_map

            if self_play_learner_address:
                self_play_learner_proxy_num = len(self_play_learner_address)
                CONFIG.self_play_learner_proxy_num = self_play_learner_proxy_num
                old_learner_address_map[CONFIG.self_play_policy] = self_play_learner_address
                to_change_key_values["self_play_learner_proxy_num"] = self_play_learner_proxy_num

            if self_play_old_learner_address:
                self_play_old_learner_proxy_num = len(self_play_old_learner_address)
                CONFIG.self_play_old_learner_proxy_num = self_play_old_learner_proxy_num
                old_learner_address_map[CONFIG.self_play_old_policy] = self_play_old_learner_address
                to_change_key_values["self_play_old_learner_proxy_num"] = self_play_old_learner_proxy_num

            to_change_key_values["learner_addrs"] = old_learner_address_map

            # 修改配置文件内容落地
            if (
                self_play_actor_address
                or self_play_learner_address
                or self_play_old_actor_address
                or self_play_old_learner_address
            ):
                if KaiwuDRLDefine.AISRV_FRAMEWORK_KAIWUDRL == CONFIG.aisrv_framework:
                    self.save_to_file(KaiwuDRLDefine.SERVER_AISRV, to_change_key_values)

                CONFIG.write_to_config(to_change_key_values)
                CONFIG.save_to_file(KaiwuDRLDefine.SERVER_AISRV, to_change_key_values)

                self.logger.info(f"AiServer {KaiwuDRLDefine.SERVER_AISRV} CONFIG save_to_file success")

    def run(self) -> None:
        if not self.before_run():
            self.logger.error(f"AiServer before_run failed, so return")
            return

        while True:
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    # 因为该run函数内只有周期性的操作和on-policy的操作, 故增加sleep时间减少CPU占用
                    time.sleep(CONFIG.idle_sleep_second * 1000)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0

            except Exception as e:
                self.logger.exception(f"AiServer failed to run {self.name} . exit. Error is: {e}, ")

    def run_once(self) -> None:
        # 步骤1, 启动定时器操作, 定时器里执行记录统计信息
        schedule.run_pending()

    # 框架运行前创建必要的文件目录
    def make_dirs(self):
        make_single_dir(CONFIG.log_dir)

    def start_aisrv_handler(self):
        """

        aisrv在启动时, 从alloc进程获取actor和learner的分配IP地址; 如果不从alloc访问默认是采用配置项, 配置项里会设置具体的地址
            在单个容器里是127.0.0.1
            在多个容器里是需要用户自己设置
        1. actor, learner的分配IP地址
        2. kaiwu_env的分配IP地址
        """
        kaiwu_env_default_address = CONFIG.kaiwu_env_default_address
        if kaiwu_env_default_address:
            kaiwu_env_address = kaiwu_env_default_address.split(",")
        else:
            kaiwu_env_address = [f"{KaiwuDRLDefine.LOCAL_HOST_IP}:{CONFIG.kaiwu_env_svr_port}"]
        if int(CONFIG.use_alloc):
            registry_result = self.aisrv_registry_to_alloc()
            if not registry_result:
                self.logger.error(f"AiServer aisrv_registry_to_alloc failed")
            else:
                # 如果需要从alloc服务获取kaiwu_env地址则获取
                if CONFIG.get_kaiwu_env_by_alloc:
                    kaiwu_env_address = self.get_kaiwu_env_ip_from_alloc()

        if not kaiwu_env_address:
            self.logger.error(f"AiServer fail to get kaiwu_env address")
            return

        # 如果是在aisrv处理多agent时采用并行方式则需要启用进程池
        if CONFIG.multi_agent_predict == KaiwuDRLDefine.MULTI_AGENT_PREDICT_PARALLEL:
            num_processes = os.cpu_count()
            self.simu_ctx.pool = multiprocessing.Pool(processes=num_processes)

        # 针对返回来的kaiwu_env的IP列表, 每个IP启动单个进程
        for index, address in enumerate(kaiwu_env_address):
            if address:
                self.simu_ctx.index = index
                handler = AiSrvHandle(address, self.simu_ctx)
                handler.start()
                self.logger.info(f"AiServer AiSrvHandle with address: {address} start")

    def before_run(self):
        self.make_dirs()

        # 设置Context
        self.simu_ctx = Context()
        # aisrv handler进程使用
        self.simu_ctx.slots = Slots(int(CONFIG.max_tcp_count), int(CONFIG.max_queue_len))

        # aisrv进程启动时, 从七彩石获取配置, 然后将该七彩石的操作句柄传给对应的子进程
        if int(CONFIG.use_rainbow):
            from kaiwudrl.common.utils.rainbow_wrapper import RainbowWrapper

            rainbow_wrapper = RainbowWrapper(self.logger)
            # 在本次对局开始前, aisrv看下参数修改情况
            rainbow_wrapper.rainbow_activate_single_process(KaiwuDRLDefine.SERVER_MAIN, self.logger)
            rainbow_wrapper.rainbow_activate_single_process(CONFIG.svr_name, self.logger)

            self.simu_ctx.rainbow_wrapper = rainbow_wrapper

        # aisrv启动时获取actor和learner地址
        if int(CONFIG.use_alloc):
            # alloc 工具类, aisrv上与alloc交互操作
            self.alloc_util = AllocUtils(self.logger)
            self.aisrv_registry_to_alloc()
            if CONFIG.need_to_start_learner:
                self.get_actor_learner_ip_from_alloc()

        # 无论从七彩石或者其他地方配置完成的配置文件后再开始检测配置文件的有效性
        if not self.check_param():
            self.logger.error(f"AiServer check_param failed, so return")
            return False

        """
        如果在小规模场景下, 因为model_file_sync进程只需要启动1个, 而actor_proxy_local进程是多个的, 故这里需要采用下面步骤:
        1. 如果需要启动learner进程, 即训练进程/预测进程之间需要传递model文件, 则:
        1.1 model_file_sync进程先启动
        1.2 将model_file_sync进程的对象句柄传入到actor_proxy_local进程里进行使用
        又由于会对不同的policy进行AsyncBuilder, 故只有将对model_file_sync的进程启动放在AsyncBuilder调用之前进行
        2. 如果不需要启动learner进程, 即训练进程/预测进程之间不需要传递model文件, 则:
        2.1 不需要启动model_file_sync进程
        """
        if CONFIG.predict_local_or_remote == KaiwuDRLDefine.PREDICT_AS_LOCAL:
            if CONFIG.need_to_start_learner:
                model_file_sync_wrapper = ModelFileSyncWrapper()
                model_file_sync_wrapper.init()

                # 因为在on-policy的情况下存在多个预测进程竞争情况故加上锁的操作
                if CONFIG.algorithm_on_policy_or_off_policy == KaiwuDRLDefine.ALGORITHM_ON_POLICY:
                    lock = multiprocessing.Lock()
                    self.simu_ctx.model_file_sync_wrapper_lock = lock

                self.simu_ctx.model_file_sync_wrapper = model_file_sync_wrapper

        """
        实例配置如下
        {
            "hero": {
                "run_handler": "app.gym.gym_run_handler.GymRunHandler",
                "rl_helper": "app.gorge_walk.environment.gorge_walk_rl_helper.GorgeWalkRLHelper",
                "policies": {
                "train_one": {
                    "policy_builder" : "kaiwudrl.server.aisrv.async_policy.AsyncBuilder",
                    "algo": "ppo",
                    "state": "app.gym.gym_proto.GymState",
                    "action": "app.gym.gym_proto.GymAction",
                    "reward": "app.gym.gym_proto.GymReward",
                    "actor_network": "app.gym.gym_network.GymDeepNetwork",
                    "learner_network": "app.gym.gym_network.GymDeepNetwork",
                    "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper"
                    }
                }
            }
        }
        """
        # 配置相关的传递
        try:
            policies_builder = {}
            policies_conf = AppConf[CONFIG.app].policies
            for policy_name, policy_conf in policies_conf.items():
                policies_builder[policy_name] = policy_conf.policy_builder(policy_name, self.simu_ctx)

            self.simu_ctx.policies_builder = policies_builder

            self.simu_ctx.kaiwu_rl_helper = AppConf[CONFIG.app].rl_helper

        except Exception as e:
            self.logger.exception(f"AiSrv server start exception: {str(e)}")
            return False

        # 启动独立的进程, 负责actor与alloc交互
        if int(CONFIG.use_alloc):
            self.alloc_proxy = AllocProxy()
            self.alloc_proxy.start()

        # 启动独立的进程, 负责actor与普罗米修斯交互
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = MonitorProxy()
            self.monitor_proxy.start()
            self.simu_ctx.monitor_proxy = self.monitor_proxy

        """
        设置了aisrv自动更新actor和learner后, 就设置按时执行
        """
        if CONFIG.actor_learner_expansion:
            set_schedule_event(
                int(CONFIG.alloc_process_per_seconds),
                self.aisrv_with_new_actor_learner_change,
            )

        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.aisrv_main_process_stat)

        self.process_run_count = 0

        # 启动aisrv_handler进程
        self.start_aisrv_handler()

        # 在before run最后打印启动成功日志
        self.logger.info(
            f"AiServer is start success at {CONFIG.aisrv_ip_address}:{CONFIG.aisrv_server_port}, "
            f"pid is {os.getpid()}, run_mode is {CONFIG.run_mode}, "
            f"self_play is {CONFIG.self_play}"
        )
        return True


class AiSrvHandle(multiprocessing.Process):
    __slots__ = (
        "logger",
        "conn",
        "simu_ctx",
        "slots",
        "slot_id",
        "msg_buff",
        "data_queue",
        "kaiwu_rl_helper",
        "min_slot_id",
        "monitor_proxy",
    )

    def __init__(self, kaiwu_env_address, simu_ctx) -> None:
        super(AiSrvHandle, self).__init__()
        self.kaiwu_env_address = kaiwu_env_address
        self.simu_ctx = simu_ctx

        # 建立zmq通信
        """
        self.client_id = get_uuid()
        parts = self.kaiwu_env_address.split(":")
        ip = parts[0]
        port = parts[1]
        self.zmq_client = ZmqClient(str(self.client_id), ip, port)
        self.zmq_client.connect()
        """

        self.simu_ctx.client_address = self.kaiwu_env_address

        self.simu_ctx.exit_flag = multiprocessing.Value("b", False)

        # slot_id
        self.slots = self.simu_ctx.slots
        self.slot_id = self.slots.get_slot()
        self.simu_ctx.slot_id = self.slot_id

        # 七彩石的操作句柄由主进程传递
        if int(CONFIG.use_rainbow):
            self.rainbow_wrapper = self.simu_ctx.rainbow_wrapper

        # 设置aisrv上对客户端的消息buff, 匹配速度
        self.msg_buff = MsgBuff(self.simu_ctx)
        self.simu_ctx.msg_buff = self.msg_buff

        # 负责统计kaiwu_rl_helper中产生的对局数据，线程和主进程间只用数据dequeue即可，减少cpu消耗
        self.data_queue = collections.deque(maxlen=CONFIG.max_queue_len)
        self.simu_ctx.data_queue = self.data_queue

        self.process_run_count = 0

    # 设置每个handler的model_version
    def set_handler_model_version(self, model_version):
        self.kaiwu_rl_helper.from_learner_model_version = model_version
        self.logger.info(f"aisrvhandle on-policy set_handler_model_version success, model_version: {model_version}")

    # aisrv在处理actor和learner的动态扩缩容逻辑
    def aisrv_with_new_actor_learner_change(self):
        if not CONFIG.actor_learner_expansion:
            return

        (
            current_actor_addrs,
            current_learner_addrs,
        ) = self.kaiwu_rl_helper.get_current_actor_learner_address()

        read_from_file_content = CONFIG.read_from_file(CONFIG.svr_name, ["actor_addrs", "learner_addrs"])

        # 本次读取的文件内容错误, 则跳过本次处理下次再进行处理
        try:
            new_actor_addrs = read_from_file_content["actor_addrs"][CONFIG.policy_name]
            new_learner_addrs = read_from_file_content["learner_addrs"][CONFIG.policy_name]
        except Exception as e:
            self.logger.info(f"aisrvhandle load actor address and learner address err, {str(e)}")
            return

        self.aisrv_with_different_actor_learner(
            current_actor_addrs,
            new_actor_addrs,
            current_learner_addrs,
            new_learner_addrs,
        )

    # actor和learner的IP区别判断, 采用2个参数进行返回
    def check_actor_ip_and_learner_ip_change(
        self, actor_address, old_actor_address, learner_address, old_learner_addrs
    ):
        actor_ip_change = False
        learner_ip_change = False

        if not actor_address and not learner_address:
            return actor_ip_change, learner_ip_change

        if actor_address:
            if not is_list_eq(actor_address, old_actor_address):
                actor_ip_change = True

        if learner_address:
            if not is_list_eq(learner_address, old_learner_addrs):
                learner_ip_change = True

        return actor_ip_change, learner_ip_change

    def aisrv_with_different_actor_learner(
        self,
        current_actor_addrs,
        new_actor_addrs,
        current_learner_addrs,
        new_learner_addrs,
    ):
        actor_ip_change, learner_ip_change = self.check_actor_ip_and_learner_ip_change(
            new_actor_addrs,
            current_actor_addrs,
            new_learner_addrs,
            current_learner_addrs,
        )

        if not actor_ip_change and not learner_ip_change:
            return

        # actor地址有变化
        if actor_ip_change:
            list_A_have_B_not_have, list_B_have_A_not_have = list_diff(current_actor_addrs, new_actor_addrs)
            if list_A_have_B_not_have:
                # 新的有, 但是旧的没有, AsyncBuilder新增actor_proxy
                actor_add_result = self.kaiwu_rl_helper.kaiwu_rl_helper_change_actor_learner_ip(
                    KaiwuDRLDefine.PROCESS_ADD, list_A_have_B_not_have, None, None
                )

            if list_B_have_A_not_have:
                # 新的没有, 但是旧的有, AsyncBuilder减少actor_ip
                actor_reduce_result = self.kaiwu_rl_helper.kaiwu_rl_helper_change_actor_learner_ip(
                    KaiwuDRLDefine.PROCESS_REDUCE,
                    list_B_have_A_not_have,
                    None,
                    None,
                )

        # learner地址有变化
        if learner_ip_change:
            list_A_have_B_not_have, list_B_have_A_not_have = list_diff(new_learner_addrs, current_learner_addrs)
            if list_A_have_B_not_have:
                # 新的有, 但是旧的没有, AsyncBuilder新增learner_proxy
                learner_add_result = self.kaiwu_rl_helper.kaiwu_rl_helper_change_actor_learner_ip(
                    None, None, KaiwuDRLDefine.PROCESS_ADD, list_A_have_B_not_have
                )

            if list_B_have_A_not_have:
                # 新的没有, 但是旧的有, AsyncBuilder减少learner_ip
                learner_reduce_result = self.kaiwu_rl_helper.kaiwu_rl_helper_change_actor_learner_ip(
                    None,
                    None,
                    KaiwuDRLDefine.PROCESS_REDUCE,
                    list_B_have_A_not_have,
                )

        # 修改配置文件内容落地
        if actor_add_result and actor_reduce_result and learner_add_result and learner_reduce_result:
            self.logger.info("aisrvhandle aisrv_with_different_actor_learner change finish sucess")

    def before_run(self):
        self.logger = KaiwuLogger()
        self.current_pid = os.getpid()
        self.logger.setLoggerFormat(
            (
                f"/{CONFIG.svr_name}/aisrv_handle_pid{self.current_pid}_log_"
                f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log"
            ),
            "aisrvhandle",
        )

        """
         aisrv下每1个客户端启动1个KaiWuRLHelper对象, 封装了强化学习流程
         1. 如果是主循环的内容在业务侧, 调用self.kaiwu_rl_helper = self.simu_ctx.kaiwu_rl_helper
         2. 如果是主循环的内容在框架侧, 调用self.kaiwu_rl_helper = KaiWuRLHelper(self.simu_ctx)
         """
        self.kaiwu_rl_helper = self.simu_ctx.kaiwu_rl_helper(self.simu_ctx)

        # self.kaiwu_rl_helper = KaiWuRLHelper(self.simu_ctx)

        self.kaiwu_rl_helper.daemon = True
        self.logger.info(f"aisrvhandle use kaiwu_rl_helper: {self.kaiwu_rl_helper}")

        self.min_slot_id, _ = self.slots.get_min_max_slot_id()
        self.logger.info(
            f"aisrvhandle established connect to {self.kaiwu_env_address}, "
            f"slot id is {self.slot_id}, min_slot_id is {self.min_slot_id}"
        )

        (
            current_actor_addrs,
            current_learner_addrs,
        ) = self.kaiwu_rl_helper.get_current_actor_learner_address()
        self.logger.info(
            f"aisrvhandle current_actor_addrs is {current_actor_addrs}, "
            f"current_learner_addrs is {current_learner_addrs}"
        )

        # 启动独立的进程, 负责aisrv与普罗米修斯交互
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = self.simu_ctx.monitor_proxy

            # 传递给kaiwu_rl_helper
            self.kaiwu_rl_helper.set_monitor_proxy(self.monitor_proxy)

        # 支持每局结束前, 动态修改配置文件
        if int(CONFIG.use_rainbow):
            # 在本次对局开始前, aisrv看下参数修改情况
            self.rainbow_wrapper.rainbow_activate_single_process(KaiwuDRLDefine.SERVER_MAIN, self.logger)
            self.rainbow_wrapper.rainbow_activate_single_process(CONFIG.svr_name, self.logger)

        """
        设置了aisrv自动更新actor和learner后, 就设置按时执行
        """
        if CONFIG.actor_learner_expansion:
            set_schedule_event(
                int(CONFIG.alloc_process_per_seconds),
                self.aisrv_with_new_actor_learner_change,
            )

        # 单局开始时
        self.episode_start()

        # 在before run最后打印启动成功日志
        self.logger.info(f"aisrvhandle start success at pid {self.current_pid}")

        # 开启kaiwu_rl_helper线程, 因为在该线程的主循环和其他的进程开始交互, 故放在before_run最后开始
        self.kaiwu_rl_helper.start()

        # 注册SIGTERM信号处理
        register_sigterm_handler(self.handle_sigterm, CONFIG.sigterm_pids_file)

        return True

    def run_once(self) -> None:
        # 步骤1, 例行任务
        schedule.run_pending()

    def run(self) -> None:
        # before_run
        if not self.before_run():
            self.logger.error(f"aisrvhandle before_run failed, so return")
            return

        # 主循环
        try:
            while not self.simu_ctx.exit_flag.value:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.idle_sleep_count == 0:
                    # 因为该run函数内只有周期性的操作和on-policy的操作, 故增加sleep时间减少CPU占用
                    time.sleep(CONFIG.idle_sleep_second * 1000)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0

        except Exception as e:
            self.logger.exception(f"aisrvhandle failed to handle message {str(e)}")
            self.simu_ctx.exit_flag.value = True

            self.episode_stop()

            raise e

    def ep_end_req(self):
        builder = flatbuffers.Builder(0)
        ep_end_req = KaiwuMsgHelper.encode_ep_end_req(builder, b"0", 0, b"")
        req = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.ep_end_req, ep_end_req)
        builder.Finish(req)
        req_msg = builder.Output()

        self.msg_buff.input_q.put(
            req_msg,
        )
        self.logger.info("aisrvhandle send Ep end frame")

    # 单次对局结束时处理
    def episode_stop(self):
        # 安全退出KaiWuRLHelper
        self.kaiwu_rl_helper.stop()

        # 其他报错同理需要回一个结束包，来结束线程，释放资源
        self.ep_end_req()

        # 回收slot_id
        self.slots.put_slot(self.slot_id)

        self.logger.info("aisrvhandle lost connection from {}", str(self.kaiwu_env_address))

    # 单次对局开始时处理
    def episode_start(self):
        pass

    def handle_sigterm(self, sig, frame):
        if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_LOCAL:
            self.logger.info(f"aisrv_handle {os.getpid()} is starting to handle the SIGTERM signal.")
            self.kaiwu_rl_helper.handle_sigterm(sig, frame)
        else:
            self.logger.info(
                f"aisrv_handle not KaiwuDRLDefine.WRAPPER_LOCAL, so {os.getpid()} not handle the SIGTERM signal."
            )
