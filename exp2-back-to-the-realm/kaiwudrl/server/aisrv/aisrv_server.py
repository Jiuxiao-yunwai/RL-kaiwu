#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file aisrv_server.py
# @brief
# @author kaiwu
# @date 2023-11-28


import json
import multiprocessing
import datetime
import os
import schedule
import sys
import time
import copy
from kaiwudrl.common.monitor.monitor_proxy_process import MonitorProxy
from kaiwudrl.common.logging.kaiwu_logger import KaiwuLogger, g_not_server_label
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.alloc.alloc_proxy import AllocProxy, AllocUtils
from kaiwudrl.common.config.app_conf import AppConf
from kaiwudrl.common.utils.common_func import (
    python_exec_shell,
    make_single_dir,
    set_schedule_event,
    actor_learner_aisrv_count,
    get_host_ip,
)


class AiServer(multiprocessing.Process):
    def __init__(
        self,
    ) -> None:
        super(AiServer, self).__init__()

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
                    time.sleep(CONFIG.idle_sleep_second)

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

    # 启动C++常驻进程
    def start_cpp_daemon(self):
        cmd = "sh tools/aisrv_cpp_server_start.sh"
        result_code, result_str = python_exec_shell(cmd)
        if result_code:
            return False

        self.logger.info(f"AiServer C++ Daemon Process starts success, cmd is {cmd}")
        return True

    # 从C++ server获取监控信息
    def cpp_stat(self):
        result = self.lib.get_cpp_server_stat_data()
        if not result:
            return

        # 进行上报, 注意取出来的数据需要强制转换下数据类型
        if int(CONFIG.use_prometheus):
            monitor_data = {
                KaiwuDRLDefine.AISRV_TCP_BATTLESRV: actor_learner_aisrv_count(self.host, CONFIG.svr_name),
                KaiwuDRLDefine.MONITOR_AISRV_SENDTO_ACTOR_SUCC_CNT: result.get(
                    KaiwuDRLDefine.MONITOR_AISRV_SENDTO_ACTOR_SUCC_CNT
                ),
                KaiwuDRLDefine.MONITOR_AISRV_SENDTO_ACTOR_ERROR_CNT: result.get(
                    KaiwuDRLDefine.MONITOR_AISRV_SENDTO_ACTOR_ERROR_CNT
                ),
                KaiwuDRLDefine.MONITOR_AISRV_RECVFROM_ACTOR_SUCC_CNT: result.get(
                    KaiwuDRLDefine.MONITOR_AISRV_RECVFROM_ACTOR_SUCC_CNT
                ),
                KaiwuDRLDefine.MONITOR_AISRV_RECVFROM_ACTOR_ERROR_CNT: result.get(
                    KaiwuDRLDefine.MONITOR_AISRV_RECVFROM_ACTOR_ERROR_CNT
                ),
                KaiwuDRLDefine.MONITOR_SEND_TO_LEARNER_PROXY_SUC_CNT: result.get(
                    KaiwuDRLDefine.MONITOR_SEND_TO_LEARNER_PROXY_SUC_CNT
                ),
                KaiwuDRLDefine.MONITOR_SEND_TO_LEARNER_PROXY_ERR_CNT: result.get(
                    KaiwuDRLDefine.MONITOR_SEND_TO_LEARNER_PROXY_ERR_CNT
                ),
                KaiwuDRLDefine.MONITOR_AISRV_SEND_TO_BATTLESRV_SUC_CNT: result.get(
                    KaiwuDRLDefine.MONITOR_AISRV_SEND_TO_BATTLESRV_SUC_CNT
                ),
                KaiwuDRLDefine.MONITOR_AISRV_SEND_TO_BATTLESRV_ERR_CNT: result.get(
                    KaiwuDRLDefine.MONITOR_AISRV_SEND_TO_BATTLESRV_ERR_CNT
                ),
                KaiwuDRLDefine.MONITOR_AISRV_RECV_FROM_BATTLESRV_SUC_CNT: result.get(
                    KaiwuDRLDefine.MONITOR_AISRV_RECV_FROM_BATTLESRV_SUC_CNT
                ),
                KaiwuDRLDefine.MONITOR_AISRV_RECV_FROM_BATTLESRV_ERR_CNT: result.get(
                    KaiwuDRLDefine.MONITOR_AISRV_RECV_FROM_BATTLESRV_ERR_CNT
                ),
                KaiwuDRLDefine.MONITOR_AISRV_MAX_PROCESSING_TIME: result.get(
                    KaiwuDRLDefine.MONITOR_AISRV_MAX_PROCESSING_TIME
                ),
            }

            self.monitor_proxy.put_data({self.current_pid: monitor_data})

            # 指标周期性复原
            self.lib.cpp_server_stat_data_reset()

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

    def before_run(self):

        # 设置日志Log配置
        self.logger = KaiwuLogger()
        self.current_pid = os.getpid()
        self.logger.setLoggerFormat(
            f"/{CONFIG.svr_name}/aiserver_pid{self.current_pid}_log_"
            f"{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log",
            "AiServer",
        )

        self.make_dirs()

        # aisrv进程启动时, 从七彩石获取配置
        if int(CONFIG.use_rainbow):
            from kaiwudrl.common.utils.rainbow_wrapper import RainbowWrapper

            rainbow_wrapper = RainbowWrapper(self.logger)
            # 在本次对局开始前, aisrv看下参数修改情况
            rainbow_wrapper.rainbow_activate_single_process(KaiwuDRLDefine.SERVER_MAIN, self.logger)
            rainbow_wrapper.rainbow_activate_single_process(CONFIG.svr_name, self.logger)

        # aisrv在启动时, 从alloc进程获取actor和learner的分配IP地址
        if int(CONFIG.use_alloc):
            self.alloc_util = AllocUtils(self.logger)
            self.aisrv_registry_to_alloc()
            self.get_actor_learner_ip_from_alloc()

        # 无论从七彩石或者其他地方配置完成的配置文件后再开始检测配置文件的有效性
        if not self.check_param():
            self.logger.error(f"AiServer check_param failed, so return")
            sys.exit(-1)

        # C++和python进程不能同时启动
        time.sleep(CONFIG.start_python_daemon_sleep_after_cpp_daemon_sec)

        # 需要等alloc获取服务正常后开始启动C++进程
        if not self.start_cpp_daemon():
            self.logger.error(f"AiServer C++ Daemon Process starts failed, please see the log")
            sys.exit(-1)

        # 启动独立的进程, 负责actor与alloc交互
        if int(CONFIG.use_alloc):
            self.alloc_proxy = AllocProxy()
            self.alloc_proxy.start()

        # 启动独立的进程, 负责actor与普罗米修斯交互
        if int(CONFIG.use_prometheus):
            self.monitor_proxy = MonitorProxy()
            self.monitor_proxy.start()

        """
        设置了aisrv自动更新actor和learner后, 就设置按时执行
        """
        if CONFIG.actor_learner_expansion:
            set_schedule_event(
                int(CONFIG.alloc_process_per_seconds),
                self.aisrv_with_new_actor_learner_change,
            )

        # 设置python调用C++的类库
        os.chdir("/data/projects/kaiwu-fwk/kaiwudrl/server/cpp/dist/aisrv/")
        from kaiwudrl.server.cpp.dist.aisrv.aisrv_server import cpp_aisrv_server

        self.lib = cpp_aisrv_server()
        self.logger.info(f"AiServer C++ lib start success")

        set_schedule_event(CONFIG.prometheus_stat_per_minutes, self.cpp_stat)

        self.process_run_count = 0

        # 获取本机IP
        self.host = get_host_ip()

        # 在before run最后打印启动成功日志
        self.logger.info(
            f"AiServer is start success at {CONFIG.aisrv_ip_address}:{CONFIG.aisrv_server_port}, "
            f"pid is {pid}, run_mode is {CONFIG.run_mode}, self_play is {CONFIG.self_play}"
        )

        return True
