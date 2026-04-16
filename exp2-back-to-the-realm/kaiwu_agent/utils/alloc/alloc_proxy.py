#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import multiprocessing
import datetime
import os
import time
import traceback
import schedule
from kaiwu_agent.conf import ini_alloc as CONFIG
from kaiwu_agent.utils.alloc.alloc_utils import AllocUtils
from kaiwu_agent.utils.logging import ArenaLogger, g_not_server_label
from kaiwu_agent.utils.common_func import set_schedule_envent
from kaiwu_agent.utils.monitor.prometheus_utils import PrometheusUtils
from kaiwu_agent.conf import tree_strdef



class AllocProxy(multiprocessing.Process):

    def __init__(self) -> None:
        super(AllocProxy, self).__init__()

        # 进程是否退出, 用于在异常条件下主动退出进程
        self.exit_flag = multiprocessing.Value('b', False)

    def before_run(self):

        # 日志处理
        self.logger = ArenaLogger()
        pid = os.getpid()
        self.logger.setLoggerFormat(
            f"/{CONFIG.main.svr_name}/alloc_proxy_pid{pid}_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", 'alloc_proxy')
        self.logger.info(f'alloc_proxy start at pid {pid}', g_not_server_label)
        self.logger.info(
            f'alloc_proxy Due to the large amount of logs, the log is printed only when the registration is wrong. ', g_not_server_label)

        # alloc 工具类, 与alloc交互操作
        self.alloc_util = AllocUtils(self.logger)

        # 访问普罗米修斯的类
        self.prometheus_utils = PrometheusUtils(self.logger)

        self.set_event_alloc_interact()

        self.process_run_count = 0

    
    def set_event_alloc_interact(self):
        set_schedule_envent(CONFIG.main.alloc_process_per_seconds,
                            self.alloc_interact, 'seconds')

    '''
    aisrv/actor/learner进程与alloc交互
    '''

    def alloc_interact(self):
        code, msg = self.alloc_util.registry()
        # 服务发现的每隔N秒进行, 导致打印的日志比较多, 这里采用出错时打印方法
        if not code:
            self.logger.error(
                f"alloc_proxy alloc interact registry fail, will rety next time, error_code is {code}", g_not_server_label)

            # 如果本次的注册失败, 表明alloc服务不稳定, 不需要进行下一步操作, 等下一次再操作
            return

        if tree_strdef.SERVER_AISRV.VALUE == CONFIG.main.svr_name:
            # 对比项目reset
            self.check_actor_learner_from_alloc_init()

            # aisrv需要从alloc拉取最新的actor和learner地址, 然后更新内存里的配置, 再更新配置文件, 更新到最新的与新的actor和learner之间的连接
            self.check_actor_learner_from_alloc()

    
    def run_once(self):

        # 启动定时器
        schedule.run_pending()

    '''
    进程停止函数
    '''

    def stop(self):
        self.exit_flag.value = True
        self.join()

        self.logger.info('alloc_proxy AllocProxy stop success',
                         g_not_server_label)

    def run(self) -> None:
        self.before_run()

        while not self.exit_flag.value:
            try:
                self.run_once()

                # 短暂sleep, 规避容器里进程CPU使用率100%问题
                self.process_run_count += 1
                if self.process_run_count % CONFIG.main.idle_sleep_count == 0:
                    time.sleep(CONFIG.main.idle_sleep_second)

                    # process_run_count置0, 规避溢出
                    self.process_run_count = 0

            except Exception as e:
                self.logger.error(
                    f'alloc_proxy run error: {str(e)}, traceback.print_exc() is {traceback.format_exc()}', g_not_server_label)




if __name__=='__main__':
    logger = ArenaLogger()

    alloc_proxy = AllocProxy()
    alloc_proxy.start()
    global g_alloc_pid
    g_alloc_pid = alloc_proxy.pid

    alloc_util = AllocUtils(logger)