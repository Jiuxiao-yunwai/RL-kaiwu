
#!/usr/bin/env python3
# -*- coding:utf-8 -*-


'''
日志记录类, git代码: https://github.com/Delgan/loguru
框架提供self.logger, 单个进程/线程下面同一个对象, 业务不需要自己定义日志对象

优点:
开箱即用，无需准备
无需初始化，导入函数即可使用
更容易的文件日志记录与转存/保留/压缩方式
更优雅的字符串格式化输出
可以在线程或主线程中捕获异常
可以设置不同级别的日志记录样式
支持异步，且线程和多进程安全
支持惰性计算
适用于脚本和库
完全兼容标准日志记录
更好的日期时间处理

实例, 打印日志地方需要按照需要设置日志级别:
self.logger = getLogger()
setLoggerFormat(f"/actor/actor_server_log_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}.log", 'actor_server')
self.logger.info('actor_server process is pid is {}', os.getpid())

系统会对进程aisrv、actor、learner的日志内容增加进程名字样, 其他的日志内容不会增加
'''

import sys
from loguru import logger
from kaiwu_env.conf import ini_logging as CONFIG
from kaiwu_env.utils.common_func import Singleton
from kaiwu_env.utils.common_func import stop_process_by_name

g_not_server_label = 'not_server'

@Singleton
class ArenaLogger(object):
    def __init__(self) -> None:

        # 清除打印到屏幕的日志输出, 即sys.stderr, 并且重新设置下针对sys.stderr的日志级别
        logger.remove(handler_id=None)
        logger.add(sys.stderr, level=CONFIG.main.level)

        # 返回的路径深度
        self.depth = 1

    '''
    调用设置日志各种参数
    1. file_name是必须的, 即日志生成的配置文件
    2. fileter_content, 如果是单进程使用, 则无需设置; 如果是单个进程里需要过滤则需要设置
    '''
    def setLoggerFormat(self, file_name, filter_content=None):
        filter_func = None
        if filter_content:
            # 开发测试阶段, 可以采用 filter=lambda x: print(x, filter_content) or filter_content in x['message']打印日志
            filter_func = lambda x: filter_content in x['message']

        logger.add(CONFIG.main.log_dir + file_name, rotation=CONFIG.main.rotation, 
                            encoding=CONFIG.main.encoding, enqueue=True, compression=CONFIG.main.compression, retention=CONFIG.main.retention, 
                                level=CONFIG.main.level, filter=filter_func, serialize=CONFIG.main.serialize)

    '''
    根据填写的字符串, 增加进程名字内容, 便于进行filter操作
    1. 如果是需要包含进程名的日志, 则前面添加进程名, 主要针对aisrv、actor、learner进程, 形如learner msg
    2. 如果是不需要包含进程名的日志, 则前面不需要添加进程名, 主要针对aisrv、actor、learner进程派生的进程, 例如learner model_file_sync
    '''
    def make_msg_content(self, msg, not_server=True):
        if not_server:
            return msg

        msg = f'{CONFIG.main.svr_name} {msg}'
        return  msg
    
    def is_not_server(self, *args):
        return g_not_server_label in args


    def debug(self, msg, *args, **kwargs):
        return logger.opt(depth=self.depth).debug(self.make_msg_content(msg, self.is_not_server(*args)), *args, **kwargs)


    def info(self, msg, *args, **kwargs):
        return logger.opt(depth=self.depth).info(self.make_msg_content(msg, self.is_not_server(*args)), *args, **kwargs)


    def warning(self, msg, *args, **kwargs):
        return logger.opt(depth=self.depth).warning(self.make_msg_content(msg, self.is_not_server(*args)), *args, **kwargs)


    def error(self, msg, *args, **kwargs):

        logger_opt = logger.opt(depth=self.depth).error(self.make_msg_content(msg, self.is_not_server(*args)), *args, **kwargs)
        return logger_opt


    def critical(self, msg, *args, **kwargs):
        return logger.opt(depth=self.depth).critical(self.make_msg_content(msg, self.is_not_server(*args)), *args, **kwargs)


    def exception(self, msg, *args, **kwargs):
        return logger.opt(depth=self.depth).critical(self.make_msg_content(msg, self.is_not_server(*args)), *args, **kwargs)


    def log(self, level, msg, *args, **kwargs):
        return logger.opt(depth=self.depth).log(level, self.make_msg_content(msg, self.is_not_server(*args)), *args, **kwargs)


if __name__=='__main__':
    from kaiwu_env.utils.logging import ArenaLogger

    logger = ArenaLogger()
    svr_name = "gorge_walk"
    now = 111
    g_battlesrv_pid = 2222
    logger.setLoggerFormat(
            f"/{svr_name}/gorge_walk_battlesrv_pid{g_battlesrv_pid}_log_{now}.log", svr_name)

    logger.error('can not get aisrv_ip or aisrv_port, so exit')
