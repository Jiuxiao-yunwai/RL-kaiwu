#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import os
import json
import warnings
from kaiwu_agent.conf import tree_strdef
from kaiwu_agent.conf import ini_alloc as CONFIG
from kaiwu_agent.utils.http_utils import http_utils_post
from kaiwu_agent.utils.common_func import get_host_ip

'''
aisrv、actor、learner的进程与alloc进程交互的类, 主要包括：
1. 注册：/api/registry
2. 请求实例：/api/get
'''

# alloc里关于KaiwuDRL进程的配置, map形式, 一般不做修改
SERVER_ROLE_CONFIGURE = {
    tree_strdef.SERVER_BATTLE.VALUE: CONFIG.main.alloc_process_role_battlesrv
}

# alloc里关于KaiwuDRL的进程的端口, map形式, 一般不做修改
SERVER_PORT_CONFIGURE = {
    tree_strdef.SERVER_BATTLE.VALUE: 5555
}

# alloc里关于KaiwuDRL进程比例的配置, map形式, 一般不做修改
# 比如aisrv选择的比例是1, 则意味着100台battlesrv与100台aisrv建立连接
# 比如aisrv选择的比例是2, 则意味着100台battlesrv与50台aisrv建立连接, 每2台battlesrv连接1台aisrv, 有50台aisrv是空闲的
SERVER_ASSIGN_LIMIT = {
    tree_strdef.SERVER_BATTLE.VALUE: CONFIG.main.alloc_process_assign_limit_battlesrv
}

'''
aisrv、actor、learner的进程与alloc进程交互的类, 主要包括：
1. 注册：/api/registry
2. 请求实例：/api/get
'''


class AllocUtils(object):
    def __init__(self, logger) -> None:
        super().__init__()

        # 舍弃ResourceWarning的警告信息
        warnings.simplefilter('ignore', ResourceWarning)

        # 日志句柄
        self.logger = logger

        self.set_name = CONFIG.main.set_name
        self.role = SERVER_ROLE_CONFIGURE.get(CONFIG.main.svr_name)
        # IP:端口形式
        self.addr = f'{get_host_ip()}:{SERVER_PORT_CONFIGURE.get(CONFIG.main.svr_name)}'

        self.alloc_addr = f'http://{CONFIG.main.alloc_process_address}'

        # task_id
        self.task_id = CONFIG.main.task_id

        '''
        由于存在self_play模式和非self_play模式, 故这里的get参数由调用者设置
        1. 非self_play模式, 增加参数target_role, 形如:
        {
            "addr":"7.7.7.7:7777", // 字符串。自己的ip:port。或者一个唯一的id
            "target_role":2 // 整数。 请求哪种实例, 含义同上
        }
        2. self_play模式, 增加参数set_list
        {
            "addr":"7.7.7.7:7777", // 字符串。自己的ip:port。或者一个唯一的id
            "set_list": [
                {
                    "set": "set1",
                    "target_role": 3
                },
                {
                    "set": "set2",
                    "target_role": 4
                }
            ]
        }

        '''
        self.get_param = {
            "addr": self.addr,
            "task_id": self.task_id,
        }

    # 注册
    def registry(self):
        # 由于每次可能更新post参数, 故post参数需要放在这里组装
        self.assign_limit = SERVER_ASSIGN_LIMIT.get(CONFIG.main.svr_name)
        self.post_param = {
            "set": self.set_name,
            "role": self.role,
            "addr": self.addr,
            "assign_limit": self.assign_limit,
            "task_id": self.task_id,
        }
        url = f'{self.alloc_addr}/api/registry'
        resp = http_utils_post(url, self.post_param)
        if not resp:
            return False, "http failed"

        if resp['code'] == 0:
            return True, None

        return resp['code'], resp['msg']

    # 获取实例, 需要填写目的端的role, set_name, self_play_set_name
    def get(self, srv_name, set_name, self_play_set_name):

        target_role = SERVER_ROLE_CONFIGURE[srv_name]

        url = f'{self.alloc_addr}/api/get'
        
        if True:
            set_list = [{
                'set': set_name,
                'target_role': target_role
            }, ]

            self.get_param['set_list'] = set_list

        # 目前需要采用post方式获取
        resp = http_utils_post(url, self.get_param)
        # resp = http_utils_post(url, self.get_param)
        # resp = http_utils_request(url, self.get_param)
        if not resp:
            # 为了适配code为0时成功, 非0时失败
            return True, "http failed", None

        # 这里直接返回了code, msg, content, 需要和alloc约定
        return resp['code'], resp['msg'], resp['content']


    def get_aisrv_ip(self, set_name):

        if not set_name:
            return None

        # 获取aisrv地址
        aisrv_address = []
        code, msg, content = self.get(
            tree_strdef.SERVER_AISRV.VALUE, set_name, None)
        if code:
            self.logger.error(
                f"get aisrv IP fail, will rety next time, error_cod is {msg}")
        else:
            # 注意需要和alloc约定格式, 返回json串
            content = json.loads(content, strict=False)
            for set_list in content['set_list']:
                aisrv_address.append(set_list['addr'])

        return aisrv_address
