#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file base_agent_kaiwudrl.py
# @brief 该文件主要是在标准化过程中使用的
# @author kaiwu
# @date 2023-11-28


from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


# 根据配置项CONFIG.wrapper_type确认调用的函数
if CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_REMOTE:
    from kaiwudrl.interface.base_agent_kaiwudrl_remote import *

elif CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_LOCAL:
    from kaiwudrl.interface.base_agent_kaiwudrl_local import *

elif CONFIG.wrapper_type == KaiwuDRLDefine.WRAPPER_NONE:
    from kaiwudrl.interface.base_agent_kaiwudrl_none import *

# 默认设置为remote模式
else:
    from kaiwudrl.interface.base_agent_kaiwudrl_remote import *


class BaseAgent:
    """
    Agent 的基类，所有的 Agent 都应该继承自这个类"""

    def __init__(self, agent_type="player", device="cpu", logger=None, monitor=None) -> None:
        self.file_queue = []

        # KaiwuDRL传递的句柄
        self.framework_handler = None

    def set_framework_handler(self, framework_handler):
        self.framework_handler = framework_handler

    @learn_wrapper
    def learn(self, list_sample_data) -> dict:
        """
        用于学习的函数，接受一个 SampleData 的列表
        - dqn/ppo: 每个 game_data 是一个 episode 的数据
        - dp: 每个 game_data 是一个 step 的数据
        """
        raise NotImplementedError

    @predict_wrapper
    def predict(self, list_obs_data: list) -> list:
        """
        用于获取动作的函数，接受一个 ObsData 的列表, 返回一个动作列表
        """
        raise NotImplementedError

    @exploit_wrapper
    def exploit(self, list_obs_data: list) -> list:
        """
        用于获取动作的函数，接受一个 ObsData 的列表, 返回一个动作列表
        """
        raise NotImplementedError

    def get_action(self, *kargs, **kwargs):
        raise NotImplementedError

    @save_model_wrapper
    def save_model(self, path, id="1"):
        raise NotImplementedError

    @load_model_wrapper
    def load_model(self, path, id="1"):
        raise NotImplementedError
