#!/usr/bin/env python
# -*- coding: utf-8 -*-


# @file app_conf.py
# @brief
# @author kaiwu
# @date 2023-11-28


from pydoc import locate

import toml

from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.singleton import Singleton
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine


class _PolicyConf:
    """业务配置类, 实例如下
    {
    "gym": 业务名
    {
        "run_handler": "app.gym.gym_run_handler.GymRunHandler",
        "rl_helper": "environment.gorge_walk_rl_helper.GorgeWalkRLHelper",
        "policies": {
        "train": {
            "policy_builder": "kaiwudrl.server.aisrv.async_policy.AsyncBuilder",
            "algo": "ppo", 算法
            "state": "app.gym.gym_proto.GymState", State
            "action": "app.gym.gym_proto.GymAction", Action
            "reward": "app.gym.gym_proto.GymReward", Reward
            "actor_network": "app.gym.gym_network.GymDeepNetwork", NetWork
            "learner_network": "app.gym.gym_network.GymDeepNetwork", NetWork
            "reward_shaper": "app.gym.gym_reward_shaper.GymRewardShaper" Reward Shaper,
            "eigent_value": "app.gym.gym_eigent_value.GymEigentValue"
        }
        }
    }
    }
    """

    def __init__(self, policy_builder, **kwargs):
        """
        一般需要定义state、action、reward以及reward_shaper等
        """
        self.policy_builder = locate(policy_builder)
        algo = kwargs.pop("algo", None)
        if algo is not None:
            self.algo = algo

        # 如果是aisrv, 不需要加载actor, learner相关的
        if CONFIG.svr_name == KaiwuDRLDefine.SERVER_AISRV:
            if "learner_network" in kwargs:
                del kwargs["learner_network"]
        else:
            # 如果是learner, 不需要加载actor相关的
            if CONFIG.svr_name == KaiwuDRLDefine.SERVER_ACTOR:
                if "learner_network" in kwargs:
                    del kwargs["learner_network"]

            # 如果是actor, 不需要加载learner相关的
            elif CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER:
                if "actor_network" in kwargs:
                    del kwargs["actor_network"]
            else:
                pass

        # 兼顾actor_network和learner_network有多个情况
        attrs = {}
        if "actor_network" in kwargs:
            actor_networks = kwargs["actor_network"].split(",")
            attrs["actor_network"] = [locate(network) for network in actor_networks]

        if "learner_network" in kwargs:
            learner_networks = kwargs["learner_network"].split(",")
            attrs["learner_network"] = [locate(network) for network in learner_networks]

        # Update attrs with the remaining kwargs, excluding 'actor_network' and 'learner_network'
        attrs.update({k: locate(clazz) for k, clazz in kwargs.items() if k not in ["actor_network", "learner_network"]})

        self.__dict__.update(**attrs)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


@Singleton
class _AppConf:
    class TupleType:
        def __init__(self, run_handler, rl_helper, policies, probs_handler=None, builder=None, **kwargs):
            if "server_name" in kwargs and kwargs["server_name"] == KaiwuDRLDefine.SERVER_AISRV:
                self.run_handler = locate(run_handler)
                self.rl_helper = locate(rl_helper)

            if "environment" in kwargs:
                self.environment = locate(kwargs["environment"])
            else:
                self.environment = None

            if probs_handler is None:
                probs_handler = "tools.action_space_1v1.DumpProbs"
            self.probs_handler = locate(probs_handler)

            self.policies = {name: _PolicyConf(**policy) for name, policy in policies.items()}
            if "actor" in kwargs:
                self.actor = locate(kwargs["actor"])
            else:
                self.actor = None

            if builder is None:
                builder = "kaiwudrl.interface.builder.Builder"
            self.builder = locate(builder)

    _instance = None

    def __init__(self):
        self.config_map = {}

    def __getitem__(self, key):
        return self.config_map[key]

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def load_conf(self, file_name, server_name=None):

        with open(file_name, "r") as file_obj:
            self._load_conf(file_obj.read(), server_name)

    def _load_conf(self, algo_str, server_name=None):
        algo_obj = toml.loads(algo_str)

        if server_name:
            algo_obj[CONFIG.app]["server_name"] = server_name

        assert CONFIG.app in algo_obj, f"failed to find {CONFIG.app} app conf"
        self.config_map[CONFIG.app] = self.TupleType(**algo_obj[CONFIG.app])


AppConf = _AppConf()
