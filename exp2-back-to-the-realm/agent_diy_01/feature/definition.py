#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np
from kaiwu_agent.back_to_the_realm.target_dqn.feature_process import bump
from ..conf.conf import Config

# The create_cls function is used to dynamically create a class.
# The first parameter of the function is the type name, and the remaining parameters are the attributes of the class.
# The default value of the attribute should be set to None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_act=None,
)


ActData = create_cls(
    "ActData",
    move_dir=None,
    use_talent=None,
)


SampleData = create_cls(
    "SampleData",
    obs=None,
    _obs=None,
    obs_legal=None,
    _obs_legal=None,
    act=None,
    rew=None,
    ret=None,
    done=None,
)


def reward_shaping(frame_no, score, terminated, truncated, remain_info, _remain_info, state_env_info, _state_env_info):
    """
    Reward shaping for DP algorithm
    DP算法的奖励塑形
    
    DP算法不依赖奖励来学习，奖励仅用于日志统计。
    简单返回基础奖励信号即可。
    """
    # 基础奖励
    reward = 0
    is_bump = 0
    reward_end_dist = 0
    reward_exploration = 0
    reward_treasure_dist = 0
    reward_treasure = 0

    if _state_env_info is not None:
        treasure_collected = int(_state_env_info.game_info.treasure_collected_count)
        prev_collected = int(state_env_info.game_info.treasure_collected_count)
        treasure_count = int(_state_env_info.game_info.treasure_count)

        # 宝箱收集奖励
        if treasure_collected > prev_collected:
            reward_treasure = 1.0
            reward += 2.0

        # 通关奖励
        if terminated:
            if treasure_collected >= treasure_count:
                reward += 10.0  # 全收集通关
            else:
                reward += 1.0   # 部分收集通关

        # 超时惩罚
        if truncated:
            reward -= 5.0

        # 撞墙检测
        pos = _state_env_info.frame_state.heroes[0].pos
        prev_pos = state_env_info.frame_state.heroes[0].pos
        is_bump = bump(pos.x, pos.z, prev_pos.x, prev_pos.z)
        if is_bump:
            reward -= 0.5

    return (
        reward,
        is_bump,
        reward_end_dist,
        reward_exploration,
        reward_treasure_dist,
        reward_treasure,
    )


@attached
def sample_process(list_game_data):
    """
    Sample processing function
    样本处理函数
    
    DP算法不需要训练样本，但框架要求此函数存在。
    """
    return [SampleData(**i.__dict__) for i in list_game_data]


# SampleData <----> NumpyData
@attached
def SampleData2NumpyData(g_data):
    """
    Convert SampleData to NumpyData for reverb storage
    将SampleData转换为NumpyData用于reverb存储
    """
    return np.hstack(
        (
            np.array(g_data.obs, dtype=np.float32),
            np.array(g_data._obs, dtype=np.float32),
            np.array(g_data.obs_legal, dtype=np.float32),
            np.array(g_data._obs_legal, dtype=np.float32),
            np.array(g_data.act, dtype=np.float32),
            np.array(g_data.rew, dtype=np.float32),
            np.array(g_data.ret, dtype=np.float32),
            np.array(g_data.done, dtype=np.float32),
        )
    )


@attached
def NumpyData2SampleData(s_data):
    """
    Convert NumpyData back to SampleData
    将NumpyData转换回SampleData
    """
    obs_data_size = (2 * Config.VIEW_SIZE + 1) ** 2 * 4 + 404
    return SampleData(
        obs=s_data[:obs_data_size],
        _obs=s_data[obs_data_size : 2 * obs_data_size],
        obs_legal=s_data[-8:-6],
        _obs_legal=s_data[-6:-4],
        act=s_data[-4],
        rew=s_data[-3],
        ret=s_data[-2],
        done=s_data[-1],
    )
