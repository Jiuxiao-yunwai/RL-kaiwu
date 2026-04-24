#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
Optimized by: RL Expert - 全面优化奖励设计和样本处理
"""


import numpy as np
from kaiwu_agent.utils.common_func import attached, create_cls
from agent_target_dqn.conf.conf import Config


def bump(a1, b1, a2, b2):
    """
    This function is used to determine whether the game hits a wall.
        - There will be no bump in the first frame.
        - Starting from the second frame, if the moving distance is less than 500, it will be considered as hitting a wall.

    该函数用于判断是否撞墙
        - 第一帧不会bump
        - 第二帧开始, 如果移动距离小于500则视为撞墙
    """
    if a2 == -1 and b2 == -1:
        return False
    if a1 == -1 and b1 == -1:
        return False

    dist = ((a1 - a2) ** 2 + (b1 - b2) ** 2) ** (0.5)

    return dist <= 500


# The create_cls function is used to dynamically create a class. The first parameter of the function is the type name,
# and the remaining parameters are the attributes of the class, which should have a default value of None.
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


def convert_pos_to_grid_pos(x, z):
    """
    Convert the position 'pos' into grid-based coordinates
    将pos转换为珊格化后坐标

    Args:
        x (float): x
        z (float): z

    Returns:
        _type_: tuple
    """
    x = (x + 2250) // 500
    z = (z + 5250) // 500

    # This step is necessary in order to be aligned with the order of json files
    # 这一步是为了与json文件的顺序保持一致
    x, z = z, x

    return x, z


def reward_shaping(
    frame_no,
    score,
    terminated,
    truncated,
    remain_info,
    _remain_info,
    obs_data,
    _obs_data,
):
    """
    优化后的奖励函数 - 采用阶段性策略和递进式奖励设计
    
    核心理念:
    1. 宝箱未全部收集时 → 强引导收集宝箱（最近优先）
    2. 宝箱全部收集后 → 强引导奔向终点
    3. 递进式宝箱奖励，越多越值钱
    4. 终点大奖，鼓励全收集通关
    5. 重惩撞墙和原地踏步
    """
    reward = 0

    # ========================================
    # 基础信息获取
    # ========================================
    
    # 当前位置坐标
    pos = _obs_data.frame_state.heroes[0].pos
    curr_pos_x, curr_pos_z = pos.x, pos.z

    # 上一帧位置
    prev_pos = obs_data.frame_state.heroes[0].pos
    prev_pos_x, prev_pos_z = prev_pos.x, prev_pos.z

    # 终点距离信息
    end_dist = _remain_info.get("end_pos").l2_distance
    prev_end_dist = remain_info.get("end_pos").l2_distance

    # 宝箱距离信息
    treasure_dists = [pos.grid_distance if pos.grid_distance > 0 else -1 for pos in _remain_info.get("treasure_pos")]
    prev_treasure_dists = [
        pos.grid_distance if pos.grid_distance > 0 else -1 for pos in remain_info.get("treasure_pos")
    ]
    
    treasure_count = _remain_info.get("treasure_count")
    treasure_collected_count = _remain_info.get("treasure_collected_count")
    prev_treasure_collected_count = remain_info.get("treasure_collected_count")

    # 是否还有剩余宝箱
    is_treasures_remain = treasure_collected_count < treasure_count
    
    # 宝箱收集进度 (0.0 ~ 1.0)
    treasure_progress = treasure_collected_count / max(treasure_count, 1)

    # ========================================
    # 奖励1: 宝箱收集奖励 (核心奖励)
    # ========================================
    reward_treasure = 0.0
    if treasure_collected_count > prev_treasure_collected_count:
        # 递进式宝箱奖励: 第1个=2.3, 第7个=4.1, 第13个=5.9
        # 量级控制在clamp(-10,10)有效范围内
        reward_treasure = 2.0 + 0.3 * treasure_collected_count

    # ========================================
    # 奖励2: 向最近宝箱靠近的奖励 (引导性奖励)
    # ========================================
    reward_treasure_dist = 0.0
    if is_treasures_remain:
        visible_dists = [(i, d) for i, d in enumerate(treasure_dists) if d > 0]
        if visible_dists:
            # 选择最近的宝箱
            nearest_idx, min_dist = min(visible_dists, key=lambda x: x[1])
            prev_dist = prev_treasure_dists[nearest_idx]
            if prev_dist > 0:
                # 距离变化的归一化奖励
                dist_delta = prev_dist - min_dist
                if dist_delta > 0:
                    reward_treasure_dist = 1.0  # 靠近宝箱
                else:
                    reward_treasure_dist = -0.5  # 远离宝箱
            elif prev_dist <= 0 and min_dist > 0:
                # 新发现宝箱
                reward_treasure_dist = 0.5

    # ========================================
    # 奖励3: 终点相关奖励
    # ========================================
    reward_end_dist = 0.0
    reward_win = 0.0
    
    if not is_treasures_remain:
        # 全部宝箱已收集 → 强引导奔向终点
        if prev_end_dist > 0:
            if end_dist < prev_end_dist:
                reward_end_dist = 2.0   # 强靠近终点奖励
            else:
                reward_end_dist = -1.5  # 强远离终点惩罚
    else:
        # 宝箱未全部收集时，也给予微弱的终点方向感知
        if prev_end_dist > 0 and treasure_collected_count > 0:
            if end_dist < prev_end_dist:
                reward_end_dist = 0.1
            else:
                reward_end_dist = -0.05

    # 到达终点的超级奖励
    if terminated:
        if treasure_collected_count >= treasure_count:
            reward_win = 10.0   # 全收集通关
        elif treasure_collected_count >= 10:
            reward_win = 5.0 + treasure_collected_count * 0.3
        else:
            reward_win = 2.0 + treasure_collected_count * 0.2

    # ========================================
    # 奖励4: buff加速增益奖励
    # ========================================
    reward_buff = 0.0
    buff_pos = _remain_info.get("buff_pos")
    prev_buff_pos = remain_info.get("buff_pos")
    if hasattr(buff_pos, 'grid_distance') and buff_pos.grid_distance > 0:
        # buff可用时，给予靠近buff的微弱引导
        if hasattr(prev_buff_pos, 'grid_distance') and prev_buff_pos.grid_distance > 0:
            if buff_pos.grid_distance < prev_buff_pos.grid_distance:
                reward_buff = 0.3  # 靠近buff
    
    # ========================================
    # 奖励5: 闪现技能使用奖励
    # ========================================
    reward_flicker = 0.0
    # 检测是否使用了闪现（通过位移距离判断）
    move_dist = ((curr_pos_x - prev_pos_x) ** 2 + (curr_pos_z - prev_pos_z) ** 2) ** 0.5
    if move_dist > 5000:  # 闪现距离约8000，大于5000基本是闪现了
        if is_treasures_remain:
            # 收集宝箱阶段使用闪现
            # 检查是否靠近了某个宝箱
            visible_dists_after = [d for d in treasure_dists if d > 0]
            if visible_dists_after and min(visible_dists_after) < 5:
                reward_flicker = 3.0  # 闪现到宝箱附近，好闪现!
            else:
                reward_flicker = 0.5  # 普通闪现
        else:
            # 奔向终点阶段使用闪现
            if end_dist < prev_end_dist:
                reward_flicker = 3.0  # 闪现靠近终点
            else:
                reward_flicker = -0.5  # 无效闪现

    # ========================================
    # 惩罚1: 撞墙惩罚
    # ========================================
    reward_bump = 0.0
    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)
    if is_bump:
        reward_bump = -2.0  # 强惩罚撞墙

    # ========================================
    # 惩罚2: 步数惩罚 (鼓励效率)
    # ========================================
    reward_step = -0.002  # 每步固定小惩罚
    if not is_treasures_remain:
        # 全部收集后步数惩罚加重，催促到终点
        reward_step = -0.01

    # ========================================
    # 惩罚3: 重复探索惩罚
    # ========================================
    reward_memory = 0.0
    memory_map = remain_info.get("memory_map")
    center_memory = memory_map[len(memory_map) // 2]
    if center_memory > 0:
        reward_memory = -0.02 * center_memory  # 按记忆强度惩罚

    # ========================================
    # 奖励6: 探索新区域奖励
    # ========================================
    reward_exploration = 0.0
    recent_position_map = remain_info.get("recent_position_map")
    hero_grid_pos = convert_pos_to_grid_pos(curr_pos_x, curr_pos_z)

    if (hero_grid_pos[0], hero_grid_pos[1]) not in recent_position_map:
        # 到达全新位置
        reward_exploration = 0.3
    else:
        pass_times = recent_position_map[(hero_grid_pos[0], hero_grid_pos[1])]
        # 指数衰减的重复惩罚
        reward_exploration = max(-0.1 * (1.5 ** pass_times), -5.0)

    # ========================================
    # 惩罚4: 超时惩罚
    # ========================================
    reward_timeout = 0.0
    if truncated:
        reward_timeout = -10.0

    # ========================================
    # 汇总所有奖励
    # ========================================
    total_reward = (
        reward_treasure +        # 宝箱收集
        reward_treasure_dist +   # 靠近宝箱
        reward_end_dist +        # 靠近终点
        reward_win +             # 通关
        reward_buff +            # buff
        reward_flicker +         # 闪现
        reward_bump +            # 撞墙惩罚
        reward_step +            # 步数惩罚
        reward_memory +          # 重复记忆惩罚
        reward_exploration +     # 探索奖励
        reward_timeout           # 超时惩罚
    )

    return (
        total_reward,
        is_bump,
        reward_end_dist,
        reward_exploration,
        reward_treasure_dist,
        reward_treasure,
    )


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


# SampleData <----> NumpyData
@attached
def SampleData2NumpyData(g_data):
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
    obs_data_size = (2 * (Config.VIEW_SIZE) + 1) ** 2 * 4 + 404
    return SampleData(
        # Refer to the DESC_OBS_SPLIT configuration in config.py for dimension reference
        # 维度参考config.py 中的 DESC_OBS_SPLIT配置
        obs=s_data[:obs_data_size],
        _obs=s_data[obs_data_size : 2 * obs_data_size],
        obs_legal=s_data[-8:-6],
        _obs_legal=s_data[-6:-4],
        act=s_data[-4],
        rew=s_data[-3],
        ret=s_data[-2],
        done=s_data[-1],
    )
