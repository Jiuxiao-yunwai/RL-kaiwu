#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
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


def clipped_progress_reward(prev_dist, curr_dist, clip_value=1.0):
    """
    Convert a distance change into a bounded dense reward.
    将距离变化转换成有界的稠密奖励。
    """
    if prev_dist is None or curr_dist is None:
        return 0.0
    if prev_dist < 0 or curr_dist < 0:
        return 0.0

    return float(np.clip(prev_dist - curr_dist, -clip_value, clip_value))


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
    # Get the current position coordinates of the agent
    # 获取当前智能体的位置坐标
    pos = _obs_data.frame_state.heroes[0].pos
    curr_pos_x, curr_pos_z = pos.x, pos.z

    # The total_score in observation is aligned with the real task objective:
    # reaching the end and collecting treasure are both reflected here,
    # and timeout episodes will naturally lose those gains.
    # observation 里的 total_score 和真实任务目标一致：
    # 到终点、拿宝箱都会体现出来，超时也会自然丢掉这些收益。
    prev_total_score = obs_data.game_info.total_score
    curr_total_score = _obs_data.game_info.total_score
    reward_score = (curr_total_score - prev_total_score) / 100.0

    # Get the current agent's distance to the end point and treasure chests
    # 获取当前智能体到终点和宝箱的距离
    end_dist = _remain_info.get("end_pos").l2_distance
    treasure_dists = [pos.grid_distance if pos.grid_distance > 0 else -1 for pos in _remain_info.get("treasure_pos")]
    treasure_count = _remain_info.get("treasure_count")
    treasure_collected_count = _remain_info.get("treasure_collected_count")

    # Get the agent's position from the previous frame
    # 获取智能体上一帧的位置
    prev_pos = obs_data.frame_state.heroes[0].pos
    prev_pos_x, prev_pos_z = prev_pos.x, prev_pos.z

    # Get the previous frame's distance to the end point and treasure chests
    # 获取智能体上一帧到终点和宝箱的距离
    prev_end_dist = remain_info.get("end_pos").l2_distance
    prev_treasure_dists = [
        pos.grid_distance if pos.grid_distance > 0 else -1 for pos in remain_info.get("treasure_pos")
    ]
    prev_treasure_collected_count = remain_info.get("treasure_collected_count")

    # Are there any remaining treasure chests
    # 是否有剩余宝箱
    is_treasures_remain = treasure_collected_count < treasure_count

    """
    Reward 1. Reward related to the end point
    奖励1. 与终点相关的奖励
    """
    reward_end_dist = clipped_progress_reward(prev_end_dist, end_dist)

    # Reward 1.2 Bonus for completing the task successfully
    # 奖励1.2 成功通关奖励
    reward_success = 1.0 if terminated else 0.0

    # Reward 1.3 Penalty for timeout / interruption
    # 奖励1.3 超时或中断惩罚
    reward_timeout = 1.0 if truncated else 0.0

    """
    Reward 2. Rewards related to the treasure chest
    奖励2. 与宝箱相关的奖励
    """
    reward_treasure_dist = 0.0
    has_visible_treasure = False

    # Reward 2.1 Reward for getting closer to the nearest visible treasure chest
    # 奖励2.1 向最近可见宝箱靠近的奖励
    if is_treasures_remain:
        visible_indices = [idx for idx, dist in enumerate(treasure_dists) if dist > 0]
        has_visible_treasure = len(visible_indices) > 0

        # Only use visible treasure chests for dense shaping to avoid noisy signals
        # 稠密奖励只使用可见宝箱，避免远处不可达目标带来噪声
        visible_dists = [treasure_dists[idx] for idx in visible_indices]
        if visible_dists:
            min_dist = min(visible_dists)
            min_index = treasure_dists.index(min_dist)
            prev_min_dist = prev_treasure_dists[min_index]
            reward_treasure_dist = clipped_progress_reward(prev_min_dist, min_dist)

    # Reward 2.2 Reward for getting the treasure chest
    # 奖励2.2 获得宝箱的奖励
    reward_treasure = max(0, treasure_collected_count - prev_treasure_collected_count)

    """
    Reward 3. Rewards related to the buff
    奖励3. 与buff相关的奖励
    """
    # Reward 3.1 Reward for getting closer to the buff
    # 奖励3.1 靠近buff的奖励 (TODO)
    reward_buff_dist = 0

    # Reward 3.2 Reward for getting the buff
    # 奖励3.2 获得buff的奖励 (TODO)
    reward_buff = 0

    """
    Reward 4. Rewards related to the flicker
    奖励4. 与闪现相关的奖励
    """
    reward_flicker = 0
    # Reward 4.1 Penalty for flickering into the wall (TODO)
    # 奖励4.1 撞墙闪现的惩罚 (TODO)

    # Reward 4.2 Reward for normal flickering (TODO)
    # 奖励4.2 正常闪现的奖励 (TODO)

    # Reward 4.3 Reward for super flickering (TODO)
    # 奖励4.3 超级闪现的奖励 (TODO)

    """
    Reward 5. Rewards for quick clearance
    奖励5. 关于快速通关的奖励
    """
    reward_step = 1
    # Reward 5.1 Penalty for not getting close to the end point after collecting all the treasure chests
    # (TODO: Give penalty after collecting all the treasure chests, encourage full collection)
    # 奖励5.1 收集完所有宝箱却未靠近终点的惩罚
    # (TODO: 收集完宝箱后再给予惩罚, 鼓励宝箱全收集)

    # Reward 5.2 Penalty for repeated exploration
    # 奖励5.2 重复探索的惩罚
    reward_memory = 0

    # Reward 5.3 Penalty for bumping into the wall
    # 奖励5.3 撞墙的惩罚
    reward_bump = 0.0
    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)
    # Determine whether it bumps into the wall
    # 判断是否撞墙
    if is_bump:
        # Give a relatively large penalty for bumping into the wall,
        # so that the agent can learn not to bump into the wall as soon as possible
        # 对撞墙给予一个比较大的惩罚，以便agent能够尽快学会不撞墙
        reward_bump = 1.0

    # Exploration Reward
    # 探索奖励
    reward_exploration = 0.0
    recent_position_map = remain_info.get("recent_position_map")
    hero_grid_pos = convert_pos_to_grid_pos(curr_pos_x, curr_pos_z)
    pass_times = recent_position_map.get((hero_grid_pos[0], hero_grid_pos[1]), 0)

    if pass_times <= 1:
        reward_exploration = 1.0
    else:
        reward_exploration = max(1.0 - 0.35 * pass_times, -1.5)

    # When treasure is already visible, exploration should step back and let
    # the policy focus on collecting the treasure or finishing the route.
    # 当宝箱已经可见时，探索奖励应适当退让，让策略更专注于拿宝箱或走终点。
    if has_visible_treasure:
        reward_exploration *= 0.25
    elif not is_treasures_remain:
        reward_exploration *= 0.1

    end_progress_weight = 0.7 if not is_treasures_remain else 0.45
    treasure_progress_weight = 0.35 if has_visible_treasure else 0.0
    exploration_weight = 0.08 if is_treasures_remain and not has_visible_treasure else 0.02

    """
    Concatenation of rewards: Here are 10 rewards provided,
    students can concatenate as needed, and can also add new rewards themselves
    奖励的拼接: 这里提供了10个奖励, 同学们按需自行拼接, 也可以自行添加新的奖励
    """
    reward_weight = {
        "reward_score": 1.0,
        "reward_end_dist": end_progress_weight,
        "reward_success": 2.0,
        "reward_timeout": -6.0,
        "reward_buff_dist": 0,
        "reward_buff": 0,
        "reward_treasure_dists": treasure_progress_weight,
        "reward_treasure": 0.4,
        "reward_flicker": 0,
        "reward_step": -0.002,
        "reward_bump": -1.0,
        "reward_memory": 0,
        "reward_exploration": exploration_weight,
    }

    reward = [
        reward_score * reward_weight["reward_score"],
        reward_end_dist * reward_weight["reward_end_dist"],
        reward_success * reward_weight["reward_success"],
        reward_timeout * reward_weight["reward_timeout"],
        reward_buff_dist * reward_weight["reward_buff_dist"],
        reward_buff * reward_weight["reward_buff"],
        reward_treasure_dist * reward_weight["reward_treasure_dists"],
        reward_treasure * reward_weight["reward_treasure"],
        reward_flicker * reward_weight["reward_flicker"],
        reward_step * reward_weight["reward_step"],
        reward_bump * reward_weight["reward_bump"],
        reward_memory * reward_weight["reward_memory"],
        reward_exploration * reward_weight["reward_exploration"],
    ]

    return (
        sum(reward),
        is_bump,
        reward_end_dist * reward_weight["reward_end_dist"],
        reward_exploration * reward_weight["reward_exploration"],
        reward_treasure_dist * reward_weight["reward_treasure_dists"],
        reward_treasure * reward_weight["reward_treasure"],
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
