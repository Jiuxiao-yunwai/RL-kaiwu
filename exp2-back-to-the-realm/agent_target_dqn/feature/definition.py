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
    reward = 0

    # Get the current position coordinates of the agent
    # 获取当前智能体的位置坐标
    pos = _obs_data.frame_state.heroes[0].pos
    curr_pos_x, curr_pos_z = pos.x, pos.z
    move_dist = ((curr_pos_x - obs_data.frame_state.heroes[0].pos.x) ** 2 + (curr_pos_z - obs_data.frame_state.heroes[0].pos.z) ** 2) ** 0.5
    curr_grid_pos = convert_pos_to_grid_pos(curr_pos_x, curr_pos_z)

    # Extract buff status from frame data for pickup reward shaping.
    # 从帧数据中提取 buff 状态，用于 buff 拾取奖励。
    # status=1 means buff exists and can be collected; status=0 means unavailable/consumed.
    # status=1 表示 buff 可拾取，status=0 表示不可拾取/已被消耗。
    def _get_buff_status(frame_obs):
        for organ in frame_obs.frame_state.organs:
            if organ.sub_type == 2:
                return organ.status
        return 0

    # Get the grid-based distance of the current agent's position relative to the end point, buff, and treasure chest
    # 获取当前智能体的位置相对于终点, buff, 宝箱的栅格化距离
    end_dist = _remain_info.get("end_pos").l2_distance
    treasure_dists = [pos.grid_distance if pos.grid_distance > 0 else -1 for pos in _remain_info.get("treasure_pos")]
    treasure_count = _remain_info.get("treasure_count")
    treasure_collected_count = _remain_info.get("treasure_collected_count")

    # Get the agent's position from the previous frame
    # 获取智能体上一帧的位置
    prev_pos = obs_data.frame_state.heroes[0].pos
    prev_pos_x, prev_pos_z = prev_pos.x, prev_pos.z

    # Get the grid-based distance of the agent's position from the previous
    # frame relative to the end point, buff, and treasure chest
    # 获取智能体上一帧相对于终点，buff, 宝箱的栅格化距离
    prev_end_dist = remain_info.get("end_pos").l2_distance
    prev_treasure_dists = [
        pos.grid_distance if pos.grid_distance > 0 else -1 for pos in remain_info.get("treasure_pos")
    ]
    prev_treasure_collected_count = remain_info.get("treasure_collected_count")
    prev_buff_dist = remain_info.get("buff_pos").grid_distance
    curr_buff_dist = _remain_info.get("buff_pos").grid_distance
    prev_buff_status = _get_buff_status(obs_data)
    curr_buff_status = _get_buff_status(_obs_data)

    # Are there any remaining treasure chests
    # 是否有剩余宝箱
    is_treasures_remain = treasure_collected_count < treasure_count

    """
    Reward 1. Reward related to the end point
    奖励1. 与终点相关的奖励
    """
    reward_end_dist = 0
    # Reward 1.1 Reward for getting closer to the end point
    # 奖励1.1 向终点靠近的奖励
    if prev_end_dist > 0 and treasure_collected_count > 0:
        reward_end_dist = 1 if end_dist < prev_end_dist else -1

    # Reward 1.2 Reward for winning
    # 奖励1.2 获胜的奖励
    reward_win = 0
    if terminated:
        reward_win = treasure_collected_count

    """
    Reward 2. Rewards related to the treasure chest
    奖励2. 与宝箱相关的奖励
    """
    reward_treasure_dist = 0
    # Reward 2.1 Reward for getting closer to the treasure chest (only consider the nearest one)
    # 奖励2.1 向宝箱靠近的奖励(只考虑最近的那个宝箱)
    if is_treasures_remain:
        # Filter out invisible treasure chests (distance is -1)
        # 过滤掉不可见的宝箱(距离为-1)
        visible_dists = [d for d in treasure_dists if d > 0]
        # If there are visible treasure chests
        # 如果有可见的宝箱
        if visible_dists:
            min_dist = min(visible_dists)
            min_index = treasure_dists.index(min_dist)
            prev_min_dist = prev_treasure_dists[min_index]
            if min_dist < prev_min_dist or prev_min_dist < 0:
                reward_treasure_dist = 1
            else:
                reward_treasure_dist = -1

    # Reward 2.2 Reward for getting the treasure chest
    # 奖励2.2 获得宝箱的奖励
    reward_treasure = 0
    if treasure_collected_count > prev_treasure_collected_count:
        reward_treasure = 1

    """
    Reward 3. Rewards related to the buff
    奖励3. 与buff相关的奖励
    """
    # Reward 3.1 Reward for getting closer to the buff
    # 奖励3.1 靠近buff的奖励 (TODO)
    reward_buff_dist = 0
    # Implemented:
    # - Only shape this term when buff is visible in at least one of the two consecutive frames.
    # - If current buff distance decreases, reward; otherwise small penalty.
    # 已实现：
    # - 仅在前后帧至少有一帧可见 buff 时计算该项；
    # - 与 buff 距离变小给奖励，否则给较小惩罚。
    if prev_buff_dist >= 0 or curr_buff_dist >= 0:
        if prev_buff_dist < 0 and curr_buff_dist >= 0:
            # buff enters local view, encourage approaching it.
            # buff 进入视野，鼓励继续接近。
            reward_buff_dist = 0.5
        elif prev_buff_dist >= 0 and curr_buff_dist >= 0:
            reward_buff_dist = 1 if curr_buff_dist < prev_buff_dist else -0.5

    # Reward 3.2 Reward for getting the buff
    # 奖励3.2 获得buff的奖励 (TODO)
    reward_buff = 0
    # Implemented:
    # reward buff pickup when status changes from available(1) to unavailable(0).
    # 已实现：当 buff 从可拾取(1)切换为不可拾取(0)时给拾取奖励。
    if prev_buff_status == 1 and curr_buff_status == 0:
        reward_buff = 1

    """
    Reward 4. Rewards related to the flicker
    奖励4. 与闪现相关的奖励
    """
    reward_flicker = 0
    is_flicker = move_dist >= 3000
    # Reward 4.1 Penalty for flickering into the wall (TODO)
    # 奖励4.1 撞墙闪现的惩罚 (TODO)
    # Implemented in a decomposed way and merged into reward_flicker:
    # - flicker + bump => explicit penalty.
    # 分解实现并汇总到 reward_flicker：
    # - 闪现且撞墙 => 明确惩罚。
    reward_flicker_bump_penalty = 0
    if is_flicker and bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z):
        reward_flicker_bump_penalty = -1

    # Reward 4.2 Reward for normal flickering (TODO)
    # 奖励4.2 正常闪现的奖励 (TODO)
    # Implemented:
    # - give a small base bonus for successful flicker (not bumping),
    #   because flicker is a scarce action with cooldown.
    # 已实现：
    # - 对“非撞墙”的有效闪现给一个小基础奖励，
    #   因为闪现属于有冷却的稀缺动作。
    reward_flicker_normal = 0
    if is_flicker and not bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z):
        reward_flicker_normal = 0.2

    # Reward 4.3 Reward for super flickering (TODO)
    # 奖励4.3 超级闪现的奖励 (TODO)
    # Implemented:
    # - extra bonus if flicker creates clear progress:
    #   approaching end significantly OR collecting treasure.
    # 已实现：
    # - 若闪现带来显著进展（明显接近终点或拿到宝箱），额外奖励。
    reward_flicker_super = 0
    if is_flicker and (prev_end_dist - end_dist > 2 or treasure_collected_count > prev_treasure_collected_count):
        reward_flicker_super = 1

    reward_flicker = reward_flicker_bump_penalty + reward_flicker_normal + reward_flicker_super

    """
    Reward 5. Rewards for quick clearance
    奖励5. 关于快速通关的奖励
    """
    reward_step = 1
    # Reward 5.1 Penalty for not getting close to the end point after collecting all the treasure chests
    # (TODO: Give penalty after collecting all the treasure chests, encourage full collection)
    # 奖励5.1 收集完所有宝箱却未靠近终点的惩罚
    # (TODO: 收集完宝箱后再给予惩罚, 鼓励宝箱全收集)
    # Implemented:
    # once all treasures are collected, apply penalty when not approaching the end.
    # 已实现：
    # 全宝箱收集后，若没有继续接近终点则给惩罚。
    reward_post_treasure_to_end = 0
    if (not is_treasures_remain) and prev_end_dist > 0 and end_dist >= prev_end_dist:
        reward_post_treasure_to_end = 1

    # Reward 5.2 Penalty for repeated exploration
    # 奖励5.2 重复探索的惩罚
    reward_memory = 0
    memory_map = remain_info.get("memory_map")
    reward_memory = memory_map[len(memory_map) // 2]

    # Reward 5.3 Penalty for bumping into the wall
    # 奖励5.3 撞墙的惩罚
    reward_bump = 0
    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)
    # Determine whether it bumps into the wall
    # 判断是否撞墙
    if is_bump:
        # Give a relatively large penalty for bumping into the wall,
        # so that the agent can learn not to bump into the wall as soon as possible
        # 对撞墙给予一个比较大的惩罚，以便agent能够尽快学会不撞墙
        reward_bump = 1

    # Exploration Reward
    # 探索奖励
    reward_exploration = 0
    recent_position_map = remain_info.get("recent_position_map")
    hero_grid_pos = curr_grid_pos

    if (hero_grid_pos[0], hero_grid_pos[1]) not in recent_position_map:
        reward_exploration = 1
    else:
        pass_times = recent_position_map[(hero_grid_pos[0], hero_grid_pos[1])]
        reward_exploration = max(-0.5 * pass_times, -10)

    # Anti-spin reward terms
    # 防止原地打转的奖励项
    # 1) Penalize low-displacement behavior (standing still or tiny jitter).
    # 1) 对低位移行为（原地不动或抖动）施加惩罚。
    reward_stall = 0
    if move_dist < 150:
        reward_stall = 1

    # 2) Additional revisit penalty for frequent back-and-forth on the same grids.
    #    We use recent_position_map frequency to suppress short loops like A-B-A-B.
    # 2) 对高频回访格子增加惩罚，抑制 A-B-A-B 这类短环路。
    revisit_times = recent_position_map.get((hero_grid_pos[0], hero_grid_pos[1]), 0)
    reward_revisit = max(revisit_times - 1, 0)

    """
    Concatenation of rewards: Here are 10 rewards provided,
    students can concatenate as needed, and can also add new rewards themselves
    奖励的拼接: 这里提供了10个奖励, 同学们按需自行拼接, 也可以自行添加新的奖励
    """
    reward_weight = {
        "reward_end_dist": 0.5,
        "reward_win": 1.0,
        "reward_buff_dist": 0.2,
        "reward_buff": 0.5,
        "reward_treasure_dists": 0.5,
        "reward_treasure": 1.0,
        "reward_flicker": 0.3,
        "reward_step": -0.001,
        "reward_bump": -1.0,
        "reward_memory": -0.01,
        "reward_exploration": 0.05,
        "reward_post_treasure_to_end": -0.3,
        "reward_stall": -0.2,
        "reward_revisit": -0.03,
    }

    reward = [
        reward_end_dist * reward_weight["reward_end_dist"],
        reward_win * reward_weight["reward_win"],
        reward_buff_dist * reward_weight["reward_buff_dist"],
        reward_buff * reward_weight["reward_buff"],
        reward_treasure_dist * reward_weight["reward_treasure_dists"],
        reward_treasure * reward_weight["reward_treasure"],
        reward_flicker * reward_weight["reward_flicker"],
        reward_step * reward_weight["reward_step"],
        reward_bump * reward_weight["reward_bump"],
        reward_memory * reward_weight["reward_memory"],
        reward_exploration * reward_weight["reward_exploration"],
        reward_post_treasure_to_end * reward_weight["reward_post_treasure_to_end"],
        reward_stall * reward_weight["reward_stall"],
        reward_revisit * reward_weight["reward_revisit"],
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
