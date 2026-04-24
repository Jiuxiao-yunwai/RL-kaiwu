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

    def _delta_in_grid_units(prev_dist, curr_dist):
        """
        Convert distance delta into comparable grid units.
        将距离差统一换算为可比较的“网格单位”。

        - If data appears normalized (typically <= 2), convert by *256.
        - If data already looks like raw grid steps, use as-is.
        - Return None when either side is invalid (<0).
        - 若数据看起来是归一化（通常<=2），按 *256 还原网格单位；
        - 若本身像原始网格步数，则直接使用；
        - 任一侧无效（<0）时返回 None。
        """
        if prev_dist < 0 or curr_dist < 0:
            return None

        delta = prev_dist - curr_dist
        if max(prev_dist, curr_dist) <= 2.0:
            return delta * 256.0
        return delta

    # Get the grid-based distance of the current agent's position relative to the end point, buff, and treasure chest
    # 获取当前智能体的位置相对于终点, buff, 宝箱的栅格化距离
    # Use continuous path-aware distance for reward shaping.
    # 使用连续且路径相关的 grid_distance 进行奖励塑形。
    end_dist = _remain_info.get("end_pos").grid_distance
    treasure_dists = [pos.grid_distance if pos.grid_distance > 0 else -1 for pos in _remain_info.get("treasure_pos")]
    treasure_count = _remain_info.get("treasure_count")
    treasure_collected_count = _remain_info.get("treasure_collected_count")
    treasure_total = max(int(treasure_count), 1)
    collection_ratio = float(treasure_collected_count) / float(treasure_total)
    # frame_no is frame index and 1 step ~= 3 frames in this env.
    # frame_no是帧编号，本环境中约3帧=1步。
    est_step = max(float(frame_no) / 3.0, 0.0) if frame_no is not None else 0.0
    # max_step comes from env config; fallback to documentation default(2000).
    # max_step来自环境配置；缺省回退到文档默认值2000。
    max_step_cfg = float(getattr(_obs_data.game_info, "max_step", 2000))
    step_efficiency = max((max_step_cfg - est_step) / max(max_step_cfg, 1.0), 0.0)

    # Get the agent's position from the previous frame
    # 获取智能体上一帧的位置
    prev_pos = obs_data.frame_state.heroes[0].pos
    prev_pos_x, prev_pos_z = prev_pos.x, prev_pos.z

    # Get the grid-based distance of the agent's position from the previous
    # frame relative to the end point, buff, and treasure chest
    # 获取智能体上一帧相对于终点，buff, 宝箱的栅格化距离
    prev_end_dist = remain_info.get("end_pos").grid_distance
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
    # Stage mode:
    # - treasure_phase=True: prioritize collecting treasures.
    # - treasure_phase=False: switch to finish phase and push to end point.
    # 阶段模式：
    # - treasure_phase=True：优先收集宝箱；
    # - treasure_phase=False：切到收官阶段，主推终点。
    treasure_phase = bool(is_treasures_remain)

    """
    Reward 1. Reward related to the end point
    奖励1. 与终点相关的奖励
    """
    reward_end_dist = 0
    # Reward 1.1 Reward for getting closer to the end point
    # 奖励1.1 向终点靠近的奖励
    # Keep this term active for the whole episode to prioritize reaching end first.
    # 该项在全程生效，用于明确“优先到达终点”的训练目标。
    delta_end_grid = _delta_in_grid_units(prev_end_dist, end_dist)
    if delta_end_grid is not None:
        # Old implementation (binary):
        # reward_end_dist = 1 if end_dist < prev_end_dist else -1
        # 旧实现（二值）：
        # reward_end_dist = 1 if end_dist < prev_end_dist else -1
        #
        # Problem:
        # - It penalizes all non-improving steps equally, including side moves
        #   that are necessary near obstacles, which may destabilize training.
        # 问题：
        # - 旧逻辑会把所有“非变好”动作一律重罚，
        #   包括绕障碍时必要的横移，容易导致学习震荡。
        #
        # New implementation:
        # - Use distance delta as a continuous signal.
        # - Add a deadband to avoid tiny numerical/geometry jitters being treated as regressions.
        # - Clip reward magnitude for stability.
        # 新实现：
        # - 使用距离变化量作为连续信号；
        # - 增加容忍区间，避免微小抖动被误判为退步；
        # - 限幅保证训练稳定。
        # Deadband is interpreted in grid units.
        # 死区按网格单位解释。
        deadband_grid = 0.1
        if abs(delta_end_grid) <= deadband_grid:
            reward_end_dist = 0
        else:
            reward_end_dist = max(min(delta_end_grid / 2.0, 1.0), -1.0)

    # Reward 1.2 Reward for winning
    # 奖励1.2 获胜的奖励
    reward_win = 0
    if terminated:
        # Align with official scoring intent:
        # score = end(150) + treasure(100 * collected) + step_bonus(0.2 * (max_step - used_step)).
        # We use normalized components to avoid exploding scale in RL training.
        # 对齐官方得分意图：
        # score = 终点(150) + 宝箱(100*收集数) + 步数奖励(0.2*(max_step-used_step))。
        # 这里用归一化形式，避免RL训练中数值过大不稳定。
        # Stage-aware terminal reward:
        # - If treasures are not fully collected, reaching end early should not be strongly rewarded.
        # - If all treasures are collected, give strong finish reward with efficiency bonus.
        # 分阶段终局奖励：
        # - 未收齐宝箱就到终点，不应给高奖励；
        # - 收齐后到终点，给较强收官奖励并叠加效率收益。
        if treasure_phase:
            reward_win = 0.2
        else:
            reward_win = 1.5 + 1.0 * collection_ratio + 0.5 * step_efficiency

    # Reward 1.3 Timeout penalty
    # 奖励1.3 超时惩罚
    reward_timeout = 1 if truncated else 0

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
            # Old implementation was binary +/-1.
            # 旧实现是二值 +/-1。
            # Here we make it continuous and robust to tiny geometry jitters.
            # 这里改成连续奖励，并加入微小抖动容忍。
            prev_visible_dists = [d for d in prev_treasure_dists if d > 0]
            if prev_visible_dists:
                curr_min_dist = min(visible_dists)
                prev_min_dist = min(prev_visible_dists)
                delta_treasure_grid = _delta_in_grid_units(prev_min_dist, curr_min_dist)
                deadband_grid = 0.1
                if delta_treasure_grid is None:
                    reward_treasure_dist = 0
                elif abs(delta_treasure_grid) <= deadband_grid:
                    reward_treasure_dist = 0
                else:
                    reward_treasure_dist = max(min(delta_treasure_grid / 2.0, 1.0), -1.0)
            else:
                # If treasure just appears in view, give a small positive shaping signal.
                # 若宝箱刚进入视野，给小幅正向塑形奖励。
                reward_treasure_dist = 0.3

    # Objective shielding:
    # If agent is effectively moving toward treasure, temporarily shield negative end-distance shaping.
    # 目标屏蔽：
    # 当智能体正在有效靠近宝箱时，临时屏蔽“远离终点”的负塑形，避免目标冲突。
    if treasure_phase and reward_treasure_dist > 0:
        # While effectively approaching treasure, suppress end-distance shaping
        # to prevent objective conflict.
        # 在寻宝阶段且正在有效靠近宝箱时，屏蔽终点距离塑形，避免目标冲突。
        reward_end_dist = 0

    if treasure_phase and reward_end_dist > 0:
        # In treasure phase, weaken positive pull-to-end to avoid premature finish.
        # 寻宝阶段弱化“拉向终点”的正向塑形，避免过早收官。
        reward_end_dist *= 0.2

    # Reward 2.2 Reward for getting the treasure chest
    # 奖励2.2 获得宝箱的奖励
    reward_treasure = 0
    if treasure_collected_count > prev_treasure_collected_count:
        reward_treasure = 1

    # Reward 2.3 Bonus when all treasures are collected for the first time in the episode
    # 奖励2.3 对局内首次“全宝箱收集完成”奖励
    reward_all_treasure = 0
    if treasure_count > 0 and prev_treasure_collected_count < treasure_count and treasure_collected_count == treasure_count:
        reward_all_treasure = 1

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
    # We intentionally neutralize the old hard penalty to avoid false positives during wall-following.
    # 这里有意中和旧的硬惩罚，避免贴墙绕障时被误判为退步。
    if (not is_treasures_remain) and (delta_end_grid is not None) and (delta_end_grid < -1.5):
        reward_post_treasure_to_end = 1

    # Reward 5.1.1 Terminal treasure consistency
    # 奖励5.1.1 终局宝箱一致性奖励/惩罚
    # Perfect clear bonus: reach end and collect all treasures.
    # 完美通关奖励：到达终点且收齐宝箱。
    reward_perfect_clear = 0
    if terminated and treasure_count > 0 and treasure_collected_count == treasure_count:
        reward_perfect_clear = 1

    # If reach end without full collection, apply a mild penalty (not too strong, keep end-first objective).
    # 若到终点但未全收集，给轻微惩罚（不宜过强，避免破坏终点优先）。
    reward_missing_treasure_on_finish = 0
    if terminated and treasure_count > 0 and treasure_collected_count < treasure_count:
        reward_missing_treasure_on_finish = (treasure_count - treasure_collected_count) / float(treasure_count)

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
        # Priority order follows task scoring semantics:
        # 1) reach end; 2) collect treasures; 3) be efficient.
        # 优先级按任务计分语义：
        # 1) 到终点；2) 收宝箱；3) 高效率。
        "reward_end_dist": 0.9,
        "reward_win": 4.0,
        "reward_timeout": -4.5,
        # Buff/flicker are auxiliary behaviors; keep small to avoid objective drift.
        # buff/闪现是辅助行为，权重保持较小，避免目标漂移。
        "reward_buff_dist": 0.05,
        "reward_buff": 0.1,
        "reward_treasure_dists": 0.5,
        "reward_treasure": 1.2,
        "reward_all_treasure": 1.5,
        "reward_flicker": 0.05,
        "reward_step": -0.001,
        "reward_bump": -1.0,
        "reward_memory": -0.01,
        "reward_exploration": 0.05,
        # Keep this weight at 0 to remove brittle wall-following false positives.
        # 将该权重置0，移除易误判的贴墙绕障惩罚。
        "reward_post_treasure_to_end": 0.0,
        "reward_perfect_clear": 2.5,
        "reward_missing_treasure_on_finish": -0.3,
        "reward_stall": -0.2,
        "reward_revisit": -0.03,
    }

    # Phase-based objective scheduling:
    # Treasure phase: strongly encourage treasure collection and discourage early finish.
    # Finish phase: strongly encourage reaching end after treasures are collected.
    # 分阶段目标调度：
    # 寻宝阶段：强化寻宝，抑制提前收官；
    # 收官阶段：在收齐后强化到终点。
    if treasure_phase:
        reward_weight["reward_end_dist"] = 0.15
        reward_weight["reward_win"] = 0.8
        reward_weight["reward_treasure_dists"] = 1.4
        reward_weight["reward_treasure"] = 2.2
        reward_weight["reward_all_treasure"] = 3.2
        reward_weight["reward_missing_treasure_on_finish"] = -3.0
    else:
        reward_weight["reward_end_dist"] = 1.2
        reward_weight["reward_win"] = 4.5
        reward_weight["reward_treasure_dists"] = 0.2
        reward_weight["reward_treasure"] = 0.5
        reward_weight["reward_all_treasure"] = 0.5
        reward_weight["reward_missing_treasure_on_finish"] = -0.2

    reward = [
        reward_end_dist * reward_weight["reward_end_dist"],
        reward_win * reward_weight["reward_win"],
        reward_timeout * reward_weight["reward_timeout"],
        reward_buff_dist * reward_weight["reward_buff_dist"],
        reward_buff * reward_weight["reward_buff"],
        reward_treasure_dist * reward_weight["reward_treasure_dists"],
        reward_treasure * reward_weight["reward_treasure"],
        reward_all_treasure * reward_weight["reward_all_treasure"],
        reward_flicker * reward_weight["reward_flicker"],
        reward_step * reward_weight["reward_step"],
        reward_bump * reward_weight["reward_bump"],
        reward_memory * reward_weight["reward_memory"],
        reward_exploration * reward_weight["reward_exploration"],
        reward_post_treasure_to_end * reward_weight["reward_post_treasure_to_end"],
        reward_perfect_clear * reward_weight["reward_perfect_clear"],
        reward_missing_treasure_on_finish * reward_weight["reward_missing_treasure_on_finish"],
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
