#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import attached
import time
import os
from collections import deque
from tools.map_data_utils import read_map_data
from tools.train_env_conf_validate import check_usr_conf, read_usr_conf
from tools.metrics_utils import get_training_metrics


TREASURE_ID_TO_POS = {
    0: (19, 14),
    1: (9, 28),
    2: (9, 44),
    3: (42, 45),
    4: (32, 23),
    5: (49, 56),
    6: (35, 58),
    7: (23, 55),
    8: (41, 33),
    9: (54, 41),
}

ALL_TREASURE_IDS = list(range(10))
TREASURE_REWARD = 100
# Strongly discourage ending the episode before collecting all configured treasures.
INCOMPLETE_FINISH_PENALTY = 1000
# Give a terminal bonus if all configured treasures are collected before finishing.
ALL_TREASURE_FINISH_BONUS = 1000


def _pos_to_state(pos):
    return int(pos[0] * 64 + pos[1])


def _parse_enabled_treasure_ids(usr_conf):
    env_conf = usr_conf.get("env_conf")
    if env_conf is None:
        # Keep compatibility with distributed workflow config layout.
        env_conf = usr_conf.get("diy", {}).get("start", [{}])[0]

    treasure_ids = env_conf.get("treasure_id", [])
    treasure_random = env_conf.get("treasure_random", False)

    # Dynamic programming training uses a fixed transition graph; random treasure placement is not modeled.
    # In random mode we still encode all 10 treasure locations for deterministic planning.
    if treasure_random:
        return ALL_TREASURE_IDS

    if not treasure_ids:
        return ALL_TREASURE_IDS

    valid_ids = [int(t_id) for t_id in treasure_ids if int(t_id) in TREASURE_ID_TO_POS]
    return valid_ids or ALL_TREASURE_IDS


def _parse_start_state(usr_conf):
    env_conf = usr_conf.get("env_conf")
    if env_conf is None:
        env_conf = usr_conf.get("diy", {}).get("start", [{}])[0]

    start = env_conf.get("start", [29, 9])
    if not isinstance(start, (list, tuple)) or len(start) != 2:
        start = [29, 9]

    return _pos_to_state((int(start[0]), int(start[1])))


def _build_augmented_map_data(base_map_data, enabled_treasure_ids, start_pos_state):
    treasure_pos_to_bit = {
        _pos_to_state(TREASURE_ID_TO_POS[t_id]): t_id for t_id in enabled_treasure_ids
    }
    enabled_mask = 0
    for t_id in enabled_treasure_ids:
        enabled_mask |= 1 << t_id

    init_aug_state = 1024 * int(start_pos_state) + enabled_mask

    augmented_map_data = {}
    visited = {init_aug_state}
    queue = deque([init_aug_state])

    # Build only states reachable from the configured start and initial treasure mask.
    while queue:
        aug_state = queue.popleft()
        pos = aug_state // 1024
        mask = aug_state % 1024

        actions = base_map_data.get(str(pos), {})
        aug_actions = {}
        for action, transition in actions.items():
            next_pos, reward, done = transition
            next_mask = mask

            if next_pos in treasure_pos_to_bit:
                bit = treasure_pos_to_bit[next_pos]
                if (mask >> bit) & 1:
                    next_mask &= ~(1 << bit)
                    reward += TREASURE_REWARD

            if done:
                if next_mask != 0:
                    reward -= INCOMPLETE_FINISH_PENALTY
                else:
                    reward += ALL_TREASURE_FINISH_BONUS

            next_state = 1024 * int(next_pos) + next_mask
            aug_actions[action] = [next_state, reward, done]

            if not done and next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)

        augmented_map_data[str(aug_state)] = aug_actions

    return augmented_map_data


@attached
def workflow(envs, agents, logger=None, monitor=None):

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_dynamic_programming/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_dynamic_programming/conf/train_env_conf.toml")
        return

    # check_usr_conf is a tool to check whether the game configuration is correct
    # It is recommended to perform a check before calling reset.env
    # check_usr_conf会检查游戏配置是否正确，建议调用reset.env前先检查一下
    valid = check_usr_conf(usr_conf, logger)
    if not valid:
        logger.error("check_usr_conf return False, please check")
        return
    env, agent = envs[0], agents[0]

    # Initializing monitoring data
    # 监控数据初始化
    monitor_data = {
        "reward": 0,
        "diy_1": 0,
        "diy_2": 0,
        "diy_3": 0,
        "diy_4": 0,
        "diy_5": 0,
    }

    logger.info("Start Training...")
    start_t = time.time()

    # Setting the state transition function
    # 设置状态转移函数
    map_data_file = "conf/map_data/F_level_1.json"
    map_data = read_map_data(map_data_file)
    if map_data is None:
        logger.error(f"map_data from file {map_data_file} failed, please check")
        return

    enabled_treasure_ids = _parse_enabled_treasure_ids(usr_conf)
    start_pos_state = _parse_start_state(usr_conf)
    augmented_map_data = _build_augmented_map_data(map_data, enabled_treasure_ids, start_pos_state)

    logger.info(f"Dynamic programming enabled treasure ids: {enabled_treasure_ids}")
    logger.info(f"Dynamic programming start state: {start_pos_state}")
    logger.info(f"Augmented transition states: {len(augmented_map_data)}")

    agent.learn(augmented_map_data)

    logger.info(f"Training time cost: {time.time() - start_t} s")

    # Reporting training progress
    # 上报训练进度
    monitor_data["reward"] = 0
    if monitor:
        monitor.put_data({os.getpid(): monitor_data})

    # model saving
    # 保存模型
    agent.save_model()
    # Retrieving training metrics
    # 获取训练中的指标
    training_metrics = get_training_metrics()
    if training_metrics:
        logger.info(f"training_metrics is {training_metrics}")

    return
