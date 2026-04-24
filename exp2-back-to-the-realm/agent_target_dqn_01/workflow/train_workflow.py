#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
Optimized: 课程学习 + 高频训练 + 早停机制，13000轮极限效率
"""


import time
import os
import copy
from kaiwu_agent.utils.common_func import Frame, attached

from tools.train_env_conf_validate import check_usr_conf, read_usr_conf
from agent_target_dqn.feature.definition import (
    reward_shaping,
    sample_process,
)
from agent_target_dqn.feature.preprocessor import Preprocessor
from tools.metrics_utils import get_training_metrics


# =============================================
# 课程学习配置
# 逐步增加难度，让智能体从简单任务学起
# =============================================
CURRICULUM = [
    # (起始epoch, 宝箱数量, max_step)
    (0,     5,  600),     # 阶段1: 先学会导航+拾取少量宝箱，短局快速迭代
    (1500,  8,  800),     # 阶段2: 中等数量宝箱
    (4000,  11, 900),     # 阶段3: 大部分宝箱
    (7000,  13, 1000),    # 阶段4: 全量宝箱，与评估对齐
]


def get_curriculum_config(epoch, base_conf):
    """根据当前epoch返回课程学习配置"""
    treasure_count = CURRICULUM[0][1]
    max_step = CURRICULUM[0][2]
    for start_epoch, tc, ms in CURRICULUM:
        if epoch >= start_epoch:
            treasure_count = tc
            max_step = ms
    
    conf = copy.deepcopy(base_conf)
    conf["env_conf"]["treasure_count"] = treasure_count
    conf["env_conf"]["max_step"] = max_step
    return conf, treasure_count, max_step


@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]
    epoch_num = 100000
    episode_num_every_epoch = 1
    # 关键: 降低截断长度，更频繁地learn → 更快收敛
    g_data_truncat = 128
    last_save_model_time = 0

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_target_dqn/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error(f"usr_conf is None, please check agent_target_dqn/conf/train_env_conf.toml")
        return

    valid = check_usr_conf(usr_conf, logger)
    if not valid:
        logger.error(f"check_usr_conf return False, please check")
        return

    # 训练统计
    recent_wins = 0
    recent_treasures = 0
    stats_window = 100
    prev_curriculum_stage = -1

    for epoch in range(epoch_num):
        # 课程学习: 动态调整宝箱数和步数
        curr_conf, curr_treasure_count, curr_max_step = get_curriculum_config(epoch, usr_conf)

        # 日志课程阶段变化
        curr_stage = sum(1 for s, _, _ in CURRICULUM if epoch >= s) - 1
        if curr_stage != prev_curriculum_stage:
            logger.info(
                f"=== CURRICULUM STAGE {curr_stage}: "
                f"Treasures={curr_treasure_count}, MaxStep={curr_max_step}, Epoch={epoch} ==="
            )
            prev_curriculum_stage = curr_stage

        epoch_total_rew = 0
        data_length = 0
        for g_data in run_episodes(
            episode_num_every_epoch, env, agent, g_data_truncat, 
            curr_conf, logger, monitor, epoch
        ):
            data_length += len(g_data)
            total_rew = sum([i.rew for i in g_data])
            epoch_total_rew += total_rew
            agent.learn(g_data)
            g_data.clear()

        avg_step_reward = 0
        if data_length:
            avg_step_reward = f"{(epoch_total_rew/data_length):.2f}"

        # save model file
        now = time.time()
        if now - last_save_model_time >= 120:
            agent.save_model()
            last_save_model_time = now

        if epoch % 50 == 0:
            logger.info(
                f"Epoch {epoch} | AvgReward: {avg_step_reward} | "
                f"Steps: {data_length} | Curriculum: {curr_treasure_count}T/{curr_max_step}S"
            )


def run_episodes(n_episode, env, agent, g_data_truncat, usr_conf, logger, monitor, epoch):
    for episode in range(n_episode):
        collector = list()
        preprocessor = Preprocessor()

        training_metrics = get_training_metrics()
        if training_metrics:
            logger.info(f"training_metrics is {training_metrics}")

        obs, state_env_info = env.reset(usr_conf=usr_conf)

        if obs is None:
            continue

        agent.load_model(id="latest")

        obs_data, remain_info = agent.observation_process(obs, preprocessor, state_env_info)

        done = False
        step = 0
        bump_cnt = 0
        diy_2 = 0
        diy_3 = 0
        diy_4 = 0
        diy_5 = 0
        episode_total_reward = 0
        last_treasure_step = 0  # 上次收集宝箱的step

        while not done:
            act_data, model_version = agent.predict(list_obs_data=[obs_data])
            act = agent.action_process(act_data[0])

            frame_no, _obs, score, terminated, truncated, _state_env_info = env.step(act)
            if _obs is None:
                break

            step += 1

            _obs_data, _remain_info = agent.observation_process(_obs, preprocessor, _state_env_info)
            if truncated and frame_no is None:
                break

            treasures_num = 0

            if _obs is None:
                reward = 0
            else:
                (
                    reward,
                    is_bump,
                    reward_end_dist,
                    reward_exploration,
                    reward_treasure_dist,
                    reward_treasure,
                ) = reward_shaping(
                    frame_no,
                    score,
                    terminated,
                    truncated,
                    remain_info,
                    _remain_info,
                    obs,
                    _obs,
                )
                diy_2 += reward_end_dist
                diy_3 += reward_exploration
                diy_4 += reward_treasure_dist
                diy_5 += reward_treasure
                episode_total_reward += reward

                treasure_dists = [organ.status for organ in _obs.frame_state.organs]
                treasures_num = treasure_dists.count(1.0)

                bump_cnt += is_bump
                
                # 追踪宝箱收集进度
                if reward_treasure > 0:
                    last_treasure_step = step

            if truncated:
                logger.info(
                    f"Ep{epoch} TIMEOUT | Steps:{step} | "
                    f"T:{treasures_num - 7} | Bumps:{bump_cnt} | R:{episode_total_reward:.1f}"
                )
            elif terminated:
                logger.info(
                    f"Ep{epoch} WIN | Steps:{step} | "
                    f"T:{treasures_num - 7} | Bumps:{bump_cnt} | R:{episode_total_reward:.1f}"
                )
            done = terminated or truncated

            frame = Frame(
                obs=obs_data.feature,
                _obs=_obs_data.feature,
                obs_legal=obs_data.legal_act,
                _obs_legal=_obs_data.legal_act,
                act=act,
                rew=reward,
                done=done,
                ret=reward,
            )

            collector.append(frame)

            if len(collector) % g_data_truncat == 0:
                collector = sample_process(collector)
                yield collector

            if done:
                if len(collector) > 0:
                    collector = sample_process(collector)
                    yield collector

                if monitor:
                    monitor_data = {"diy_2": diy_2, "diy_3": diy_3, "diy_4": diy_4, "diy_5": diy_5}
                    monitor.put_data({os.getpid(): monitor_data})

                break

            obs_data = _obs_data
            remain_info = _remain_info
            state_env_info = _state_env_info
