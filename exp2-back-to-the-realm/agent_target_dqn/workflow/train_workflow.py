#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import time
import os
from kaiwu_agent.utils.common_func import Frame, attached

from tools.train_env_conf_validate import check_usr_conf, read_usr_conf
from agent_target_dqn.feature.definition import (
    reward_shaping,
    sample_process,
)
from agent_target_dqn.feature.preprocessor import Preprocessor
from tools.metrics_utils import get_training_metrics


TRAINING_METRICS_LOG_INTERVAL = 300


def _get_collected_treasure_stats(obs):
    if obs is None or not hasattr(obs, "game_info"):
        return None, None

    game_info = obs.game_info
    collected = getattr(game_info, "treasure_collected_count", None)
    total = getattr(game_info, "treasure_count", None)
    return collected, total


def _try_load_latest_model(agent, logger):
    try:
        agent.load_model(id="latest")
        return True
    except Exception as exc:
        logger.warning(f"load latest model skipped: {exc}")
        return False


def _maybe_log_training_metrics(logger, last_log_time):
    now = time.time()
    if now - last_log_time < TRAINING_METRICS_LOG_INTERVAL:
        return last_log_time

    training_metrics = get_training_metrics()
    if training_metrics:
        logger.info(f"training_metrics is {training_metrics}")
    return now


@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]
    epoch_num = 100000
    episode_num_every_epoch = 1
    g_data_truncat = 128
    last_save_model_time = 0
    last_metrics_log_time = 0

    base_usr_conf = read_usr_conf("agent_target_dqn/conf/train_env_conf.toml", logger)
    if base_usr_conf is None:
        logger.error(f"usr_conf is None, please check agent_target_dqn/conf/train_env_conf.toml")
        return

    valid = check_usr_conf(base_usr_conf, logger)
    if not valid:
        logger.error("check_usr_conf return False, please check")
        return

    _try_load_latest_model(agent, logger)

    for epoch in range(epoch_num):
        usr_conf = base_usr_conf
        stage_name = "train-13"
        epoch_total_rew = 0
        data_length = 0
        last_metrics_log_time = _maybe_log_training_metrics(logger, last_metrics_log_time)
        for g_data in run_episodes(
            episode_num_every_epoch,
            env,
            agent,
            g_data_truncat,
            usr_conf,
            logger,
            monitor,
            epoch,
            stage_name,
        ):
            data_length += len(g_data)
            total_rew = sum([i.rew for i in g_data])
            epoch_total_rew += total_rew
            agent.learn(g_data)
            g_data.clear()

        avg_step_reward = 0
        if data_length:
            avg_step_reward = f"{(epoch_total_rew / data_length):.2f}"

        now = time.time()
        if now - last_save_model_time >= 120:
            agent.save_model()
            last_save_model_time = now

        logger.info(f"Avg Step Reward: {avg_step_reward}, Epoch: {epoch}, Stage: {stage_name}, Data Length: {data_length}")


def run_episodes(n_episode, env, agent, g_data_truncat, usr_conf, logger, monitor, epoch, stage_name):
    for episode in range(n_episode):
        collector = list()
        preprocessor = Preprocessor()

        obs, state_env_info = env.reset(usr_conf=usr_conf)
        if obs is None:
            continue

        obs_data, remain_info = agent.observation_process(obs, preprocessor, state_env_info)

        done = False
        step = 0
        bump_cnt = 0
        diy_2 = 0
        diy_3 = 0
        diy_4 = 0
        diy_5 = 0
        episode_total_reward = 0.0

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

            if _obs is None:
                reward = 0
                is_bump = 0
                reward_end_dist = 0
                reward_exploration = 0
                reward_treasure_dist = 0
                reward_treasure = 0
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
                bump_cnt += is_bump

            if truncated:
                collected, total = _get_collected_treasure_stats(_obs)
                collected_desc = "unknown" if collected is None or total is None else f"{collected}/{total}"
                logger.info(
                    f"Ep{epoch} {stage_name} TIMEOUT | Steps:{step} | "
                    f"T:{collected_desc} | Bumps:{bump_cnt} | R:{episode_total_reward:.2f}"
                )
            elif terminated:
                collected, total = _get_collected_treasure_stats(_obs)
                collected_desc = "unknown" if collected is None or total is None else f"{collected}/{total}"
                logger.info(
                    f"Ep{epoch} {stage_name} WIN | Steps:{step} | "
                    f"T:{collected_desc} | Bumps:{bump_cnt} | R:{episode_total_reward:.2f}"
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
