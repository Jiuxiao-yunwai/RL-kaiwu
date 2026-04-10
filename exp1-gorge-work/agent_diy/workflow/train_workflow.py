#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from agent_diy.feature.definition import (
    sample_process,
    reward_shaping,
)
from agent_diy.conf.conf import Config
from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached
from tools.train_env_conf_validate import check_usr_conf, read_usr_conf
from tools.metrics_utils import get_training_metrics
import time
import os


@attached
def workflow(envs, agents, logger=None, monitor=None):
    """
    Users can define their own training workflows here
    用户可以在此处自行定义训练工作流
    """

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    # check_usr_conf is a tool to check whether the game configuration is correct
    # It is recommended to perform a check before calling reset.env
    # check_usr_conf会检查游戏配置是否正确，建议调用reset.env前先检查一下
    valid = check_usr_conf(usr_conf, logger)
    if not valid:
        logger.error("check_usr_conf return False, please check")
        return

    env, agent = envs[0], agents[0]
    EPISODES = Config.EPISODES

    # 监控数据初始化
    monitor_data = {
        "reward": 0,
        "diy_1": 0,
        "diy_2": 0,
        "diy_3": 0,
        "diy_4": 0,
        "diy_5": 0,
    }
    last_report_monitor_time = time.time()

    logger.info("Start Training ...")
    start_t = time.time()

    total_rew, win_cnt = 0, 0
    agent.epsilon = 1.0

    for episode in range(EPISODES):
        # 获取训练中的指标
        training_metrics = get_training_metrics()
        if training_metrics:
            logger.info(f"training_metrics is {training_metrics}")

        # 重置环境, 并获取初始状态
        obs, state = env.reset(usr_conf=usr_conf)
        if obs is None:
            continue

        # 首帧处理
        obs_data = agent.observation_process(obs, state)

        # 任务循环
        done, prev_score = False, 0
        while not done:
            # epsilon按步衰减
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

            # Agent推理动作
            act_data, model_version = agent.predict(list_obs_data=[obs_data])
            _ = model_version
            act_data = act_data[0]
            act = agent.action_process(act_data)

            # 与环境交互
            frame_no, _obs, score, terminated, truncated, state = env.step(act)
            if _obs is None:
                break

            # 特征处理
            _obs_data = agent.observation_process(_obs, state)

            # 计算reward
            reward = reward_shaping(frame_no, score, prev_score, terminated, truncated, obs, _obs)

            # 判断结束
            done = terminated or truncated
            if terminated:
                win_cnt += 1

            # 组织训练样本
            sample = Frame(
                state=obs_data.feature,
                action=act,
                reward=reward,
                next_state=_obs_data.feature,
                done=done,
            )
            sample = sample_process([sample])

            # 训练
            agent.learn(sample)

            # 更新状态
            total_rew += reward
            prev_score = score
            obs = _obs
            obs_data = _obs_data

        # 上报训练进度
        now = time.time()
        if now - last_report_monitor_time > 60:
            logger.info(f"Episode: {episode + 1}, Reward: {total_rew}")
            logger.info(f"Training Win Rate: {win_cnt / (episode + 1)}")
            monitor_data["reward"] = total_rew
            if monitor:
                monitor.put_data({os.getpid(): monitor_data})
            total_rew = 0
            last_report_monitor_time = now

        # 收敛判定
        if win_cnt / (episode + 1) > 0.9 and episode > 100:
            logger.info(f"Training Converged at Episode: {episode + 1}")
            monitor_data["reward"] = total_rew
            if monitor:
                monitor.put_data({os.getpid(): monitor_data})
            break

    end_t = time.time()
    logger.info(f"Training Time for {episode + 1} episodes: {end_t - start_t} s")
    agent.episodes = episode + 1

    # 保存模型
    agent.save_model()

    return
