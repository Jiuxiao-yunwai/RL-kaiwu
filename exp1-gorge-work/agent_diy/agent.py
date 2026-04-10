#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import numpy as np
import torch
from kaiwu_agent.agent.base_agent import BaseAgent
from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)
from kaiwu_agent.utils.common_func import create_cls, attached
from agent_diy.conf.conf import Config
from agent_diy.algorithm.algorithm import Algorithm

ObsData = create_cls("ObsData", feature=None)
ActData = create_cls("ActData", act=None)


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.logger = logger

        # 参数初始化
        self.state_dim = Config.STATE_DIM
        self.action_size = Config.ACTION_SIZE
        self.learning_rate = Config.LEARNING_RATE
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        self.episodes = Config.EPISODES

        self.algorithm = Algorithm(
            state_dim=self.state_dim,
            action_size=self.action_size,
            gamma=self.gamma,
            lr=self.learning_rate,
            batch_size=Config.BATCH_SIZE,
            buffer_capacity=Config.BUFFER_CAPACITY,
            target_update_freq=Config.TARGET_UPDATE_FREQ,
        )

        super().__init__(agent_type, device, logger, monitor)

    def _to_state_vector(self, feature):
        vec = np.asarray(feature, dtype=np.float32).reshape(-1)

        # 维度兜底：若输入与网络期望维度不一致，做截断/零填充，避免线性层shape报错。
        if vec.shape[0] > self.state_dim:
            vec = vec[: self.state_dim]
        elif vec.shape[0] < self.state_dim:
            padded = np.zeros((self.state_dim,), dtype=np.float32)
            padded[: vec.shape[0]] = vec
            vec = padded

        return vec

    def _greedy_action(self, state_vec):
        state_tensor = torch.as_tensor(state_vec, dtype=torch.float32, device=self.algorithm.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.algorithm.eval_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def _epsilon_greedy(self, state_vec, epsilon):
        if np.random.rand() < epsilon:
            return int(np.random.randint(0, self.action_size))
        return self._greedy_action(state_vec)

    @predict_wrapper
    def predict(self, list_obs_data):
        state_vec = self._to_state_vector(list_obs_data[0].feature)
        act = self._epsilon_greedy(state_vec=state_vec, epsilon=self.epsilon)
        return [ActData(act=act)]

    @exploit_wrapper
    def exploit(self, list_obs_data):
        state_vec = self._to_state_vector(list_obs_data[0].feature)
        act = self._greedy_action(state_vec)
        return [ActData(act=act)]

    @learn_wrapper
    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def observation_process(self, raw_obs, game_info):
        # 与内置算法保持一致：将基础距离特征 + 位置特征 + 局部视野 + 宝箱状态拼接成连续向量。
        pos = [game_info.pos_x, game_info.pos_z]

        # 特征1: 一维位置状态
        state = [int(pos[0] * 64 + pos[1])]

        # 特征2: 位置one-hot
        pos_row = [0] * 64
        pos_row[pos[0]] = 1
        pos_col = [0] * 64
        pos_col[pos[1]] = 1

        # 特征3/4: 终点距离 + 宝箱距离（环境原始输入）
        end_treasure_dists = raw_obs

        # 特征5: 局部视野图特征
        local_view = [game_info.local_view[i : i + 5] for i in range(0, len(game_info.local_view), 5)]
        obstacle_map, treasure_map, end_map = [], [], []
        for sub_list in local_view:
            obstacle_map.append([1 if i == 0 else 0 for i in sub_list])
            treasure_map.append([1 if i == 4 else 0 for i in sub_list])
            end_map.append([1 if i == 3 else 0 for i in sub_list])

        # 特征6: 视野图展平
        obstacle_flat, treasure_flat, end_flat = [], [], []
        for i in obstacle_map:
            obstacle_flat.extend(i)
        for i in treasure_map:
            treasure_flat.extend(i)
        for i in end_map:
            end_flat.extend(i)

        # 特征7: 局部记忆图
        memory_flat = []
        for i in range(game_info.view * 2 + 1):
            idx_start = (pos[0] - game_info.view + i) * 64 + (pos[1] - game_info.view)
            memory_flat.extend(game_info.location_memory[idx_start : (idx_start + game_info.view * 2 + 1)])

        # 已采集的宝箱在环境中可能标记为2，这里统一映射为0（不可再收集）
        tmp_treasure_status = [x if x != 2 else 0 for x in game_info.treasure_status]

        full_obs = np.concatenate(
            [
                state,
                pos_row,
                pos_col,
                end_treasure_dists,
                obstacle_flat,
                treasure_flat,
                end_flat,
                memory_flat,
                tmp_treasure_status,
            ]
        )

        return ObsData(feature=self._to_state_vector(full_obs))

    def action_process(self, act_data):
        return int(act_data.act)

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pth"
        state_dict = {
            "eval_net": self.algorithm.eval_net.state_dict(),
            "target_net": self.algorithm.target_net.state_dict(),
            "optimizer": self.algorithm.optimizer.state_dict(),
            "learn_step": self.algorithm.learn_step,
        }
        torch.save(state_dict, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pth"
        if not os.path.exists(model_file_path):
            self.logger.info(f"File {model_file_path} not found")
            exit(1)

        checkpoint = torch.load(model_file_path, map_location=self.algorithm.device)
        self.algorithm.eval_net.load_state_dict(checkpoint["eval_net"])

        if "target_net" in checkpoint:
            self.algorithm.target_net.load_state_dict(checkpoint["target_net"])
        else:
            self.algorithm.target_net.load_state_dict(checkpoint["eval_net"])

        if "optimizer" in checkpoint:
            self.algorithm.optimizer.load_state_dict(checkpoint["optimizer"])

        self.algorithm.learn_step = int(checkpoint.get("learn_step", 0))
        self.logger.info(f"load model {model_file_path} successfully")
