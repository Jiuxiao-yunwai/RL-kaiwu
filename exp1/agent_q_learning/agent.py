#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)
from kaiwu_agent.utils.common_func import create_cls, attached
from kaiwu_agent.agent.base_agent import BaseAgent
from agent_q_learning.conf.conf import Config
from agent_q_learning.algorithm.algorithm import Algorithm


ObsData = create_cls("ObsData", feature=None)
ActData = create_cls("ActData", act=None)


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.logger = logger

        # Initialize parameters
        # 参数初始化
        self.state_size = Config.STATE_SIZE
        self.action_size = Config.ACTION_SIZE
        self.learning_rate = Config.LEARNING_RATE
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON
        self.episodes = Config.EPISODES
        self.algorithm = Algorithm(self.gamma, self.learning_rate, self.state_size, self.action_size)

        super().__init__(agent_type, device, logger, monitor)

    @predict_wrapper
    def predict(self, list_obs_data):
        """
        The input is list_obs_data, and the output is list_act_data.
        """
        """
        输入是 list_obs_data, 输出是 list_act_data
        """
        state = list_obs_data[0].feature
        act = self._epsilon_greedy(state=state, epsilon=self.epsilon)

        return [ActData(act=act)]

    @exploit_wrapper
    def exploit(self, list_obs_data):
        state = list_obs_data[0].feature
        # Old code (kept for comparison):
        # act = np.argmax(self.algorithm.Q[state, :])
        # New code: random tie-break among best actions to avoid argmax first-index bias.
        act = self._greedy_action(state)

        return [ActData(act=act)]

    def _greedy_action(self, state):
        # New helper: choose uniformly from all actions that share the max Q value.
        q_row = self.algorithm.Q[state, :]
        best_actions = np.flatnonzero(q_row == np.max(q_row))
        return int(np.random.choice(best_actions))

    def _epsilon_greedy(self, state, epsilon=0.1):
        """
        Epsilon-greedy algorithm for action selection
        """
        """
        ε-贪心算法用于动作选择
        """
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, self.action_size)

        # Exploitation
        # 探索
        else:
            """
            Break ties randomly
            If all actions are the same for this state we choose a random one
            (otherwise `np.argmax()` would always take the first one)
            """
            """
            随机打破平局,在某些情况下，当有多个动作或策略具有相同的评估值或优先级时，需要进行决策。
            为了避免总是选择第一个动作或策略，可以使用随机选择的方法来打破平局。以增加多样性和随机性
            """
            # Old code (kept for comparison):
            # if np.all(self.algorithm.Q[state, :]) == self.algorithm.Q[state, 0]:
            #     action = np.random.randint(0, self.action_size)
            # else:
            #     action = np.argmax(self.algorithm.Q[state, :])
            # New code: directly use stable greedy helper with random tie-break.
            action = self._greedy_action(state)

        return action

    @learn_wrapper
    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def _extract_treasure_status(self, raw_obs, game_info):
        # New helper: robustly extract treasure status from game_info first,
        # then fallback to raw_obs when game_info misses this field.
        treasure_status = getattr(game_info, "treasure_status", None)
        if treasure_status is None:
            feature = getattr(raw_obs, "feature", raw_obs)
            feature = np.asarray(feature).reshape(-1)
            if feature.size >= 10:
                treasure_status = feature[-10:]
            else:
                treasure_status = np.zeros(10, dtype=np.int32)

        treasure_status = np.asarray(treasure_status).reshape(-1)
        if treasure_status.size < 10:
            padded = np.zeros(10, dtype=np.int32)
            padded[: treasure_status.size] = treasure_status
            treasure_status = padded

        treasure_status = [0 if int(x) == 2 else int(x) for x in treasure_status[:10]]
        return treasure_status

    def observation_process(self, raw_obs, game_info):
        # Old code (kept for comparison, short form):
        # 1) build full 250-dim feature by concatenating state/one-hot/dist/local_view/memory/treasure_status
        # 2) pos = int(raw_obs[0]); treasure_status = raw_obs[-10:]
        # 3) state = 1024 * pos + sum([treasure_status[i] * (2**i) for i in range(10)])
        # New code: directly build state from reliable env fields (pos_x, pos_z, treasure_status),
        # reducing coupling to raw feature layout and making observation parsing more robust.
        pos_x = int(np.clip(getattr(game_info, "pos_x", 0), 0, 63))
        pos_z = int(np.clip(getattr(game_info, "pos_z", 0), 0, 63))
        pos = pos_x * 64 + pos_z

        # Q-learning state uses position + treasure mask for better Markov property.
        treasure_status = self._extract_treasure_status(raw_obs, game_info)
        treasure_mask = sum([treasure_status[i] * (2**i) for i in range(10)])
        state = 1024 * pos + treasure_mask

        return ObsData(feature=int(state))

    def action_process(self, act_data):
        return act_data.act

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        np.save(model_file_path, self.algorithm.Q)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.npy"
        try:
            self.algorithm.Q = np.load(model_file_path)
            self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            self.logger.info(f"File {model_file_path} not found")
            exit(1)
