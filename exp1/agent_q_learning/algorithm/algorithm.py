#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np


class Algorithm:
    def __init__(self, gamma, learning_rate, state_size, action_size):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.state_size = state_size
        self.action_size = action_size

        # Reset the Q-table
        # 重置Q表
        # Old code (kept for comparison):
        # self.Q = np.ones([self.state_size, self.action_size])
        # New code: use float32 to reduce memory footprint for large state tables.
        self.Q = np.ones([self.state_size, self.action_size], dtype=np.float32)

    def learn(self, list_sample_data):
        """
        Update the Q-table with the given game data:
            - list_sample: each sampple is [state, action, reward, new_state]
        Using the following formula to update q value:
            - Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        """
        """
        使用给定的数据更新Q表格:
        list_sample:每个样本是[state, action, reward, new_state]
        使用以下公式更新Q值:
        Q(s,a) := Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        其中：
        Q(s,a) 表示状态s下采取动作a的Q值
        lr 是学习率(learning rate), 用于控制每次更新的幅度
        R(s,a) 是在状态s下采取动作a所获得的奖励
        gamma 是折扣因子(discount factor), 用于平衡当前奖励和未来奖励的重要性
        max Q(s',a') 表示在新状态s'下采取所有可能动作a'的最大Q值
        """
        # Old code (kept for comparison):
        # sample = list_sample_data[0]
        # state, action, reward, next_state = (
        #     sample.state,
        #     sample.action,
        #     sample.reward,
        #     sample.next_state,
        # )
        # delta = reward + self.gamma * np.max(self.Q[next_state, :]) - self.Q[state, action]
        # self.Q[state, action] += self.learning_rate * delta

        # New code: support batched samples and terminal-state update (no bootstrap when done=True).
        if not list_sample_data:
            return

        for sample in list_sample_data:
            state, action, reward, next_state = (
                int(sample.state),
                int(sample.action),
                float(sample.reward),
                int(sample.next_state),
            )
            done = bool(getattr(sample, "done", False))

            if done:
                target = reward
            else:
                target = reward + self.gamma * float(np.max(self.Q[next_state, :]))

            delta = target - float(self.Q[state, action])
            self.Q[state, action] += self.learning_rate * delta

        return
