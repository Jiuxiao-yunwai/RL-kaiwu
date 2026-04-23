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
import numpy as np
import torch
from copy import deepcopy
from agent_target_dqn.model.model import Model
from agent_target_dqn.conf.conf import Config
from agent_target_dqn.feature.definition import ActData


class PrioritizedReplayBuffer:
    """
    A minimal PER scaffold for future integration with off-policy replay.
    用于后续离策略训练的PER基础框架。
    """

    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment=1e-6, eps=1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = eps
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, transition, priority=None):
        max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0
        if priority is None:
            priority = max_prio

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = float(priority)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return [], [], []

        prios = self.priorities[: len(self.buffer)]
        probs = np.power(prios + self.eps, self.alpha)
        probs = probs / probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = np.power(len(self.buffer) * probs[indices], -self.beta)
        weights = weights / weights.max()
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(prio)


class Algorithm:
    def __init__(self, device, monitor):
        self.act_shape = Config.DIM_OF_ACTION_DIRECTION + Config.DIM_OF_TALENT
        self.direction_space = Config.DIM_OF_ACTION_DIRECTION
        self.talent_direction = Config.DIM_OF_TALENT
        self.obs_shape = Config.DIM_OF_OBSERVATION
        self.epsilon_start = Config.EPSILON
        self.min_epsilon = Config.MIN_EPSILON
        self.epsilon = self.epsilon_start
        self.egp = Config.EPSILON_GREEDY_PROBABILITY
        self.target_update_freq = Config.TARGET_UPDATE_FREQ
        self.obs_split = Config.DESC_OBS_SPLIT
        self._gamma = Config.GAMMA
        self.lr = Config.START_LR
        self.use_double_dqn = Config.USE_DOUBLE_DQN
        self.use_per = Config.USE_PER
        self.device = device
        self.model = Model(
            state_shape=self.obs_shape,
            action_shape=self.act_shape,
            softmax=False,
            use_dueling=Config.USE_DUELING,
        )
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.target_model = deepcopy(self.model)
        self.target_model.to(self.device)
        self.train_step = 0
        self.predict_count = 0
        self.last_report_monitor_time = 0
        self.monitor = monitor
        self.per_buffer = PrioritizedReplayBuffer() if self.use_per else None

    def learn(self, list_sample_data):

        t_data = list_sample_data
        batch = len(t_data)

        # [b, d]
        batch_feature_vec = [frame.obs[: self.obs_split[0]] for frame in t_data]
        batch_feature_map = [frame.obs[self.obs_split[0] :] for frame in t_data]
        batch_action = torch.LongTensor(np.array([int(frame.act) for frame in t_data])).view(-1, 1).to(self.device)

        _batch_obs_legal = torch.stack([frame._obs_legal for frame in t_data])
        _batch_obs_legal = (
            torch.cat(
                (
                    _batch_obs_legal[:, 0].unsqueeze(1).expand(batch, self.direction_space),
                    _batch_obs_legal[:, 1].unsqueeze(1).expand(batch, self.talent_direction),
                ),
                1,
            )
            .bool()
            .to(self.device)
        )

        rew = torch.tensor(np.array([frame.rew for frame in t_data]), device=self.device)
        _batch_feature_vec = [frame._obs[: self.obs_split[0]] for frame in t_data]
        _batch_feature_map = [frame._obs[self.obs_split[0] :] for frame in t_data]
        not_done = torch.tensor(
            np.array([0 if frame.done == 1 else 1 for frame in t_data]),
            device=self.device,
        )

        batch_feature = [
            self.__convert_to_tensor(batch_feature_vec),
            self.__convert_to_tensor(batch_feature_map).view(batch, *self.obs_split[1]),
        ]
        _batch_feature = [
            self.__convert_to_tensor(_batch_feature_vec),
            self.__convert_to_tensor(_batch_feature_map).view(batch, *self.obs_split[1]),
        ]

        target_model = getattr(self, "target_model")
        target_model.eval()
        online_model = getattr(self, "model")
        online_model.eval()
        with torch.no_grad():
            next_target_q, _ = target_model(_batch_feature, state=None)
            next_target_q = next_target_q.masked_fill(~_batch_obs_legal, -1e9)

            if self.use_double_dqn:
                next_online_q, _ = online_model(_batch_feature, state=None)
                next_online_q = next_online_q.masked_fill(~_batch_obs_legal, -1e9)
                next_action = next_online_q.argmax(dim=1, keepdim=True)
                q_max = next_target_q.gather(1, next_action).view(-1).detach()
            else:
                q_max = next_target_q.max(dim=1).values.detach()

        target_q = rew + self._gamma * q_max * not_done

        self.optim.zero_grad()

        model = getattr(self, "model")
        model.train()
        logits, h = model(batch_feature, state=None)

        loss = torch.square(target_q - logits.gather(1, batch_action).view(-1)).mean()
        loss.backward()
        self.optim.step()

        self.train_step += 1

        # Update the target network
        # 更新target网络
        if self.train_step % self.target_update_freq == 0:
            self.update_target_q()

        value_loss = loss.detach().item()
        q_value = target_q.mean().detach().item()
        reward = rew.mean().detach().item()

        # Periodically report monitoring
        # 按照间隔上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "value_loss": value_loss,
                "q_value": q_value,
                "reward": reward,
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})

            self.last_report_monitor_time = now

    def __convert_to_tensor(self, data):
        if isinstance(data, list):
            data = [np.array(item, dtype=np.float32) for item in data]
        elif isinstance(data, np.ndarray):
            if data.dtype == np.object:
                data = data.astype(np.float32)
            else:
                data = data.astype(np.float32)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        tensor = torch.stack([torch.tensor(item) for item in data]).to(self.device)
        return tensor

    def predict_detail(self, list_obs_data, exploit_flag=False):
        batch = len(list_obs_data)
        feature_vec = [obs_data.feature[: self.obs_split[0]] for obs_data in list_obs_data]
        feature_map = [obs_data.feature[self.obs_split[0] :] for obs_data in list_obs_data]
        legal_act = [obs_data.legal_act for obs_data in list_obs_data]
        legal_act = torch.tensor(np.array(legal_act))
        legal_act = (
            torch.cat(
                (
                    legal_act[:, 0].unsqueeze(1).expand(batch, self.direction_space),
                    legal_act[:, 1].unsqueeze(1).expand(batch, self.talent_direction),
                ),
                1,
            )
            .bool()
            .to(self.device)
        )

        # Additional directional action mask from local obstacle perception.
        # 来自局部障碍感知的额外方向掩码。
        move_masks = [
            obs_data.move_mask if getattr(obs_data, "move_mask", None) is not None else [1] * self.direction_space
            for obs_data in list_obs_data
        ]
        move_masks = torch.tensor(np.array(move_masks), dtype=torch.bool).to(self.device)
        legal_act[:, : self.direction_space] = legal_act[:, : self.direction_space] & move_masks

        # Safety fallback: if a sample has no legal action after masking, allow all movement directions.
        # 兜底：若掩码后没有合法动作，放开移动方向避免动作无定义。
        row_has_legal = legal_act.any(dim=1)
        for i in range(batch):
            if not bool(row_has_legal[i]):
                legal_act[i, : self.direction_space] = True

        model = self.model
        model.eval()

        # Stable linear epsilon schedule with a hard minimum.
        # 线性epsilon调度并设置最小值下限。
        decay_ratio = min(1.0, self.predict_count / max(1, self.egp))
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon_start - (self.epsilon_start - self.min_epsilon) * decay_ratio,
        )

        with torch.no_grad():
            # epsilon greedy
            if not exploit_flag and np.random.rand() < self.epsilon:
                random_action = np.random.rand(batch, self.act_shape)
                random_action = torch.tensor(random_action, dtype=torch.float32).to(self.device)
                random_action = random_action.masked_fill(~legal_act, -1.0)
                act = random_action.argmax(dim=1).cpu().view(-1, 1).tolist()
            else:
                feature = [
                    self.__convert_to_tensor(feature_vec),
                    self.__convert_to_tensor(feature_map).view(batch, *self.obs_split[1]),
                ]
                logits, _ = model(feature, state=None)
                logits = logits.masked_fill(~legal_act, -1e9)
                act = logits.argmax(dim=1).cpu().view(-1, 1).tolist()

        format_action = [[instance[0] % self.direction_space, instance[0] // self.direction_space] for instance in act]
        self.predict_count += 1
        return [ActData(move_dir=i[0], use_talent=i[1]) for i in format_action]

    def update_target_q(self):
        self.target_model.load_state_dict(self.model.state_dict())
