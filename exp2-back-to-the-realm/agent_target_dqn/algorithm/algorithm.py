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


class Algorithm:
    def __init__(self, device, monitor):
        self.act_shape = Config.DIM_OF_ACTION_DIRECTION + Config.DIM_OF_TALENT
        self.direction_space = Config.DIM_OF_ACTION_DIRECTION
        self.talent_direction = Config.DIM_OF_TALENT
        self.obs_shape = Config.DIM_OF_OBSERVATION
        self.epsilon = Config.EPSILON
        self.egp = Config.EPSILON_GREEDY_PROBABILITY
        self.target_update_freq = Config.TARGET_UPDATE_FREQ
        self.obs_split = Config.DESC_OBS_SPLIT
        self._gamma = Config.GAMMA
        self.lr = Config.START_LR
        self.epsilon_start = Config.EPSILON
        self.epsilon_min = Config.EPSILON_MIN
        self.max_grad_norm = Config.MAX_GRAD_NORM
        self.device = device
        self.model = Model(
            state_shape=self.obs_shape,
            action_shape=self.act_shape,
            softmax=False,
        )
        self.model.to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.target_model = deepcopy(self.model)
        self.train_step = 0
        self.predict_count = 0
        self.last_report_monitor_time = 0
        self.monitor = monitor

    def _legal_to_mask(self, legal_act):
        def normalize_row(row):
            if isinstance(row, torch.Tensor):
                row = row.detach().cpu().numpy()
            row = np.array(row, dtype=np.float32).reshape(-1).tolist()
            if len(row) == 2:
                return [row[0]] * self.direction_space + [row[1]] * self.talent_direction
            if len(row) == self.act_shape:
                return row
            padded = [1.0] * self.act_shape
            copy_size = min(len(row), self.act_shape)
            padded[:copy_size] = row[:copy_size]
            return padded

        if isinstance(legal_act, torch.Tensor):
            legal_rows = legal_act.detach().cpu().numpy()
            rows = [legal_rows] if legal_rows.ndim == 1 else list(legal_rows)
        else:
            try:
                legal_rows = np.array(legal_act, dtype=np.float32)
                rows = [legal_rows] if legal_rows.ndim == 1 else list(legal_rows)
            except ValueError:
                rows = legal_act

        legal = torch.tensor([normalize_row(row) for row in rows], dtype=torch.bool, device=self.device)
        invalid_rows = legal.sum(dim=1) == 0
        if invalid_rows.any():
            legal[invalid_rows, : self.direction_space] = True
        return legal

    def _current_epsilon(self):
        progress = min(1.0, self.predict_count / max(1, self.egp))
        return self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (1.0 - progress)

    def learn(self, list_sample_data):

        t_data = list_sample_data
        batch = len(t_data)

        # [b, d]
        batch_feature_vec = [frame.obs[: self.obs_split[0]] for frame in t_data]
        batch_feature_map = [frame.obs[self.obs_split[0] :] for frame in t_data]
        batch_action = torch.LongTensor(np.array([int(frame.act) for frame in t_data])).view(-1, 1).to(self.device)

        _batch_obs_legal = self._legal_to_mask([frame._obs_legal for frame in t_data])

        rew = torch.tensor(np.array([frame.rew for frame in t_data], dtype=np.float32), device=self.device)
        _batch_feature_vec = [frame._obs[: self.obs_split[0]] for frame in t_data]
        _batch_feature_map = [frame._obs[self.obs_split[0] :] for frame in t_data]
        not_done = torch.tensor(
            np.array([0 if frame.done == 1 else 1 for frame in t_data], dtype=np.float32),
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

        with torch.no_grad():
            online_q, _ = self.model(_batch_feature, state=None)
            online_q = online_q.masked_fill(~_batch_obs_legal, -1e9)
            next_action = online_q.argmax(dim=1, keepdim=True)

            self.target_model.eval()
            target_next_q, _ = self.target_model(_batch_feature, state=None)
            q_max = target_next_q.gather(1, next_action).view(-1).detach()

        target_q = rew + self._gamma * q_max * not_done

        self.optim.zero_grad()

        self.model.train()
        logits, h = self.model(batch_feature, state=None)

        loss = torch.nn.functional.smooth_l1_loss(logits.gather(1, batch_action).view(-1), target_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
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
            if data.dtype == object:
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
        legal_act = self._legal_to_mask(legal_act)
        model = self.model
        model.eval()
        # Exploration factor,
        # we want epsilon to decrease as the number of prediction steps increases, until it reaches 0.1
        # 探索因子, 我们希望epsilon随着预测步数越来越小，直到0.1为止
        self.epsilon = self.epsilon_min if exploit_flag else self._current_epsilon()

        with torch.no_grad():
            # epsilon greedy
            if not exploit_flag and np.random.rand(1) < self.epsilon:
                random_action = np.random.rand(batch, self.act_shape)
                random_action = torch.tensor(random_action, dtype=torch.float32).to(self.device)
                random_action = random_action.masked_fill(~legal_act, -1e9)
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
