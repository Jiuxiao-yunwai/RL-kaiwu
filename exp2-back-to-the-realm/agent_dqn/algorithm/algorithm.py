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
from agent_dqn.conf.conf import Config
from agent_dqn.model.model import Model
from agent_dqn.feature.definition import ActData


class Algorithm:
    def __init__(self, device, monitor):
        self.device = device
        self.lr = Config.START_LR
        self.epsilon = Config.EPSILON
        self.egp = Config.EPSILON_GREEDY_PROBABILITY
        self.epsilon_start = Config.EPSILON
        self.epsilon_min = Config.EPSILON_MIN
        self._gamma = Config.GAMMA
        self.max_grad_norm = Config.MAX_GRAD_NORM
        self.obs_split = Config.DESC_OBS_SPLIT
        self.obs_shape = Config.DIM_OF_OBSERVATION
        self.talent_direction = Config.DIM_OF_TALENT
        self.direction_space = Config.DIM_OF_ACTION_DIRECTION
        self.act_shape = Config.DIM_OF_ACTION_DIRECTION + Config.DIM_OF_TALENT

        self.train_step = 0
        self.predict_count = 0
        self.model = Model(
            state_shape=self.obs_shape,
            action_shape=self.act_shape,
            softmax=False,
        )
        self.model.to(self.device)
        self.device = device
        self.monitor = monitor
        self.last_report_monitor_time = 0
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _legal_to_mask(self, legal_act):
        def normalize_row(row):
            if isinstance(row, torch.Tensor):
                row = row.detach().cpu().numpy()
            row = np.array(row, dtype=np.float32).reshape(-1)
            if row.size == 2:
                return [bool(row[0])] * self.direction_space + [bool(row[1])] * self.talent_direction
            if row.size == self.act_shape:
                return row.astype(bool).tolist()

            mask = np.ones(self.act_shape, dtype=bool)
            copy_size = min(row.size, self.act_shape)
            mask[:copy_size] = row[:copy_size].astype(bool)
            return mask.tolist()

        rows = legal_act
        if isinstance(legal_act, torch.Tensor):
            rows = legal_act.detach().cpu().numpy()
        rows = np.array(rows, dtype=object)
        if rows.ndim == 1 and rows.size in (2, self.act_shape):
            rows = [rows]

        mask = torch.tensor([normalize_row(row) for row in rows], dtype=torch.bool, device=self.device)
        invalid_rows = mask.sum(dim=1) == 0
        if invalid_rows.any():
            mask[invalid_rows, : self.direction_space] = True
        return mask

    def _current_epsilon(self):
        progress = min(1.0, self.predict_count / max(1, self.egp))
        return self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (1.0 - progress)

    def learn(self, list_sample_data):
        t_data = list_sample_data
        batch = len(t_data)
        if batch == 0:
            return

        obs_size = self.obs_split[0] + int(np.prod(self.obs_split[1]))
        batch_obs = [self.__to_1d_float_array(frame.obs, obs_size, "obs") for frame in t_data]
        _batch_obs = [self.__to_1d_float_array(frame._obs, obs_size, "_obs") for frame in t_data]
        batch_feature_vec = [obs[: self.obs_split[0]] for obs in batch_obs]
        batch_feature_map = [obs[self.obs_split[0] :] for obs in batch_obs]
        batch_action = torch.LongTensor(
            np.array([int(self.__to_1d_float_array(frame.act, name="act")[0]) for frame in t_data])
        ).view(-1, 1).to(self.device)

        _batch_obs_legal = self._legal_to_mask([frame._obs_legal for frame in t_data])

        rew = torch.tensor(
            np.array([self.__to_1d_float_array(frame.rew, name="rew")[0] for frame in t_data], dtype=np.float32),
            device=self.device,
        )
        _batch_feature_vec = [obs[: self.obs_split[0]] for obs in _batch_obs]
        _batch_feature_map = [obs[self.obs_split[0] :] for obs in _batch_obs]
        not_done = torch.tensor(
            np.array(
                [0 if self.__to_1d_float_array(frame.done, name="done")[0] == 1 else 1 for frame in t_data],
                dtype=np.float32,
            ),
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

        model = getattr(self, "model")
        model.eval()
        with torch.no_grad():
            q, h = model(_batch_feature, state=None)
            q = q.masked_fill(~_batch_obs_legal, -1e9)
            q_max = q.max(dim=1).values.detach()

        target_q = rew + self._gamma * q_max * not_done

        self.optim.zero_grad()

        model = getattr(self, "model")
        model.train()
        logits, h = model(batch_feature, state=None)

        loss = torch.square(target_q - logits.gather(1, batch_action).view(-1)).mean()
        loss.backward()

        model_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        self.optim.step()

        self.train_step += 1

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
                "diy_1": float(model_grad_norm),
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})

            self.last_report_monitor_time = now

    def __to_1d_float_array(self, data, expected_size=None, name="data"):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        arr = np.asarray(data)
        if arr.dtype == object:
            parts = []
            for item in arr.reshape(-1):
                if isinstance(item, (list, tuple, np.ndarray, torch.Tensor)):
                    parts.append(self.__to_1d_float_array(item))
                else:
                    parts.append(np.asarray([item], dtype=np.float32))
            arr = np.concatenate(parts) if parts else np.asarray([], dtype=np.float32)
        else:
            arr = arr.astype(np.float32, copy=False).reshape(-1)

        if expected_size is not None and arr.size != expected_size:
            raise ValueError(f"{name} has size {arr.size}, expected {expected_size}")
        return arr

    def __convert_to_tensor(self, data):
        if not isinstance(data, (list, tuple, np.ndarray)):
            raise TypeError(f"Unsupported data type: {type(data)}")

        data = np.stack([self.__to_1d_float_array(item) for item in data], axis=0)
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    def predict_detail(self, list_obs_data, exploit_flag=False):
        batch = len(list_obs_data)
        obs_size = self.obs_split[0] + int(np.prod(self.obs_split[1]))
        features = [self.__to_1d_float_array(obs_data.feature, obs_size, "feature") for obs_data in list_obs_data]
        feature_vec = [feature[: self.obs_split[0]] for feature in features]
        feature_map = [feature[self.obs_split[0] :] for feature in features]
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
                random_action = torch.rand(batch, self.act_shape, device=self.device)
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
