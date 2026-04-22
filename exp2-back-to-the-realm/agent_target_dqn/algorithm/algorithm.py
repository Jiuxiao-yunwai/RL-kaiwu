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
        self.epsilon_min = Config.EPSILON_MIN
        self.egp = Config.EPSILON_GREEDY_PROBABILITY
        self.target_update_freq = Config.TARGET_UPDATE_FREQ
        self.obs_split = Config.DESC_OBS_SPLIT
        self._gamma = Config.GAMMA
        self.lr = Config.START_LR
        self.grad_clip = Config.GRAD_NORM_CLIP
        self.device = device
        self.model = Model(
            state_shape=self.obs_shape,
            action_shape=self.act_shape,
            softmax=False,
        )
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.target_model = deepcopy(self.model)
        self.train_step = 0
        self.predict_count = 0
        self.last_report_monitor_time = 0
        self.monitor = monitor

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

        map_dim = int(np.prod(self.obs_split[1]))
        batch_feature = [
            self.__convert_to_tensor(batch_feature_vec, expected_dim=self.obs_split[0]),
            self.__convert_to_tensor(batch_feature_map, expected_dim=map_dim).view(batch, *self.obs_split[1]),
        ]
        _batch_feature = [
            self.__convert_to_tensor(_batch_feature_vec, expected_dim=self.obs_split[0]),
            self.__convert_to_tensor(_batch_feature_map, expected_dim=map_dim).view(batch, *self.obs_split[1]),
        ]

        online_model = getattr(self, "model")
        target_model = getattr(self, "target_model")
        online_model.eval()
        target_model.eval()
        with torch.no_grad():
            # Double-DQN style target:
            # 1) online network selects next action
            # 2) target network evaluates that action
            # Double-DQN目标:
            # 1) 在线网络选下一步动作
            # 2) 目标网络评估该动作
            next_q_online, _ = online_model(_batch_feature, state=None)
            next_q_online = next_q_online.masked_fill(~_batch_obs_legal, -1e9)
            next_action = next_q_online.argmax(dim=1, keepdim=True)

            next_q_target, _ = target_model(_batch_feature, state=None)
            next_q = next_q_target.gather(1, next_action).view(-1).detach()

        target_q = rew + self._gamma * next_q * not_done

        self.optim.zero_grad()

        model = getattr(self, "model")
        model.train()
        logits, h = model(batch_feature, state=None)

        pred_q = logits.gather(1, batch_action).view(-1)
        loss = torch.nn.functional.smooth_l1_loss(pred_q, target_q)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
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
                "diy_1": float(grad_norm),
                "diy_2": float(self.epsilon),
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})

            self.last_report_monitor_time = now

    def __convert_to_tensor(self, data, expected_dim=None):
        def _to_fixed_1d(item):
            arr = np.asarray(item, dtype=np.float32).reshape(-1)
            if expected_dim is None:
                return arr
            if arr.size == expected_dim:
                return arr
            if arr.size > expected_dim:
                return arr[:expected_dim]
            out = np.zeros(expected_dim, dtype=np.float32)
            out[: arr.size] = arr
            return out

        if isinstance(data, np.ndarray) and data.dtype != object and expected_dim is None:
            arr = data.astype(np.float32, copy=False)
            return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

        if not isinstance(data, (list, tuple, np.ndarray)):
            raise TypeError(f"Unsupported data type: {type(data)}")

        rows = [_to_fixed_1d(item) for item in data]
        if expected_dim is None:
            # Fallback for ragged batches: align to the longest sample.
            # 对不等长样本做兜底：按最长样本对齐。
            max_dim = max((r.size for r in rows), default=0)
            fixed_rows = []
            for r in rows:
                if r.size == max_dim:
                    fixed_rows.append(r)
                else:
                    tmp = np.zeros(max_dim, dtype=np.float32)
                    tmp[: r.size] = r
                    fixed_rows.append(tmp)
            rows = fixed_rows

        arr = np.stack(rows, axis=0).astype(np.float32, copy=False)
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

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
        model = self.model
        model.eval()
        # Exploration factor with linear decay.
        # 线性衰减探索率。
        decay_ratio = min(float(self.predict_count) / float(max(self.egp, 1)), 1.0)
        self.epsilon = self.epsilon_min + (Config.EPSILON - self.epsilon_min) * (1.0 - decay_ratio)

        with torch.no_grad():
            # epsilon greedy
            if not exploit_flag and np.random.rand(1) < self.epsilon:
                # Uniformly sample one legal action to improve exploration quality.
                # 从合法动作中均匀采样，提升探索质量。
                legal_float = legal_act.float()
                legal_count = legal_float.sum(dim=1, keepdim=True).clamp(min=1.0)
                action_prob = legal_float / legal_count
                act = torch.multinomial(action_prob, num_samples=1).cpu().tolist()
            else:
                feature = [
                    self.__convert_to_tensor(feature_vec, expected_dim=self.obs_split[0]),
                    self.__convert_to_tensor(feature_map, expected_dim=int(np.prod(self.obs_split[1]))).view(
                        batch, *self.obs_split[1]
                    ),
                ]
                logits, _ = model(feature, state=None)
                logits = logits.masked_fill(~legal_act, float(torch.min(logits)))
                act = logits.argmax(dim=1).cpu().view(-1, 1).tolist()

        format_action = [[instance[0] % self.direction_space, instance[0] // self.direction_space] for instance in act]
        self.predict_count += 1
        return [ActData(move_dir=i[0], use_talent=i[1]) for i in format_action]

    def update_target_q(self):
        self.target_model.load_state_dict(self.model.state_dict())
