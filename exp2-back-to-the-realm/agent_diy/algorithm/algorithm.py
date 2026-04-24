#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from copy import deepcopy
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData
from agent_diy.model.model import Model


class Algorithm:
    def __init__(self, device, monitor):
        self.act_shape = Config.DIM_OF_ACTION_DIRECTION + Config.DIM_OF_TALENT
        self.direction_space = Config.DIM_OF_ACTION_DIRECTION
        self.talent_direction = Config.DIM_OF_TALENT
        self.obs_shape = Config.DIM_OF_OBSERVATION
        self.epsilon_start = Config.EPSILON_START
        self.epsilon_end = Config.EPSILON_END
        self.epsilon_decay_steps = Config.EPSILON_DECAY_STEPS
        self.epsilon = self.epsilon_start
        self.target_update_freq = Config.TARGET_UPDATE_FREQ
        self.target_soft_tau = Config.TARGET_SOFT_TAU
        self.obs_split = Config.DESC_OBS_SPLIT
        self.gamma = Config.GAMMA
        self.n_step = Config.N_STEP
        self.grad_clip_norm = Config.GRAD_CLIP_NORM
        self.lr = Config.START_LR
        self.epsilon_warmup_steps = Config.EPSILON_WARMUP_STEPS
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

        rew = torch.tensor(np.array([frame.rew for frame in t_data]), dtype=torch.float32, device=self.device)
        _batch_feature_vec = [frame._obs[: self.obs_split[0]] for frame in t_data]
        _batch_feature_map = [frame._obs[self.obs_split[0] :] for frame in t_data]
        done_flag = torch.tensor(np.array([1 if frame.done == 1 else 0 for frame in t_data]), dtype=torch.float32, device=self.device)

        batch_feature = [
            self.__convert_to_tensor(batch_feature_vec),
            self.__convert_to_tensor(batch_feature_map).view(batch, *self.obs_split[1]),
        ]
        _batch_feature = [
            self.__convert_to_tensor(_batch_feature_vec),
            self.__convert_to_tensor(_batch_feature_map).view(batch, *self.obs_split[1]),
        ]

        online_model = self.model
        target_model = self.target_model
        online_model.eval()
        target_model.eval()
        with torch.no_grad():
            q_online, _ = online_model(_batch_feature, state=None)
            q_online = q_online.masked_fill(~_batch_obs_legal, float(torch.min(q_online)))
            next_action = q_online.argmax(dim=1, keepdim=True)

            q_target, _ = target_model(_batch_feature, state=None)
            q_target = q_target.masked_fill(~_batch_obs_legal, float(torch.min(q_target)))
            q_max = q_target.gather(1, next_action).view(-1)

        # Build n-step returns from sequential fragments in current mini-batch.
        target_q = torch.zeros_like(rew)
        for i in range(batch):
            discount = 1.0
            returns_i = 0.0
            terminal_reached = False
            last_index = i

            for k in range(self.n_step):
                idx = i + k
                if idx >= batch:
                    break
                returns_i += discount * rew[idx]
                last_index = idx
                if done_flag[idx] > 0.5:
                    terminal_reached = True
                    break
                discount *= self.gamma

            if not terminal_reached:
                returns_i += discount * q_max[last_index]
            target_q[i] = returns_i

        self.optim.zero_grad()
        self.model.train()
        logits, _ = self.model(batch_feature, state=None)
        pred_q = logits.gather(1, batch_action).view(-1)

        td_abs = (target_q - pred_q).detach().abs()
        td_weight = 0.4 + 0.6 * (td_abs / (td_abs.mean() + 1e-6))
        element_loss = F.smooth_l1_loss(pred_q, target_q, reduction="none")
        loss = (td_weight * element_loss).mean()
        loss.backward()
        model_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optim.step()

        self.train_step += 1

        self.soft_update_target_q()
        if self.train_step % self.target_update_freq == 0:
            self.update_target_q()

        value_loss = loss.detach().item()
        q_value = target_q.mean().detach().item()
        reward = rew.mean().detach().item()

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "value_loss": value_loss,
                "q_value": q_value,
                "reward": reward,
                "diy_1": model_grad_norm,
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})
            self.last_report_monitor_time = now

    def __convert_to_tensor(self, data):
        if isinstance(data, list):
            data = np.stack([np.asarray(item, dtype=np.float32) for item in data], axis=0)
        elif isinstance(data, np.ndarray):
            if data.dtype == np.object_:
                data = data.astype(np.float32)
            else:
                data = data.astype(np.float32, copy=False)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        return torch.from_numpy(data).to(self.device)

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

        self.model.eval()
        if self.predict_count < self.epsilon_warmup_steps:
            self.epsilon = self.epsilon_start
        else:
            decay_steps = max(float(self.epsilon_decay_steps - self.epsilon_warmup_steps), 1.0)
            decay_ratio = min(float(self.predict_count - self.epsilon_warmup_steps) / decay_steps, 1.0)
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_ratio

        with torch.no_grad():
            if not exploit_flag and np.random.rand(1) < self.epsilon:
                random_action = np.random.rand(batch, self.act_shape)
                random_action = torch.tensor(random_action, dtype=torch.float32).to(self.device)
                random_action = random_action.masked_fill(~legal_act, 0)
                act = random_action.argmax(dim=1).cpu().view(-1, 1).tolist()
            else:
                feature = [
                    self.__convert_to_tensor(feature_vec),
                    self.__convert_to_tensor(feature_map).view(batch, *self.obs_split[1]),
                ]
                logits, _ = self.model(feature, state=None)
                logits = logits.masked_fill(~legal_act, float(torch.min(logits)))
                act = logits.argmax(dim=1).cpu().view(-1, 1).tolist()

        format_action = [[instance[0] % self.direction_space, instance[0] // self.direction_space] for instance in act]
        self.predict_count += 1
        return [ActData(move_dir=i[0], use_talent=i[1]) for i in format_action]

    def soft_update_target_q(self):
        tau = self.target_soft_tau
        with torch.no_grad():
            for target_param, online_param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

    def update_target_q(self):
        self.target_model.load_state_dict(self.model.state_dict())
