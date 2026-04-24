#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
Optimized: Double DQN + 精简训练流程，极致训练效率
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
        self.epsilon_decay = Config.EPSILON_DECAY
        self.target_update_freq = Config.TARGET_UPDATE_FREQ
        self.obs_split = Config.DESC_OBS_SPLIT
        self._gamma = Config.GAMMA
        self.lr = Config.START_LR
        self.device = device
        self.model = Model(
            state_shape=self.obs_shape,
            action_shape=self.act_shape,
            softmax=False,
        )
        self.model.to(self.device)
        
        # Adam优化器
        self.optim = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.lr,
        )
        
        self.target_model = deepcopy(self.model)
        self.train_step = 0
        self.predict_count = 0
        self.last_report_monitor_time = 0
        self.monitor = monitor
        
        # Huber Loss (Smooth L1 Loss) - 对异常值更鲁棒
        self.loss_fn = torch.nn.SmoothL1Loss()

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
        
        # 奖励裁剪: 避免极端奖励导致Q值不稳定
        rew = rew.clamp(-10.0, 10.0)
        
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

        # =============================================
        # Double DQN: 用online网络选动作，用target网络评估Q值
        # 这样可以有效缓解Q值过高估计的问题
        # =============================================
        
        # 1. 使用online网络选择下一状态的最优动作
        self.model.eval()
        with torch.no_grad():
            q_online, _ = self.model(_batch_feature, state=None)
            q_online = q_online.masked_fill(~_batch_obs_legal, float('-inf'))
            next_actions = q_online.argmax(dim=1, keepdim=True)
        
        # 2. 使用target网络评估所选动作的Q值
        self.target_model.eval()
        with torch.no_grad():
            q_target, _ = self.target_model(_batch_feature, state=None)
            q_max = q_target.gather(1, next_actions).squeeze(1).detach()

        target_q = rew + self._gamma * q_max * not_done

        # 3. 前向传播计算当前Q值
        self.optim.zero_grad()
        self.model.train()
        logits, _ = self.model(batch_feature, state=None)
        current_q = logits.gather(1, batch_action).squeeze(1)

        # 使用Huber Loss替代MSE，对异常值更鲁棒
        loss = self.loss_fn(current_q, target_q)
        loss.backward()
        
        # 梯度裁剪 - 防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        
        self.optim.step()

        self.train_step += 1

        # 软更新target网络 (Polyak averaging)
        if self.train_step % self.target_update_freq == 0:
            self.soft_update_target(tau=0.01)

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
            if data.dtype == np.object_:
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
        model = self.model
        model.eval()
        
        # =============================================
        # 改进的epsilon-greedy策略
        # 使用指数衰减，更快收敛到利用模式
        # =============================================
        if not exploit_flag:
            self.epsilon = max(
                self.epsilon_min, 
                self.epsilon * self.epsilon_decay
            )

        with torch.no_grad():
            # epsilon greedy
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
                logits, _ = model(feature, state=None)
                logits = logits.masked_fill(~legal_act, float('-inf'))
                act = logits.argmax(dim=1).cpu().view(-1, 1).tolist()

        format_action = [[instance[0] % self.direction_space, instance[0] // self.direction_space] for instance in act]
        self.predict_count += 1
        return [ActData(move_dir=i[0], use_talent=i[1]) for i in format_action]

    def soft_update_target(self, tau=0.01):
        """
        软更新target网络 (Polyak averaging)
        θ_target = τ * θ_online + (1 - τ) * θ_target
        比硬拷贝更平滑，训练更稳定。tau=0.01比0.005更快跟进
        """
        for target_param, online_param in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )

    def update_target_q(self):
        self.target_model.load_state_dict(self.model.state_dict())
