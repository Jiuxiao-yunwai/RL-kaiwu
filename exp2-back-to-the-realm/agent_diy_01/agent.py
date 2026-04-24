#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
BFS Path Planning Agent with Global Map Memory (DP approach)
BFS路径规划智能体 + 全局地图记忆 (动态规划方法)

不使用神经网络，通过积累全局地图信息实现跨轮次学习。
"""

import torch
import numpy as np
from kaiwu_agent.agent.base_agent import (
    predict_wrapper, exploit_wrapper, learn_wrapper,
    save_model_wrapper, load_model_wrapper, BaseAgent,
)
from kaiwu_agent.utils.common_func import attached
from .feature.definition import ActData, ObsData
from .algorithm.algorithm import Algorithm
from .feature.preprocessor import get_grid_pos
from arena_proto.back_to_the_realm.custom_pb2 import RelativeDirection


def one_hot_encoding(grid_pos):
    """
    One-hot encoding for grid position (256-dim vector)
    此函数将网格位置特征进行one_hot_encoding处理, 返回一个长度为256的向量
    """
    one_hot_pos_x, one_hot_pos_z = np.zeros(128).tolist(), np.zeros(128).tolist()
    one_hot_pos_x[grid_pos.x], one_hot_pos_z[grid_pos.z] = 1, 1
    return one_hot_pos_x + one_hot_pos_z


def read_relative_position(rel_pos):
    """Read relative position info, return 9-dim vector"""
    direction = [0] * 8
    if rel_pos.direction != RelativeDirection.RELATIVE_DIRECTION_NONE:
        if 1 <= rel_pos.direction <= 8:
            direction[rel_pos.direction - 1] = 1
    grid_distance = 1 if rel_pos.grid_distance < 0 else rel_pos.grid_distance / (128 * 128)
    return direction + [grid_distance]


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.agent_type = agent_type
        self.logger = logger
        self.device = device
        self.algorithm = Algorithm(logger, monitor)
        self._hero_grid_pos = None
        self._remain_info = {}

    @predict_wrapper
    def predict(self, list_obs_data):
        obs_data = list_obs_data[0]
        feature = obs_data.feature
        vec_dim = 404
        ms = 51
        mt = ms * ms
        obstacle_map = feature[vec_dim : vec_dim + mt]
        end_map = feature[vec_dim + mt : vec_dim + 2*mt]
        treasure_map = feature[vec_dim + 2*mt : vec_dim + 3*mt]

        act = self.algorithm.plan_action(
            hero_pos=self._hero_grid_pos,
            remain_info=self._remain_info,
            obstacle_map=obstacle_map,
            treasure_map=treasure_map,
            end_map=end_map,
            legal_act=obs_data.legal_act,
        )
        return [act]

    @exploit_wrapper
    def exploit(self, list_obs_data):
        return self.predict(list_obs_data)

    @learn_wrapper
    def learn(self, list_sample_data):
        pass

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        """Save global map to disk for cross-session persistence"""
        if path:
            self.algorithm.gmap.save(path, id=id)
            self.logger.info(f"DP: saved global map (explored {self.algorithm.gmap.explored_ratio:.1%})")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        """Load global map from disk"""
        if path:
            loaded = self.algorithm.gmap.load(path, id=id)
            if loaded:
                self.logger.info(
                    f"DP: loaded global map (explored {self.algorithm.gmap.explored_ratio:.1%}, "
                    f"treasures: {len(self.algorithm.gmap.treasure_pos)})"
                )

    def observation_process(self, raw_obs, preprocessor, state_env_info=None):
        """
        Feature processing + global map update
        特征处理 + 全局地图更新

        该函数是特征处理的重要函数, 主要负责：
            - 解析原始数据里的信息
            - 解析预处理后的特征数据
            - 对特征进行处理, 并返回处理后的特征向量
            - 特征的拼接
            - 合法动作的标注
            - 【新增】更新全局地图中的宝箱/终点/buff位置
        """
        (
            norm_pos, grid_pos, start_pos, end_pos, buff_pos,
            treasure_pos_list, obstacle_map, memory_map,
            treasure_map, end_map, recent_position_map,
            treasure_collected_count, treasure_count,
        ) = preprocessor.process(raw_obs)

        self._hero_grid_pos = grid_pos

        # === 更新全局地图: 记录宝箱/终点/buff绝对位置 ===
        if raw_obs and hasattr(raw_obs, 'frame_state'):
            self.algorithm.gmap.update_organs(raw_obs.frame_state.organs, get_grid_pos)

        if state_env_info and hasattr(state_env_info, 'game_info'):
            gi = state_env_info.game_info
            if hasattr(gi, 'end_pos') and gi.end_pos:
                ep = get_grid_pos(gi.end_pos.x, gi.end_pos.z)
                self.algorithm.gmap.update_end((ep.x, ep.z))

        # 构建特征向量 (与标准格式兼容)
        one_hot_pos = one_hot_encoding(grid_pos)
        norm_pos_vec = [norm_pos.x, norm_pos.z]
        end_pos_features = read_relative_position(end_pos)
        treasure_features = []
        for tp in treasure_pos_list:
            treasure_features += list(read_relative_position(tp))

        buff_availability = 0
        talent_availability = 0
        if raw_obs:
            for organ in raw_obs.frame_state.organs:
                if organ.sub_type == 2:
                    buff_availability = organ.status
            talent_availability = raw_obs.frame_state.heroes[0].talent.status

        feature_vec = norm_pos_vec + one_hot_pos + end_pos_features + treasure_features + [buff_availability, talent_availability]
        feature_map = obstacle_map + end_map + treasure_map + memory_map
        legal_act = list(raw_obs.legal_act)

        remain_info = {
            "memory_map": memory_map, "end_pos": end_pos, "buff_pos": buff_pos,
            "treasure_pos": treasure_pos_list, "recent_position_map": recent_position_map,
            "treasure_collected_count": treasure_collected_count, "treasure_count": treasure_count,
        }
        self._remain_info = remain_info

        return ObsData(feature=feature_vec + feature_map, legal_act=legal_act), remain_info

    def action_process(self, act_data):
        return act_data.move_dir + act_data.use_talent * 8
