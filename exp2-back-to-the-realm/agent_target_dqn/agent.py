#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from agent_target_dqn.feature.definition import ObsData
import numpy as np
from kaiwu_agent.agent.base_agent import (
    BaseAgent,
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)
from kaiwu_agent.utils.common_func import attached


from agent_target_dqn.algorithm.algorithm import Algorithm
from arena_proto.back_to_the_realm.custom_pb2 import (
    RelativeDirection,
)


def one_hot_encoding(grid_pos):
    """
    This function performs one_hot_encoding on the grid position features transmitted by proto and returns a vector of length 256.
        - The first 128 dimensions are the one-hot features of the x-axis
        - The last 128 dimensions are the one-hot features of the z-axis

    此函数将proto传输的网格位置特征进行one_hot_encoding处理, 返回一个长度为256的向量
        - 前128维是x轴的one-hot特征
        - 后128维是z轴的one-hot特征
    """
    one_hot_pos_x, one_hot_pos_z = np.zeros(128).tolist(), np.zeros(128).tolist()
    one_hot_pos_x[grid_pos.x], one_hot_pos_z[grid_pos.z] = 1, 1

    return one_hot_pos_x + one_hot_pos_z


def read_relative_position(rel_pos):
    """
    This function unpacks and processes the relative position features transmitted by proto, and returns a vector of length 9.
        - The first 8 dimensions are one-hot direction features
        - The last dimension is the distance feature

    此函数将proto传输的相对位置特征进行拆包并处理, 返回一个长度为9的向量
        - 前8维是one-hot的方向特征
        - 最后一维是距离特征
    """
    direction = [0] * 8
    if rel_pos.direction != RelativeDirection.RELATIVE_DIRECTION_NONE:
        if 1 <= rel_pos.direction <= 8:
            direction[rel_pos.direction - 1] = 1

    if rel_pos.grid_distance < 0:
        grid_distance = 1.0
    else:
        grid_distance = min(float(rel_pos.grid_distance) / 256.0, 1.0)
    feature = direction + [grid_distance]
    return feature


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.agent_type = agent_type
        self.logger = logger
        self.algorithm = Algorithm(device, monitor)

    @predict_wrapper
    def predict(self, list_obs_data):
        return self.algorithm.predict_detail(list_obs_data, exploit_flag=False)

    @exploit_wrapper
    def exploit(self, list_obs_data):
        return self.algorithm.predict_detail(list_obs_data, exploit_flag=True)

    @learn_wrapper
    def learn(self, list_sample_data):
        self.algorithm.learn(list_sample_data)

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.algorithm.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.algorithm.model.load_state_dict(torch.load(model_file_path, map_location=self.algorithm.device))
        self.algorithm.update_target_q()
        self.logger.info(f"load model {model_file_path} successfully")

    def action_process(self, act_data):
        result = act_data.move_dir
        result += act_data.use_talent * 8
        return result

    def observation_process(self, raw_obs, preprocessor, state_env_info=None):
        """
        This function is an important feature processing function, mainly responsible for:
            - Parsing information in the raw data
            - Parsing preprocessed feature data
            - Processing the features and returning the processed feature vector
            - Concatenation of features
            - Annotation of legal actions
        Function inputs:
            - raw_obs: Preprocessed feature data
            - state_env_info: Environment information returned by the game
        Function outputs:
            - observation: Feature vector
            - legal_action: Annotation of legal actions

        该函数是特征处理的重要函数, 主要负责：
            - 解析原始数据里的信息
            - 解析预处理后的特征数据
            - 对特征进行处理, 并返回处理后的特征向量
            - 特征的拼接
            - 合法动作的标注
        函数的输入：
            - raw_obs: 预处理后的特征数据
            - state_env_info: 游戏返回的环境信息
        函数的输出：
            - observation: 特征向量
            - legal_action: 合法动作的标注
        """
        (
            norm_pos,
            grid_pos,
            start_pos,
            end_pos,
            buff_pos,
            treasure_pos_list,
            obstacle_map,
            memory_map,
            treasure_map,
            end_map,
            recent_position_map,
            treasure_collected_count,
            treasure_count,
        ) = preprocessor.process(raw_obs)

        one_hot_pos = one_hot_encoding(grid_pos)
        norm_pos = [norm_pos.x, norm_pos.z]
        end_pos_features = read_relative_position(end_pos)

        treasure_pos_features = []
        for treasure_pos in treasure_pos_list:
            treasure_pos_features = treasure_pos_features + list(read_relative_position(treasure_pos))

        buff_availability = 0
        if raw_obs:
            for organ in raw_obs.frame_state.organs:
                if organ.sub_type == 2:
                    buff_availability = organ.status

        talent_availability = 0
        if raw_obs:
            talent_availability = raw_obs.frame_state.heroes[0].talent.status

        feature_vec = (
            norm_pos + one_hot_pos + end_pos_features + treasure_pos_features + [buff_availability, talent_availability]
        )
        feature_map = obstacle_map + end_map + treasure_map + memory_map
        legal_act = list(raw_obs.legal_act)

        remain_info = {
            "memory_map": memory_map,
            "end_pos": end_pos,
            "buff_pos": buff_pos,
            "treasure_pos": treasure_pos_list,
            "recent_position_map": recent_position_map,
            "treasure_collected_count": treasure_collected_count,
            "treasure_count": treasure_count,
        }

        return ObsData(feature=feature_vec + feature_map, legal_act=legal_act), remain_info
