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
from agent_target_dqn.conf.conf import Config
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
        direction[rel_pos.direction - 1] = 1

    grid_distance = 1 if rel_pos.grid_distance < 0 else rel_pos.grid_distance / (128 * 128)
    feature = direction + [grid_distance]
    return feature


ACTION_DIRECTION_DELTAS = [
    (0, 1),    # Angle_0, world x+
    (1, 1),    # Angle_45
    (1, 0),    # Angle_90, world z+
    (1, -1),   # Angle_135
    (0, -1),   # Angle_180
    (-1, -1),  # Angle_225
    (-1, 0),   # Angle_270
    (-1, 1),   # Angle_315
]


def _is_active_position(rel_pos):
    return rel_pos.direction != RelativeDirection.RELATIVE_DIRECTION_NONE


def _target_sort_key(rel_pos):
    if rel_pos.grid_distance >= 0:
        return rel_pos.grid_distance
    return rel_pos.l2_distance + 128


def select_navigation_target(end_pos, treasure_pos_list):
    active_treasures = [pos for pos in treasure_pos_list if _is_active_position(pos)]
    if active_treasures:
        return min(active_treasures, key=_target_sort_key)
    return end_pos


def build_wall_aware_legal_act(raw_legal_act, obstacle_map):
    """
    Convert the environment legal action into a 16-dimension mask and block directions
    whose next two grids are unsafe.
    将环境合法动作转换为16维掩码，并屏蔽前方两格不安全的方向。
    """
    raw_legal_act = list(raw_legal_act) if raw_legal_act else [1, 0]
    if len(raw_legal_act) >= 16:
        base_move = raw_legal_act[: Config.DIM_OF_ACTION_DIRECTION]
        base_talent = raw_legal_act[
            Config.DIM_OF_ACTION_DIRECTION : Config.DIM_OF_ACTION_DIRECTION + Config.DIM_OF_TALENT
        ]
    else:
        move_legal = int(raw_legal_act[0]) if len(raw_legal_act) > 0 else 1
        talent_legal = int(raw_legal_act[1]) if len(raw_legal_act) > 1 else 0
        base_move = [move_legal] * Config.DIM_OF_ACTION_DIRECTION
        base_talent = [talent_legal] * Config.DIM_OF_TALENT

    view_len = Config.VIEW_SIZE * 2 + 1
    grid = np.array(obstacle_map, dtype=np.int8).reshape(view_len, view_len)
    center = Config.VIEW_SIZE

    def passable(row, col):
        return 0 <= row < view_len and 0 <= col < view_len and grid[row][col] > 0

    direction_mask = []
    for row_delta, col_delta in ACTION_DIRECTION_DELTAS:
        safe = True
        for step in range(1, Config.ACTION_MASK_LOOKAHEAD + 1):
            row = center + row_delta * step
            col = center + col_delta * step
            if not passable(row, col):
                safe = False
                break
            if row_delta != 0 and col_delta != 0:
                if not passable(center + row_delta * step, center) or not passable(center, center + col_delta * step):
                    safe = False
                    break
        direction_mask.append(1 if safe else 0)

    if not any(direction_mask):
        direction_mask = [1] * Config.DIM_OF_ACTION_DIRECTION

    move_mask = [int(base_move[i] and direction_mask[i]) for i in range(Config.DIM_OF_ACTION_DIRECTION)]
    talent_mask = [int(base_talent[i] and direction_mask[i]) for i in range(Config.DIM_OF_TALENT)]

    if not any(move_mask) and any(base_move):
        move_mask = [int(v) for v in base_move]
    return move_mask + talent_mask


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
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # Copy the model's state dictionary to the CPU
        # 将模型的状态字典拷贝到CPU
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.algorithm.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)

        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.algorithm.model.load_state_dict(torch.load(model_file_path, map_location=self.algorithm.device))
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
        feature, legal_act = [], []

        # Unpack the preprocessed feature data according to the protocol
        # 对预处理后的特征数据按照协议进行解包
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

        # Feature processing 1: One-hot encoding of the current position
        # 特征处理1：当前位置的one-hot编码
        one_hot_pos = one_hot_encoding(grid_pos)

        # Feature processing 2: Normalized position
        # 特征处理2：归一化位置
        norm_pos = [norm_pos.x, norm_pos.z]

        # Feature processing 3: Information about the current position relative to the end point
        # 特征处理3：当前位置相对终点点位的信息
        end_pos_features = read_relative_position(end_pos)

        # Feature processing 4: Information about the current position relative to the treasure position
        # 特征处理4: 当前位置相对宝箱位置的信息
        treasure_pos_features = []
        for treasure_pos in treasure_pos_list:
            treasure_pos_features = treasure_pos_features + list(read_relative_position(treasure_pos))

        # Feature processing 5: Whether the buff is collectable
        # 特征处理5：buff是否可收集
        buff_availability = 0
        if raw_obs:
            for organ in raw_obs.frame_state.organs:
                if organ.sub_type == 2:
                    buff_availability = organ.status

        # Feature processing 6: Whether the flash skill can be used
        # 特征处理6：闪现技能是否可使用
        talent_availability = 0
        if raw_obs:
            talent_availability = raw_obs.frame_state.heroes[0].talent.status

        # Feature processing 7: Next treasure chest to find
        # 特征处理7：下一个需要寻找的宝箱
        navigation_target = select_navigation_target(end_pos, treasure_pos_list)
        end_pos_features = read_relative_position(navigation_target)

        # Feature concatenation:
        # Concatenate all necessary features as vector features (2 + 128*2 + 9  + 9*15 + 2 + 4*51*51 = 10808)
        # 特征拼接：将所有需要的特征进行拼接作为向量特征 (2 + 128*2 + 9  + 9*15 + 2 + 4*51*51 = 10808)
        feature_vec = (
            norm_pos + one_hot_pos + end_pos_features + treasure_pos_features + [buff_availability, talent_availability]
        )
        feature_map = obstacle_map + end_map + treasure_map + memory_map
        # Legal actions
        # 合法动作
        legal_act = build_wall_aware_legal_act(raw_obs.legal_act, obstacle_map)

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
