#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import yaml
import time
import logging
import sys
import os
import numpy as np
from random import choice, sample
from kaiwu_env.conf import yaml_gorge_walk_game as game_conf
from kaiwu_env.gorge_walk.utils import (
    map_init,
    get_legal_pos,
    show_map,
    bump,
    find_treasure,
    show_local_view,
    get_logger,
    discrete_bfs_dist,
)


class Game:
    def __init__(self, scene="fish", max_steps=0, is_MDP=True, logger=logging, monitor=None):
        """
        初始化游戏环境 \n
        输入参数:
            - scene(str): 场景名称 (fish, crab, turtle, bunny)
            - max_steps(int): 最大步数
            - is_MDP(bool): 是否生成一个 MDP 环境, 即是否提供状态转移函数 self.F
            - logger: 日志记录器
        """
        self.scene = scene
        self.max_steps = max_steps if max_steps != 0 else game_conf.max_step
        self.logger = logger
        self.is_MDP = is_MDP
        self.grid, self.height, self.width = map_init(self.scene)
        self.view = game_conf.view
        self.legal_pos = get_legal_pos(self.grid)
        if self.is_MDP:
            from kaiwu_env.conf import json_gorge_walk_F_level_1 as F

            self.F = F  # MDP 条件下的状态转移矩阵

        self.logger.info(f"Game Init: Scene: {self.scene}, Max Steps: {self.max_steps}")
        self.monitor = monitor
        if self.monitor:
            self.last_time = 0
            self.avg_max_steps = None
            self.avg_treasure_random = None
            self.avg_total_treasures = None

    def check_user_conf_valid(self):
        # 排除起点终点在障碍物上的情况
        if tuple(self.start) not in self.legal_pos:
            self.logger.error(f"起点{tuple(self.start)}不在合法位置中")
            raise Exception("起点不在合法位置中")
        if tuple(self.end) not in self.legal_pos:
            self.logger.error(f"终点{tuple(self.end)}不在合法位置中")
            raise Exception("终点不在合法位置中")
        # 排除treasure_id重复
        if len(self.treasure_id) != len(set(self.treasure_id)):
            self.logger.error(f"宝箱id重复")
            raise Exception("treasure_id中元素重复")
        # 排除treasure_count和treasure_id不在规定范围内
        if not set(self.treasure_id).issubset(set([i for i in range(10)])):
            self.logger.error(f"宝箱id{set(self.treasure_id)}应该为{set([i for i in range(10)])}子集")
            raise Exception("treasure_id中元素应该为0-9")
        if self.treasure_count not in [i for i in range(0, 11)]:
            self.logger.error(f"宝箱数{self.treasure_count}不合理,应该为0-10")
            raise Exception("treasure_count应为0-10")
        if self.treasure_random not in [0, 1]:
            self.logger.error(f"treasure_random字段只能为0或1")
            raise Exception("treasure_random字段只能输入0或1")
        # 排除起点、终点、宝箱位置有两个或以上重复
        from kaiwu_env.conf import yaml_gorge_walk_treasure_path_fish as treasure_data

        treasure_pos = [treasure_data[i] for i in self.treasure_id]
        if self.start == self.end:
            self.logger.error(f"终点起点不应重复")
            raise Exception("终点起点不应重复")
        if self.start in treasure_pos:
            self.logger.error(f"起点与宝箱位置重复")
            raise Exception("起点与宝箱位置重复")
        if self.end in treasure_pos:
            self.logger.error(f"终点与宝箱位置重复")
            raise Exception("终点与宝箱位置重复")

    def reset(self, game_id="default-id", usr_conf=None):
        """
        重置游戏环境 \n
        输入值: \n
            - game_id(str): 游戏的唯一标识符, 默认值为 'default-id' \n
            - usr_conf(dict): 用户通过 usr_conf 传入自定义的参数, 如下所示 \n
        user_conf 是一个字典类型, 包含: \n
            - start(list): 起点坐标 \n
            - end(list): 终点坐标 \n
            - treasure_id(list): 宝箱编号, 范围是 [0, 10) \n
        如果 start, end, treasure_id 是 None, 则使用配置文件里的默认值 \n
        返回值:
            - game_id(str): 游戏的唯一标识符 \n
            - frame_no(int): 游戏帧数 \n
            - observation(list): 环境返回的状态 \n
            - terminated(bool): 是否游戏胜利, 即智能体是否走到了终点 \n
            - truncated(bool): 是否游戏超时, 即智能体是否超过了最大步数 \n
            - game_info(tuple): 游戏信息 (score, total_score, treasure_id, treasure_data) \n
        """
        # 解析用户配置, 如果配置为空, 则使用系统默认配置
        self.start, self.end, self.treasure_id, self.treasure_count, self.treasure_random, self.max_steps = (
            self._read_usr_conf(usr_conf)
        )
        self.check_user_conf_valid()
        # treasure_random 表示是否随机生成宝箱, 0表示否，1表示是
        treasure_random = self.treasure_random
        # treasure_random 配置文件中默认为0，若同时输入treasure_id与treasure_random == 1优先随机宝箱
        if self.treasure_random == 1:
            self.treasure_id = sample(range(10), self.treasure_count)

        if self.monitor:
            self.avg_max_steps = self.max_steps if self.avg_max_steps is None else self.avg_max_steps
            self.avg_treasure_random = treasure_random if self.avg_treasure_random is None else self.avg_treasure_random
            self.avg_total_treasures = (
                len(self.treasure_id) if self.avg_total_treasures is None else self.avg_total_treasures
            )

            self.avg_max_steps = game_conf.ALPHA * self.max_steps + (1 - game_conf.ALPHA) * self.avg_max_steps
            self.avg_treasure_random = (
                game_conf.ALPHA * treasure_random + (1 - game_conf.ALPHA) * self.avg_treasure_random
            )
            self.avg_total_treasures = (
                game_conf.ALPHA * len(self.treasure_id) + (1 - game_conf.ALPHA) * self.avg_total_treasures
            )

            now = time.time()
            if now - self.last_time > game_conf.TIME_WINDOW:
                monitor_data = {
                    "max_steps": self.avg_max_steps,
                    "treasure_random": self.avg_treasure_random,
                    "total_treasures": self.avg_total_treasures,
                }
                self.monitor.put_data({os.getpid(): monitor_data})
                self.last_time = now

        # 更新地图信息，包括起点、终点和宝箱
        # 其中 0 表示障碍物, 1 表示可通行, 2 表示起点, 3 表示终点, 4 表示宝箱
        self.logger.debug(f"Game Reset with the following setup:")
        self.treasure_data = self._add_treasure_to_map(self.scene, self.treasure_id)
        self._add_start_end_to_map(self.start, self.end)

        # 初始化游戏状态
        self.game_id = game_id  # 游戏的唯一标识符
        self.step_no = 0  # 游戏步数
        self.frame_no = 1  # 游戏帧数
        self.scores = []  # 每一步的得分
        self.treasure_score = 0  # 累计的宝箱得分
        self.total_score = 0  # 累计的总得分
        self.pos = np.array(self.start)  # 智能体的当前位置
        self.__update_local_view()  # 更新智能体的局部视野
        self.location_memory = [0.0 for _ in range(self.height * self.width)]  # 智能体走过的地图信息初始化
        self.__update_location_memory()  # 更新智能体走过的地图信息
        self.treasure_cnt = 0  # 宝箱的数量
        self.treasure_status = [
            1 if i in self.treasure_id else 2 for i in range(10)
        ]  # 宝箱的状态信息初始化: 1 表示有宝箱 2 表示无宝箱 0表示被收集, 注意组装样本时要把2变成0
        self.game_info = (
            0,
            0,
            self.step_no,
            self.pos[0],
            self.pos[1],
            self.treasure_cnt,
            self.treasure_score,
            self.treasure_status,
        )  # 游戏信息初始化

        return self.game_id, self.frame_no, self._build_obs(), False, False, self.game_info

    def step(self, game_id, frame_no, command, stop_game=False):
        """
        与环境交互, 执行一步游戏 \n
        当达到一个回合的结束时, 你需要负责调用 reset 方法来重置该环境的状态 \n
        功能为接受一个 StepFrameRsp (game_id, frame_no, command, stop_game)
        并返回一个 StepFrameReq (game_id, frame_no, frame_state, terminated, truncated, game_info) \n
        输入参数:
            - command(int): {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT} 的其中一个动作, 由智能体决定 \n
            - game_id(str): 游戏的唯一标识符
            - frame_no(int): 游戏帧数
            - stop_game(bool): 是否停止游戏, 由智能体决定 \n
        返回值:
            - game_id(str): 游戏的唯一标识符
            - frame_no(int): 游戏帧数
            - observation(list): 环境返回的状态
            - terminated(bool): 是否游戏胜利, 即智能体是否走到了终点
            - truncated(bool): 是否游戏超时, 即智能体是否超过了最大步数
            - game_info(tuple): 游戏信息 (score, total_score)
        """
        score, terminated, truncated = 0, False, False

        # 判断智能体是否主动停止游戏
        if stop_game:
            truncated = True

        # 判断游戏是否超时
        if self.step_no < self.max_steps:
            self.step_no += 1
            self.frame_no += 3
            score, terminated, truncated = self._move(command)
        else:
            # 正常情况下, 返回的 frame_state 是下一帧的状态
            # 如果游戏超时(truncated), 则返回上一帧的 observation
            truncated = True

        self.game_info = (
            score,
            self.total_score,
            self.step_no,
            self.pos[0],
            self.pos[1],
            self.treasure_cnt,
            self.treasure_score,
            self.treasure_status,
        )

        return self.game_id, self.frame_no, self._build_obs(), terminated, truncated, self.game_info

    def render(self, mode="local"):
        """
        地图可视化 \n
        输入参数:
            - mode(str): 可视化模式, 可选值为 ['local', 'global']
        local 模式下, 可视化智能体的局部视野, 以智能体所在位置为中心, 大小为 5x5 的矩阵 \n
        该大小可以从配置文件里的 view 参数中调节 \n
        global 模式下, 可视化整个地图, 大小为 64x64 的矩阵 \n
        """
        assert mode in ["local", "global"], f"Invalid render mode: {mode}"

        if mode == "local":
            show_local_view(self.grid, self.pos, self.view)
        elif mode == "global":
            show_map(self.grid)

    def run(self, interactive=False, visualize=False, cmd_list=None):
        """
        与环境交互, 执行一次完整的游戏 \n
        玩家可以在命令行通过输入指令的方式一步一步的手动游玩游戏, 也可以提供智能体每一步决策需要的 cmd_list \n
        如果以上两种方式均未采用, 则智能体每一步的决策是随机的 \n
        玩家还可以选择是否可视化游戏过程 \n
        输入参数:
            - interactive(bool): 是否开启交互模式
            - visualize(bool): 是否开启可视化模式
            - cmd_list(list): 智能体每一步决策需要的指令列表 \n
        返回值:
            - scores(list): 每一步的得分
            - treasure_score(int): 累计的宝箱得分
            - total_score(int): 累计的总得分
        """
        while True:
            self.logger.info(f"Step: {self.step_no}, Pos: {self.pos}, Score: {self.total_score}")
            self.logger.info(f"Tresure Status: {self.treasure_status}")

            if visualize:
                self.render("local")

            if interactive:
                print("Please input your command from the following options: ")
                print("w, a, s, d, q")
                try:
                    cmd = input()
                    if not cmd in ["w", "a", "s", "d", "q"]:
                        continue
                    if cmd == "q":
                        break
                    action = game_conf.command[cmd]
                except:
                    break
            elif cmd_list:
                if self.step_no < len(cmd_list):
                    action = cmd_list[self.step_no - 1]
                    time.sleep(0.2)
                else:
                    break
            else:
                action = choice(range(4))

            _, _, frame_state, terminated, truncated, game_info = self.step(self.game_id, self.frame_no, action)
            print(f"##### {frame_state[189:214]}")
            print(f"##### {frame_state[201]}")
            state = frame_state[0]
            score = game_info[0]

            self.logger.info(
                f"State: {state}, Action: {action}, Score: {score}, Terminated: {terminated}, Truncated: {truncated}"
            )

            if terminated:
                self.logger.info(f"Game Win! Total Score: {self.total_score}, Total Steps: {self.step_no}")
                self.logger.info(f"Treasure Score: {self.treasure_score}, Step Bonus: {score - game_conf.score.win}")
                break

            if truncated:
                self.logger.info(f"Game Fail! Total Score: {self.total_score}, Total Steps: {self.step_no}")
                self.logger.info(f"Treasure Count: {self.treasure_cnt}")
                break

        sys.exit(0)

    def _move(self, action, score_decay=False):
        """
        具体的移动逻辑, 由 step 方法调用 \n
        输入参数:
            - action(int): {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT} 的其中一个动作, 由智能体决定
            - score_decay(bool): 宝箱的得分是否随时间进行衰减 \n
        返回值:
            - score(int): 本次动作的得分
            - terminated(bool): 是否游戏胜利, 即智能体是否走到了终点
        """
        # 返回值的初始化
        score, terminated, truncated = 0, False, False

        # 检查输入动作是否合法, 合法动作为 [0, 1, 2, 3]
        assert action in game_conf.action_space, f"Invalid action: {action}"

        # 将 int 类型的动作转换为 str 类型，对应关系为 {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        action = game_conf.action_space[action]

        # 计算移动后的位置
        new_pos = self.pos + game_conf.direction[action]

        # 撞墙检测，如果撞墙则不移动
        if bump(self.grid, new_pos):
            if game_conf.bump_exit:
                truncated = True
        else:
            self.pos = new_pos
            self.__update_location_memory()

        # 宝箱检测，如果碰到宝箱则获得宝箱的得分, 并更新地图，将宝箱变为可通行
        t_score = 0
        if find_treasure(self.grid, self.pos):
            self.treasure_cnt += 1
            t_score = game_conf.score.treasure
            self.grid[self.pos[0], self.pos[1]] = 1
            # 更新宝箱状态, 需要先找到宝箱的编号
            for key, value in self.treasure_data.items():
                if value == list(self.pos):
                    treasure_id = int(key)
            self.treasure_status[treasure_id] = 0
            if score_decay:
                t_score = t_score * (1 - self.step_no / self.max_steps)
        self.treasure_score += int(t_score)
        score += int(t_score)

        # 终点检测，如果到达终点则获得终点的得分, 以及步数奖励得分
        if list(self.pos) == self.end:
            score += game_conf.score.win
            step_bonus = (self.max_steps - self.step_no) * game_conf.score.step_bonus
            score += step_bonus

            # 游戏结束, 更新状态
            terminated = True

        # 数据更新
        self.scores.append(score)
        self.total_score += score
        self.__update_local_view()

        return int(score), terminated, truncated

    def _read_usr_conf(self, usr_conf):
        def __get_value(key):
            if key in game_conf.diy.keys():
                return game_conf.diy[key]
            elif key in game_conf.keys():
                return game_conf[key]
            else:
                raise KeyError

        if usr_conf:
            game_conf.render_config_from_dict(usr_conf)
        else:
            return (
                game_conf.start,
                game_conf.end,
                game_conf.treasure_id,
                game_conf.treasure_count,
                game_conf.treasure_random,
                game_conf.max_step,
            )

        start = __get_value("start")
        end = __get_value("end")
        treasure_id = __get_value("treasure_id")
        treasure_count = __get_value("treasure_count")
        treasure_random = __get_value("treasure_random")
        max_step = __get_value("max_step")

        return start, end, treasure_id, treasure_count, treasure_random, max_step

    def _add_start_end_to_map(self, start, end):
        """
        将指定的起点和终点信息添加到地图矩阵里 \n
        其中 2 表示起点, 3 表示终点
        """
        if start == end:
            raise ValueError("Start and End should not be the same!")
        if self.grid[start[0], start[1]] in [0, 4]:
            raise ValueError("Invalid start point, can't be the same as obstacle or treasure!")
        if self.grid[end[0], end[1]] in [0, 4]:
            raise ValueError("Invalid end point, can't be the same as obstacle or treasure!")

        self.grid[start[0], start[1]] = 2
        self.grid[end[0], end[1]] = 3

        self.logger.debug(f"Start: {start}, End: {end}")

    def _add_treasure_to_map(self, scene, treasure_id):
        """
        将指定的宝箱信息添加到地图矩阵里 \n
        其中 4 表示宝箱, treasure_id 表示需要生成的宝箱的编号 \n
        返回值:
            - treasure_data(dict): 宝箱的坐标信息, key 是宝箱的编号, value 是宝箱的坐标
        """
        from kaiwu_env.conf import yaml_gorge_walk_treasure_path_fish as treasure_data

        generated_treasure = {}
        for key, value in treasure_data.items():
            if key in treasure_id:
                self.grid[value[0], value[1]] = 4
                generated_treasure[key] = value

        self.logger.debug(f"Treasure: {generated_treasure}")

        return treasure_data

    def _build_obs(self):
        """
        构建环境的观测值 \n
        返回值:
            - observation(list): 环境返回的状态 \n
        具体包括:
            - state(int): 当前状态, 0-4095, s = x * 64 + y [0]
            - pos_row(list): 当前位置横坐标的 one-hot 编码  [1:65]
            - pos_col(list): 当前位置纵坐标的 one-hot 编码  [65:129]
            - end_dist(int): 当前位置相对于终点的离散化距离, 0-6, 数字越到表示越远 [129:130]
            - treasure_dist(list): 当前位置相对于宝箱的离散化距离, 0-6, 数字越到表示越远, 未生成的宝箱用999表示  [130:140]
            - obstacle_flat(list): flat 后的局部视野中障碍物的信息, 1 表示障碍物, 0 表示可通行  [140:165]
            - treasure_flat(list): flat 后的局部视野中宝箱的信息, 1 表示宝箱, 0 表示没有宝箱    [165:190]
            - end_flat(list): flat 后的局部视野中终点的信息, 1 表示终点, 0 表示没有终点        [190:215]
            - memory_flat(list): flat 后的局部记忆图, 0-1, 每走过一次, 记忆值+0.1 [215:240]
            - treasure_status(list): 宝箱的状态, 1 表示可以被收集, 0 表示不可被收集(未生成或者以及收集过), 长度为 10 [240:250]
        """
        # 特征#1: 智能体当前 state (1维表示)
        state = [int(self.pos[0] * 64 + self.pos[1])]

        # 特征#2: 智能体当前位置信息的 one-hot 编码
        pos_row = [0] * 64
        pos_row[self.pos[0]] = 1
        pos_col = [0] * 64
        pos_col[self.pos[1]] = 1

        # 特征#3: 智能体当前位置相对于终点的距离(离散化)
        end_dist = [discrete_bfs_dist(self.pos, self.end)]

        # 特征#4: 智能体当前位置相对于宝箱的距离(离散化)
        treasure_dist = []
        for i in range(10):
            # 对于未生成的宝箱，返回一个 999
            if i not in self.treasure_id:
                treasure_dist.append(999)
            else:
                treasure_dist.append(discrete_bfs_dist(self.pos, self.treasure_data[i]))

        # 特征#5: 图特征生成(障碍物信息, 宝箱信息, 终点信息)
        obstacle_map, treasure_map, end_map = [], [], []
        for sub_list in self.local_view:
            obstacle_map.append([1 if i == 0 else 0 for i in sub_list])
            treasure_map.append([1 if i == 4 else 0 for i in sub_list])
            end_map.append([1 if i == 3 else 0 for i in sub_list])

        # 特征#6: 图特征转换为向量特征
        obstacle_flat, treasure_flat, end_flat = [], [], []
        for i in obstacle_map:
            obstacle_flat.extend(i)
        for i in treasure_map:
            treasure_flat.extend(i)
        for i in end_map:
            end_flat.extend(i)

        # 特征#7: 智能体当前局部视野中的走过的地图信息
        memory_flat = []
        for i in range(self.view * 2 + 1):
            idx_start = (self.pos[0] - self.view + i) * 64 + (self.pos[1] - self.view)
            memory_flat.extend(self.location_memory[idx_start : (idx_start + self.view * 2 + 1)])

        tmp_treasure_status = [x if x != 2 else 0 for x in self.treasure_status]
        return np.concatenate(
            [
                state,
                pos_row,
                pos_col,
                end_dist,
                treasure_dist,
                obstacle_flat,
                treasure_flat,
                end_flat,
                memory_flat,
                tmp_treasure_status,
            ]
        )

    def __update_local_view(self):
        """
        更新智能体的局部视野 \n
        返回值: 以智能体所在位置为中心, 大小为 5x5 的矩阵 \n
        其中, 0 表示障碍物, 1 表示可通行, 2 表示起点, 3 表示终点, 4 表示宝箱
        """
        view = self.view
        self.local_view = [[0 for _ in range(view * 2 + 1)] for _ in range(view * 2 + 1)]
        x, y = self.pos[0], self.pos[1]

        for i in range(-view, view + 1):
            for j in range(-view, view + 1):
                if x + i >= 0 and x + i < self.width and y + j >= 0 and y + j < self.height:
                    self.local_view[view + i][view + j] = self.grid[x + i][y + j]

        return self.local_view

    def __update_location_memory(self):
        """
        更新智能体走过的地图的记忆信息 \n
        返回值: 64x64 的全局矩阵, 其中每个元素的取值范围是 [0, 1], 每走过一次，记忆值增加 0.1
        """
        idx = self.pos[0] * 64 + self.pos[1]
        self.location_memory[idx] = min(1.0, self.location_memory[idx] + 0.1)

        return self.location_memory


if __name__ == "__main__":
    print(game_conf.diy)

    logger = get_logger(level=logging.DEBUG)

    # Create a game environment
    env = Game(logger=logger)

    # Reset the game environment
    usr_conf = {
        "diy": {
            "start": [29, 9],
            "end": [11, 55],
            "treasure_id": [0, 1, 2, 3, 4],
        }
    }
    env.reset(usr_conf=usr_conf)

    # Show the global map
    env.render("global")

    # Run the game in interactive mode
    env.run(interactive=True, visualize=True)
