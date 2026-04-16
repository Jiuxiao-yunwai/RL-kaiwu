import datetime
import json
import os
import time

from arena_proto.arena2plat_pb2 import CampInfo, GameData, GameStatus
from arena_proto.back_to_the_realm.arena2aisvr_pb2 import (
    AIServerRequest,
    AIServerResponse,
)
from arena_proto.back_to_the_realm.custom_pb2 import (
    Action,
    Command,
    EndInfo,
    State,
    Frame,
    Frames,
    FrameState,
    GameInfo,
    Observation,
    Position,
    RealmHero,
    RealmOrgan,
    ScoreInfo,
    StartInfo,
    MapInfo,
)
from arena_proto.back_to_the_realm.game2arena_pb2 import StepFrameReq, StepFrameRsp
from google.protobuf.json_format import MessageToJson
from google.protobuf.timestamp_pb2 import Timestamp
import numpy as np

from kaiwu_env.back_to_the_realm.utils import (
    _print_debug_log,
    convert_pos_to_grid_pos,
    get_game_info,
    get_hero_info,
    get_local_frame_state,
    get_local_grid_info,
    get_nature_pos,
    get_organ_info,
    get_legal_act,
    polar_to_pos,
)
from kaiwu_env.conf import json_back_to_the_realm_map_data_map_1 as map_data
from kaiwu_env.conf import yaml_arena
from kaiwu_env.conf import yaml_back_to_the_realm_game as game_conf
from kaiwu_env.conf import yaml_back_to_the_realm_treasure_path_crab as treasure_data
from kaiwu_env.env.protocol import BaseSkylarenaDataHandler


class SkylarenaDataHandler(BaseSkylarenaDataHandler):
    def __init__(self, logger, monitor) -> None:
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0

        if self.monitor:
            self.last_time = 0
            self.avg_finished_steps = 0
            self.avg_total_score = 0
            self.avg_collected_treasures = 0
            self.avg_treasure_score = 0

            self.avg_buff_count = 0
            self.avg_skill_sount = 0

            self.avg_max_steps = None
            self.avg_treasure_random = None
            self.avg_total_treasures = None

            self.last_episode_cnt = 0

    def reset(self, usr_conf):
        if self.logger:
            self.logger.info(f"reset usr_conf is {usr_conf}")

        # 对局相关数据初始化
        self.game_id = None
        self.game_status = 0
        self.frame_no = 0
        self.step_no = 0
        self.score = 0
        self.max_steps = usr_conf["env_conf"]["max_step"]
        self.total_score = 0
        self.buff_count = 0
        # 已收集到的宝箱个数
        self.treasure_collected_count = 0
        self.treasure_score = 0
        self.treasure_data = treasure_data
        self.frames = Frames()
        self.speed_up = 0
        self.talent_count = 0
        self.hero_pos = get_nature_pos(usr_conf["env_conf"]["start"])
        self.telent_tpye = 0
        self.telent_status = 0
        self.telent_cooldown = 0
        self.organs = list()
        self.start_id = usr_conf["env_conf"]["start"]
        self.end_id = usr_conf["env_conf"]["end"]

        self.treasure_random = 1 if usr_conf["env_conf"]["treasure_random"] == 1 else 0

        # 获取当前的UTC时间
        now = datetime.datetime.utcnow()
        self.start_timestamp = Timestamp()
        self.start_timestamp.FromDatetime(now)
        # 对局开始信息
        self.start_info = StartInfo(
            start=get_nature_pos(usr_conf["env_conf"]["start"]),
            end=get_nature_pos(usr_conf["env_conf"]["end"]),
        )

        if self.monitor:
            self.avg_max_steps = self.max_steps if self.avg_max_steps is None else self.avg_max_steps
            self.avg_treasure_random = (
                self.treasure_random if self.avg_treasure_random is None else self.avg_treasure_random
            )

            self.avg_max_steps = game_conf.ALPHA * self.max_steps + (1 - game_conf.ALPHA) * self.avg_max_steps
            self.avg_treasure_random = (
                game_conf.ALPHA * self.treasure_random + (1 - game_conf.ALPHA) * self.avg_treasure_random
            )

    def step(self, pb_stepframe_req, pb_aisvr_rsp):
        self.frame_no = pb_stepframe_req.frame_no
        self.step_no = pb_stepframe_req.game_info.step_no
        self.score = pb_stepframe_req.game_info.score
        self.total_score = pb_stepframe_req.game_info.total_score
        self.treasure_collected_count = pb_stepframe_req.game_info.treasure_collected_count
        self.treasure_score = pb_stepframe_req.game_info.treasure_score
        self.hero_pos = pb_stepframe_req.frame_state.heroes[0].pos
        self.speed_up = pb_stepframe_req.frame_state.heroes[0].speed_up
        self.buff_count = pb_stepframe_req.game_info.buff_count
        self.talent_count = pb_stepframe_req.game_info.talent_count
        self.telent_tpye = pb_stepframe_req.frame_state.heroes[0].talent.talent_type
        self.telent_status = pb_stepframe_req.frame_state.heroes[0].talent.status
        self.telent_cooldown = pb_stepframe_req.frame_state.heroes[0].talent.cooldown
        # self.organs=get_organ_info(pb_stepframe_req)

        if self.step_no == 1:
            self.game_id = pb_stepframe_req.game_id
            init_organs = get_organ_info(pb_stepframe_req.frame_state)
            self.start_info.organs.extend(init_organs)

            # 如果是第一步，构造第0步的初始数据
            import copy

            hero = copy.deepcopy(pb_stepframe_req.frame_state.heroes[0])
            game_info = copy.deepcopy(pb_stepframe_req.game_info)

            hero.pos.x = self.start_info.start.x
            hero.pos.z = self.start_info.start.z
            hero.talent.status = 1
            hero.talent.cooldown = 0
            game_info.talent_count = 0
            frame = Frame(
                frame_no=1,
                step_no=0,
                hero=get_hero_info(hero),
                organs=get_organ_info(pb_stepframe_req.frame_state),
                game_info=get_game_info(hero, game_info, 0),
            )
            self.frames.frames.append(frame)

        frame = Frame(
            frame_no=self.frame_no,
            step_no=self.step_no,
            hero=get_hero_info(pb_stepframe_req.frame_state.heroes[0]),
            organs=get_organ_info(pb_stepframe_req.frame_state),
            game_info=get_game_info(
                pb_stepframe_req.frame_state.heroes[0], pb_stepframe_req.game_info, pb_stepframe_req.frame_no
            ),
        )
        self.frames.frames.append(frame)

        # _print_debug_log(self=self,freq=3)

        pass

    def report_monitor_info(self):
        """
        由于gamecore可能会异常, entity进程会在容灾脚本里拉起来, 导致这期间的监控数据没有及时上报, 从而展示出来有下降的趋势, 故这里专门处理了对局数目的监控上报
        """
        now = time.time()
        if now - self.last_report_monitor_time > game_conf.TIME_WINDOW:
            monitor_data = {
                "episode_cnt": self.episode_cnt,
            }

            if self.monitor:
                # 使用 monitor_label 作为监控的 label
                self.monitor.put_data({os.getpid(): monitor_data})
            self.last_report_monitor_time = now

    def finish(self):
        if self.game_id == None:
            self.logger.error("finish enter no game_id, so return")
            return

        self.episode_cnt += 1
        # 如果超过最大步数，需要额外处理
        self.game_status = 3 if (self.step_no + 1) >= self.max_steps else 1
        self.total_score = 0 if (self.step_no + 1) >= self.max_steps else self.total_score
        # step_no 截断
        self.step_no = min(self.step_no + 1, self.max_steps)

        if self.monitor:
            self.avg_finished_steps += self.step_no
            self.avg_total_score += int(self.total_score)
            self.avg_collected_treasures += int(self.treasure_collected_count)
            self.avg_treasure_score += int(self.treasure_score)
            self.avg_buff_count += int(self.buff_count)
            self.avg_skill_sount += int(self.talent_count)

            now = time.time()
            if now - self.last_time > game_conf.TIME_WINDOW and self.episode_cnt > self.last_episode_cnt:
                monitor_data = {
                    "finished_steps": self.avg_finished_steps / (self.episode_cnt - self.last_episode_cnt),
                    "score": int(self.score),  # self.avg_total_score / (self.episode_cnt - self.last_episode_cnt),
                    "total_score": self.avg_total_score / (self.episode_cnt - self.last_episode_cnt),
                    "collected_treasures": self.avg_collected_treasures / (self.episode_cnt - self.last_episode_cnt),
                    "treasure_score": self.avg_treasure_score / (self.episode_cnt - self.last_episode_cnt),
                    "buff_cnt": self.avg_buff_count / (self.episode_cnt - self.last_episode_cnt),
                    "skill_cnt": self.avg_skill_sount / (self.episode_cnt - self.last_episode_cnt),
                    "max_steps": self.avg_max_steps,
                    "treasure_random": self.avg_treasure_random,
                    "total_treasures": self.avg_total_treasures,
                }
                self.monitor.put_data({os.getpid(): monitor_data})
                self.last_time = now

                self.avg_finished_steps = 0
                self.avg_total_score = 0
                self.avg_collected_treasures = 0
                self.avg_treasure_score = 0
                self.avg_buff_count = 0
                self.avg_skill_sount = 0
                self.last_episode_cnt = self.episode_cnt
                self.logger.info(f"finish monitor_data is {monitor_data}")

        # 只有在评估模式下才会落平台数据
        if yaml_arena.train_or_eval == "eval" or yaml_arena.train_or_eval == "exam":
            log_folder = yaml_arena.platform_log_dir
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            self.save_game_stat(f"{log_folder}/{self.game_id}.json")
            self.logger.info(f"save_game_stat success, file_path is {log_folder}/{self.game_id}.json")
        self.logger.info("data_handler finish success")

    def save_game_stat(self, file_path):
        """
        根据后端pb协议返回对局数据, 保存到json文件,不暴露给用户
        """
        end_info = EndInfo(
            frame=self.frame_no,
            step=self.step_no,
            total_score=int(self.total_score),
            treasure_count=int(self.treasure_collected_count),
            treasure_score=int(self.treasure_score),
            buff_count=int(self.buff_count),
            talent_count=int(self.talent_count),
        )

        camp = CampInfo(
            camp_type="blue",
            camp_code="A",
            start_info=MessageToJson(
                self.start_info,
                including_default_value_fields=True,
                preserving_proto_field_name=True,
                use_integers_for_enums=True,
            ),
            end_info=MessageToJson(
                end_info,
                including_default_value_fields=True,
                preserving_proto_field_name=True,
                use_integers_for_enums=True,
            ),
        )
        json_messages = MessageToJson(
            self.frames,
            including_default_value_fields=True,
            preserving_proto_field_name=True,
            use_integers_for_enums=True,
        )

        # 获取当前的UTC时间
        now = datetime.datetime.utcnow()
        # 将当前时间转换为protobuf的Timestamp类型
        end_timestamp = Timestamp()
        end_timestamp.FromDatetime(now)

        output = GameData(
            name=self.game_id,
            project_code="back_to_the_realm",
            status=self.game_status,
            camps=[camp],
            frames=json_messages,
            start_time=self.start_timestamp,
            end_time=end_timestamp,
        )

        # 将pb数据转换成json格式, 保存到文件
        with open(file_path, "w") as outfile:
            out_data = MessageToJson(
                output,
                including_default_value_fields=True,
                preserving_proto_field_name=True,
                use_integers_for_enums=True,
            )
            json.dump(json.loads(out_data), outfile, indent=4)

        # 在写完json文件后再写一个done文件，前面的文件名保持一致
        done_file = file_path.replace("json", "done")
        with open(done_file, "w") as done:
            done.writelines("done")

    def StepFrameReq2AISvrReq(self, pb_stepframe_req):
        """
        pb_stepframe_req 是已经反序列化后的StepFrameReq
        """
        # 只在第一帧上报监控
        if pb_stepframe_req.frame_no == 1 and self.monitor:
            # 去除掉buff的长度,orgins -1
            self.avg_total_treasures = (
                (len(pb_stepframe_req.frame_state.organs) - 1)
                if self.avg_total_treasures is None
                else self.avg_total_treasures
            )

            self.avg_total_treasures = (
                game_conf.ALPHA * (len(pb_stepframe_req.frame_state.organs) - 1)
                + (1 - game_conf.ALPHA) * self.avg_total_treasures
            )

        hero = pb_stepframe_req.frame_state.heroes[0]
        # 天赋技能状态
        talent_status = hero.talent.status

        frame_state = pb_stepframe_req.frame_state
        game_info = pb_stepframe_req.game_info

        # 处理局部视野
        view_size = game_conf.view

        local_frame_state = get_local_frame_state(frame_state, view_size)

        grid = np.array(map_data["Flags"]).reshape(map_data["Height"], map_data["Width"])

        hero_grid_x, hero_grid_z = convert_pos_to_grid_pos(hero.pos.x, hero.pos.z)

        local_grid_info = get_local_grid_info(grid, (hero_grid_x, hero_grid_z), view_size)

        observation = Observation(
            frame_state=local_frame_state, game_info=game_info, legal_act=get_legal_act(talent_status)
        )

        # 将 local_grid_info 存储到 obs.map_info
        for row in local_grid_info:
            map_info = MapInfo()
            map_info.values.extend(row)
            observation.map_info.append(map_info)

        return AIServerRequest(
            game_id=pb_stepframe_req.game_id,
            frame_no=pb_stepframe_req.frame_no,
            obs=observation,
            score_info=ScoreInfo(
                score=int(pb_stepframe_req.game_info.score),
                total_score=int(pb_stepframe_req.game_info.total_score),
                treasure_count=int(pb_stepframe_req.game_info.treasure_collected_count),
                buff_count=int(pb_stepframe_req.game_info.buff_count),
                talent_count=int(pb_stepframe_req.game_info.talent_count),
            ),
            terminated=pb_stepframe_req.terminated,
            truncated=pb_stepframe_req.truncated,
            state=State(),
        ).SerializeToString()

    def AISvrRsp2StepFrameRsp(self, pb_aisvr_rsp):
        position = polar_to_pos(
            self.hero_pos,
            pb_aisvr_rsp.action.move_dir,
            bool(pb_aisvr_rsp.action.use_talent),
            bool(self.speed_up),
            bool(self.telent_status),
        )

        command = Command(
            hero_id=1112,
            move_dir=pb_aisvr_rsp.action.move_dir,
            talent_type=pb_aisvr_rsp.action.use_talent,
            move_pos=position,
        )

        return StepFrameRsp(
            game_id=pb_aisvr_rsp.game_id,
            frame_no=pb_aisvr_rsp.frame_no,
            command=command,
            stop_game=1 if pb_aisvr_rsp.stop_game else 0,
        ).SerializeToString()
