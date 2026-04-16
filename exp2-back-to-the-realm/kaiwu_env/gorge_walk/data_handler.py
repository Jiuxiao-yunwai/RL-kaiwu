from arena_proto.gorge_walk.arena2aisvr_pb2 import AIServerRequest, AIServerResponse
from arena_proto.gorge_walk.game2arena_pb2 import StepFrameReq, StepFrameRsp
from arena_proto.gorge_walk.custom_pb2 import (
    Action,
    FrameState,
    GameInfo,
    Observation,
    ScoreInfo,
    EnvInfo,
    Command,
)
from arena_proto.gorge_walk.custom_pb2 import (
    StartInfo,
    Frame,
    Frames,
    EndInfo,
    GorgeWalkHero,
    GorgeWalkOrgan,
    GorgeWalkPosition,
)
from arena_proto.arena2plat_pb2 import GameData, CampInfo, GameStatus
from kaiwu_env.env.protocol import BaseSkylarenaDataHandler
import os
from google.protobuf.json_format import MessageToJson
import json
from kaiwu_env.gorge_walk.utils import get_nature_pos, get_hero_info, get_organ_info
from kaiwu_env.conf import yaml_gorge_walk_treasure_path_fish as treasure_data
from kaiwu_env.conf import yaml_arena
from google.protobuf.timestamp_pb2 import Timestamp
from datetime import datetime
from kaiwu_env.conf import yaml_gorge_walk_game as game_conf
import time
from common_python.utils.common_func import register_sigterm_handler
from common_python.config.config_control import CONFIG


class SkylarenaDataHandler(BaseSkylarenaDataHandler):
    def __init__(self, logger, monitor) -> None:
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        if self.monitor:
            self.last_time = 0
            self.avg_finished_steps = 0
            self.avg_total_score = 0
            self.avg_collected_treasures = 0
            self.avg_treasure_score = 0
            self.last_episode_cnt = 0

            # 注册SIGTERM信号处理, 如果没有设置监控, 则不用走安全退出逻辑
            register_sigterm_handler(self.handle_sigterm, CONFIG.sigterm_pids_file)

    def reset(self, usr_conf):
        # 对局相关数据初始化
        self.game_id = None
        # 根据开悟后端协议, enum GameStatus { unknown = 0;   // 未知状态，默认占位
        # success = 1;   // 成功
        # error = 2;     // 失败
        # overtime = 3;  // 超时
        self.game_status = 0
        self.frame_no = 0
        self.step_no = 0
        self.max_steps = (
            usr_conf["diy"]["max_step"]
            if "max_step" in usr_conf["diy"]
            else game_conf.max_step
        )
        self.total_score = 0
        self.treasure_cnt = 0
        self.treasure_score = 0
        self.treasure_data = treasure_data
        self.frames = Frames()

        # 获取当前的UTC时间
        now = datetime.utcnow()
        # 将当前时间转换为protobuf的Timestamp类型
        self.start_timestamp = Timestamp()
        self.start_timestamp.FromDatetime(now)

        # 对局开始信息
        self.start_info = StartInfo(
            start=get_nature_pos(usr_conf["diy"]["start"]),
            end=get_nature_pos(usr_conf["diy"]["end"]),
        )

    def step(self, pb_stepframe_req, pb_aisvr_rsp):
        self.frame_no = pb_stepframe_req.frame_no
        self.step_no += 1
        self.total_score = pb_stepframe_req.game_info.total_score
        self.treasure_cnt = pb_stepframe_req.game_info.treasure_count
        self.treasure_score = pb_stepframe_req.game_info.treasure_score

        if self.step_no == 1:
            self.game_id = pb_stepframe_req.game_id
            init_organs = get_organ_info(pb_stepframe_req)
            self.start_info.organs.extend(init_organs)

        frame = Frame(
            frame_no=self.frame_no,
            step_no=self.step_no,
            hero=get_hero_info(pb_stepframe_req),
            organs=get_organ_info(pb_stepframe_req),
        )
        self.frames.frames.append(frame)

        pass

    def finish(self):
        if self.game_id == None:
            return

        self.episode_cnt += 1
        # 如果超过最大步数，需要额外处理
        self.game_status = 3 if self.step_no > self.max_steps else 1
        self.total_score = 0 if self.step_no > self.max_steps else self.total_score
        # step_no 截断
        self.step_no = min(self.step_no, self.max_steps)

        if self.monitor:
            # 算术平均
            self.avg_finished_steps += self.step_no
            self.avg_total_score += int(self.total_score)
            self.avg_collected_treasures += int(self.treasure_cnt)
            self.avg_treasure_score += int(self.treasure_score)

            # 指数平均
            # self.avg_finished_steps      = game_conf.ALPHA * self.step_no + (1 - game_conf.ALPHA) * self.avg_finished_steps
            # self.avg_total_score         = game_conf.ALPHA * int(self.total_score) + (1 - game_conf.ALPHA) * self.avg_total_score
            # self.avg_collected_treasures = game_conf.ALPHA * int(self.treasure_cnt) + (1 - game_conf.ALPHA) * self.avg_collected_treasures
            # self.avg_treasure_score      = game_conf.ALPHA * int(self.treasure_score) + (1 - game_conf.ALPHA) * self.avg_treasure_score

            now = time.time()
            if (
                now - self.last_time > game_conf.TIME_WINDOW
                and self.episode_cnt > self.last_episode_cnt
            ):
                monitor_data = {
                    "finished_steps": self.avg_finished_steps
                    / (self.episode_cnt - self.last_episode_cnt),
                    "total_score": self.avg_total_score
                    / (self.episode_cnt - self.last_episode_cnt),
                    "collected_treasures": self.avg_collected_treasures
                    / (self.episode_cnt - self.last_episode_cnt),
                    "treasure_score": self.avg_treasure_score
                    / (self.episode_cnt - self.last_episode_cnt),
                    "episode_cnt": self.episode_cnt,
                }
                self.monitor.put_data({os.getpid(): monitor_data})
                self.last_time = now

                self.avg_finished_steps = 0
                self.avg_total_score = 0
                self.avg_collected_treasures = 0
                self.avg_treasure_score = 0
                self.last_episode_cnt = self.episode_cnt

        # 只有在评估模式下才会落平台数据
        if yaml_arena.train_or_eval == "eval" or yaml_arena.train_or_eval == "exam":
            log_folder = yaml_arena.platform_log_dir
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)
            self.save_game_stat(f"{log_folder}/{self.game_id}.json")

    def save_game_stat(self, file_path):
        """
        根据后端pb协议返回对局数据, 保存到json文件,不暴露给用户
        """

        end_info = EndInfo(
            frame=self.frame_no,
            step=self.step_no,
            total_score=int(self.total_score),
            treasure_count=int(self.treasure_cnt),
            treasure_score=int(self.treasure_score),
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
        now = datetime.utcnow()
        # 将当前时间转换为protobuf的Timestamp类型
        end_timestamp = Timestamp()
        end_timestamp.FromDatetime(now)

        output = GameData(
            name=self.game_id,
            project_code="gorge_walk",
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
        observation = Observation(
            feature=pb_stepframe_req.frame_state.game_state,
            legal_act=pb_stepframe_req.frame_state.legal_act,
        )

        return AIServerRequest(
            game_id=pb_stepframe_req.game_id,
            frame_no=pb_stepframe_req.frame_no,
            obs=observation,
            score_info=ScoreInfo(score=pb_stepframe_req.game_info.score),
            terminated=pb_stepframe_req.terminated,
            truncated=pb_stepframe_req.truncated,
            env_info=EnvInfo(),
        ).SerializeToString()

    def AISvrRsp2StepFrameRsp(self, pb_aisvr_rsp):
        return StepFrameRsp(
            game_id=pb_aisvr_rsp.game_id,
            frame_no=pb_aisvr_rsp.frame_no,
            command=Command(cmd=pb_aisvr_rsp.action.act),
            stop_game=1 if pb_aisvr_rsp.stop_game else 0,
        ).SerializeToString()

    def handle_sigterm(self, sig, frame):
        self.logger.info(
            f"data_handler {os.getpid()} is starting to handle the SIGTERM signal."
        )
        if hasattr(self, "monitor") and self.episode_cnt > self.last_episode_cnt:
            monitor_data = {"episode_cnt": self.episode_cnt}
            self.monitor.put_data({os.getpid(): monitor_data})

            # 确保监控数据上传
            time.sleep(5)
