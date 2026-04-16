from arena_proto.back_to_the_realm.arena2aisvr_pb2 import (
    AIServerRequest,
    AIServerResponse,
)
from arena_proto.back_to_the_realm.game2arena_pb2 import StepFrameReq, StepFrameRsp
from arena_proto.back_to_the_realm.custom_pb2 import (
    Action,
    FrameState,
    GameInfo,
    RealmHero,
    Position,
    Talent,
    RealmOrgan,

)
from kaiwu_env.back_to_the_realm.utils import get_nature_pos
from kaiwu_env.env.protocol import BaseSkylarenaDataHandler


class Parse_AIServerRequest:

    # AIServerRequest的encode应该由场景接入方在SkaylarenaDataHandler.StepFrameReq2AISvrReq中实现

    @staticmethod
    def decode(byte_aisvr_req):
        """
        将AIServerRequest反序列化后, 转换成用户调用env.reset或env.step期望获得的结构化数据
        转换逻辑由self.game_name决定(场景接入方实现), 该函数逻辑与env.step返回的数据强相关, 需要接入方仔细实现
        """
        aisvr_req = AIServerRequest()
        aisvr_req.ParseFromString(byte_aisvr_req)
        game_id = aisvr_req.game_id
        frame_no = aisvr_req.frame_no
        obs = aisvr_req.obs
        score_info = aisvr_req.score_info
        terminated = aisvr_req.terminated
        truncated = aisvr_req.truncated
        state = aisvr_req.state

        return game_id, frame_no, obs, score_info, terminated, truncated, state


class Parse_AIServerResponse:
    @staticmethod
    def encode(game_id, frame_no, action, stop_game):
        """
        用户env.step传入int或float类型的动作, 转换成AIServerResponse并序列化,
        转换逻辑由self.game_name决定(场景接入方实现), 该函数逻辑与env.step返回的参数强相关, 需要接入方仔细实现
        """
        # action 输入是 0 - 15 ，解析成为Action.move_dir, use_talent
        move_dir = action % 8
        use_talent = action // 8
        return AIServerResponse(
            game_id=game_id,
            frame_no=frame_no,
            action=Action(move_dir=move_dir, use_talent=use_talent),
            stop_game=stop_game,
        ).SerializeToString()

    @staticmethod
    def decode(byte_aisvr_rsp):
        """
        Skylarena中调用, 用来解析aisvr传递过来的byte, 转换成pb, 一般业务方写成
        return AIServerResponse().ParseFromString(byte_aisvr_rsp)
        """
        pb_aisvr_rsp = AIServerResponse()
        pb_aisvr_rsp.ParseFromString(byte_aisvr_rsp)
        return pb_aisvr_rsp


class Parse_StepFrameReq:
    @staticmethod
    def encode(game_id, frame_no, frame_state, terminated, truncated, game_info):
        """
        将game.reset或game.step返回的每个游戏自定义的结构化数据转换成StepFrameReq并序列化
        转换逻辑由self.game_name决定(场景接入方实现), 该函数逻辑与game.step返回的数据强相关, 需要接入方仔细实现
        """
        heroes = []
        if hasattr(frame_state.heroes[0], 'pos_x'):
            hero_pos = Position(x=frame_state.heroes[0].pos_x, z=frame_state.heroes[0].pos_z)
        else:
            hero_pos = Position(x=frame_state.heroes[0].pos.x, z=frame_state.heroes[0].pos.z)
        hero = RealmHero(
            hero_id=frame_state.heroes[0].hero_id,
            pos=hero_pos,
            speed_up=frame_state.heroes[0].speed_up,
            talent=Talent(
                talent_type=frame_state.heroes[0].talent.talent_type,
                status=frame_state.heroes[0].talent.status,
                cooldown=frame_state.heroes[0].talent.cooldown,
            ),
        )
        heroes.append(hero)

        realmOrgans = []
        for organ in frame_state.organs:
            if hasattr(organ, 'pos_x'):
                organ_pos = Position(x=organ.pos_x, z=organ.pos_z)
            else:
                organ_pos = Position(x=organ.pos.x, z=organ.pos.z)
            temp = RealmOrgan(
                sub_type=organ.sub_type,
                config_id=organ.config_id,
                status=organ.status,
                pos=organ_pos,
                cooldown=organ.cooldown,
            )
            realmOrgans.append(temp)

        return StepFrameReq(
            game_id=game_id,
            frame_no=frame_no,
            frame_state=FrameState(
                frame_no=frame_no, heroes=heroes, organs=realmOrgans
            ),
            terminated=1 if terminated else 0,
            truncated=1 if truncated else 0,
            game_info = GameInfo(
                score =game_info.score,
                total_score = game_info.total_score,
                step_no= int(game_info.step_no) ,
                pos=Position(x= int(game_info.hero_pos_x) , z = int(game_info.hero_pos_z)),
                start_pos = get_nature_pos(game_info.start_id),
                end_pos = get_nature_pos(game_info.end_id),
                treasure_collected_count = game_info.treasure_count,
                treasure_score = game_info.treasure_score,
                treasure_count = game_info.total_treasure_count,
                buff_count = game_info.buff_count,
                talent_count = game_info.talent_count,
                buff_remain_time = int(game_info.buff_remain_time),
                buff_duration = int(game_info.buff_duration) 
                
            )
        ).SerializeToString()

    @staticmethod
    def decode(byte_stepframe_req):
        """
        Skaylarena中调用, 用来解析game传递过来的byte, 转换成pb, 一般业务方写成
        return StepFrameReq().ParseFromString(byte_stepframe_req)
        """
        pb_stepframe_req = StepFrameReq()
        pb_stepframe_req.ParseFromString(byte_stepframe_req)
        return pb_stepframe_req


class Parse_StepFrameRsp:

    # StepFrameRsp的encode应该由场景接入方在SkaylarenaDataHandler.AISvrRsp2StepFrameRsp中实现

    @staticmethod
    def decode(byte_game_rsp):
        """
        将StepFrameRsp反序列化得到pb数据, 将pb数据转换成能被game.step接受的输入(每个游戏自定义的结构化数据)
        转换逻辑由self.game_name决定(场景接入方实现), 该函数逻辑与game.step的参数强相关, 需要接入方仔细实现
        """
        game_rsp = StepFrameRsp()
        game_rsp.ParseFromString(byte_game_rsp)
        game_id = game_rsp.game_id
        frame_no = game_rsp.frame_no

        command = {
            "heroid":game_rsp.command.hero_id,
            "move_dir":game_rsp.command.move_dir,
            "talent_type":game_rsp.command.talent_type,
            "move_to_pos_x":game_rsp.command.move_pos.x,
            "move_to_pos_z":game_rsp.command.move_pos.z,
        }

        stop_game = bool(game_rsp.stop_game)
        return game_id, frame_no, command, stop_game
