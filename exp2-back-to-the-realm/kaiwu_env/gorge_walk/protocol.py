
from arena_proto.gorge_walk.arena2aisvr_pb2 import AIServerRequest, AIServerResponse
from arena_proto.gorge_walk.game2arena_pb2 import StepFrameReq, StepFrameRsp
from arena_proto.gorge_walk.custom_pb2 import Action, FrameState, GameInfo, Observation, ScoreInfo, EnvInfo, Command 
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
        obs = aisvr_req.obs.feature
        score_info = aisvr_req.score_info.score
        terminated = aisvr_req.terminated
        truncated = aisvr_req.truncated
        env_info = aisvr_req.env_info.game_info
        
        return game_id, frame_no, obs, score_info, terminated, truncated, env_info

class Parse_AIServerResponse:
    @staticmethod
    def encode(game_id, frame_no, action, stop_game):
        """
            用户env.step传入int或float类型的动作, 转换成AIServerResponse并序列化, 
            转换逻辑由self.game_name决定(场景接入方实现), 该函数逻辑与env.step返回的参数强相关, 需要接入方仔细实现
        """
        return AIServerResponse(
                    game_id = game_id, 
                    frame_no = frame_no, 
                    action = Action(act=action),
                    stop_game = stop_game
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
        return StepFrameReq(
                    game_id = game_id, 
                    frame_no = frame_no, 
                    frame_state = FrameState(
                        game_state = frame_state,
                        legal_act = [0, 1, 2, 3]), 
                    terminated = 1 if terminated else 0,
                    truncated = 1 if truncated else 0,
                    game_info = GameInfo(
                        score = game_info[0],
                        total_score = game_info[1],
                        step_no=game_info[2],
                        pos_x=game_info[3],
                        pos_z=game_info[4],
                        treasure_count=game_info[5],
                        treasure_score=game_info[6],
                        treasure_status=game_info[7] 
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
        command = game_rsp.command.cmd
        stop_game = game_rsp.stop_game
        return game_id, frame_no, command, stop_game
