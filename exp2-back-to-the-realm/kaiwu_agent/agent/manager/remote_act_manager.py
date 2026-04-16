
from kaiwu_agent.agent.protocol.mempoolproto import MemPoolProtocol
from kaiwu_agent.zlib.batchihc import batch_client
from kaiwu_agent.agent.protobuf.rlsar_pb2 import PBObservationList, PBObservation, PBActionList, PBAction
from kaiwu_agent.conf import yaml_rmt_agent
import kaiwu_agent
from kaiwu_agent.zlib.batchproxy import BatchProxy
from dill import loads, dumps

class RemoteActManager():
    def __init__(self, game_name="gorge_walk", scene_name="default_scene", logger=None, monitor=None):
        self.game_name = game_name
        self.scene_name = scene_name

        self.sample_sender = batch_client(1, yaml_rmt_agent.predict.client.url_ftend)
        self.sample_sender.send(None)
        self.proto_sample = MemPoolProtocol()


    def get_remote_action(self, list_obs_data):
        # rlsar_pb2.PBObservation的定义恰好和ObsData一致所以可以相互初始化
        list_pb_obs = [kaiwu_agent.agent.protocol.protocol.ObsData2PBObs(obs_data) for obs_data in list_obs_data]
        pb_obs_list = PBObservationList(observations=list_pb_obs)
        b_obs_list = pb_obs_list.SerializeToString()

        b_act_list = self.sample_sender.send(b_obs_list)
        
        pb_act_list = PBActionList()
        pb_act_list.ParseFromString(b_act_list)

        return [kaiwu_agent.agent.protocol.protocol.PBAct2ActData(pb_act) for pb_act in pb_act_list.actions]
        
    def parse_obs_data(self, b_obs_list):
        pb_obs_list = PBObservationList()
        pb_obs_list.ParseFromString(b_obs_list)
        return [kaiwu_agent.agent.protocol.protocol.PBObs2ObsData(pb_obs) for pb_obs in pb_obs_list.observations]
    
    def ret_remote_action(self, list_act_data):
         # rlsar_pb2.PBAction的定义恰好和ObsAct一致所以可以相互初始化
        list_pb_act = [kaiwu_agent.agent.protocol.protocol.ActData2PBAct(act_data) for act_data in list_act_data]
        pb_act_list = PBActionList(actions=list_pb_act)
        b_act_list = pb_act_list.SerializeToString()
        return b_act_list

    @staticmethod
    def run_batch_proxy():
        batch_proxy = BatchProxy(yaml_rmt_agent.predict.server.url_ftend, yaml_rmt_agent.predict.server.url_bkend)
        batch_proxy.run()
 
class RemoteActManagerZX():
    def __init__(self, game_name="gorge_walk", scene_name="default_scene", logger=None, monitor=None):
        super().__init__(game_name, scene_name, logger, monitor)
    
    def get_remote_action(self, list_obs_data):
        b_obs_list = dumps(list_obs_data)
        b_act_list = self.sample_sender.send(b_obs_list)
        list_act_data = loads(b_act_list)
        return list_act_data
        
        
    def parse_obs_data(self, b_obs_list):
        list_obs_data = loads(b_obs_list)
        return list_obs_data
    
    def ret_remote_action(self, list_act_data):
        b_act_list = dumps(list_act_data)
        return b_act_list