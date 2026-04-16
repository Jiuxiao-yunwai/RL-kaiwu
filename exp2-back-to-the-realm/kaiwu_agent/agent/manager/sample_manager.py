# from shuttle.app.actor.common.rl_data_info import RLDataInfo
import numpy as np
import collections

from kaiwu_agent.agent.protocol.mempoolproto import MemPoolProtocol
import kaiwu_agent
from kaiwu_agent.zlib.rrrclient import client_for_arena
from kaiwu_agent.conf import yaml_rmt_agent
from kaiwu_agent.zlib.zhelper import SAMPLE_REQ

class SampleManager():
    def __init__(self, num_agents, game_name, scene_name='default_scene', logger=None, monitor=None):
        self.game_name = game_name
        self.scene_name = scene_name

        self.sample_sender = client_for_arena(1, yaml_rmt_agent.mempool_fe.client.url_ftend)
        self.sample_sender.send(None)
        self.sample_recver = client_for_arena(1, yaml_rmt_agent.mempool_be.client.url_ftend)
        self.sample_recver.send(None)
        self.proto_sample = MemPoolProtocol()

    def process_and_send(self, g_data):
        samples = [kaiwu_agent.agent.protocol.protocol.SampleData2NumpyData(i) for i in g_data]
        if len(samples) > 0:
            format_samples = self.proto_sample.format_batch_samples_array(samples, priorities=None, max_sample_num=128)
            for format_sample in format_samples:
                self.sample_sender.send(format_sample)

    def recv_and_process(self):
        byte_samples = self.sample_recver.send(SAMPLE_REQ)
        data_batch = MemPoolProtocol().generate_samples(byte_samples)
        return [kaiwu_agent.agent.protocol.protocol.NumpyData2SampleData(i) for i in data_batch]
