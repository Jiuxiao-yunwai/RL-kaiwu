
from kaiwu_agent.agent.protocol.protocol import SampleData, observation_process, action_process, sample_process
from kaiwu_agent.utils.common_func import Frame
import numpy as np
from random import choice


def run_episodes(n_episode, env, agent, g_data_truncat, extra_info={}):
    for index in range(n_episode):
        collector = list()
        usr_conf = {}

        if env.game_name == "gorge_walk":
            from kaiwu_agent.gorge_walk.data import sort_legal_pos
            pos = np.array(choice(sort_legal_pos[extra_info['idx_start']:extra_info['idx_end']]))
            usr_conf = {"diy":{'start': pos.tolist()}}
        
        obs = env.reset(usr_conf=usr_conf)   # obs = env.reset() aisvr可以传入指定的配置修改游戏配置
        if obs == None:
            continue
        obs_data = observation_process(obs)
        while True:
            act_data = agent.predict(list_obs_data=[obs_data])[0]
            act = action_process(act_data)
            frame_no, _obs, score_info, terminated, truncated, env_info = env.step(act)
            _obs_data = observation_process(_obs)
            if truncated == True and frame_no == None:
                break
            frame = Frame(obs=obs_data.feature, _obs=_obs_data.feature, act=act_data.act, rew=score_info, done=1 if terminated or truncated else 0, ret=score_info)
            
            collector.append(frame)

            if len(collector) % g_data_truncat == 0:
                collector = sample_process(collector)
                yield collector
            
            if terminated or truncated:
                if len(collector) > 0:
                    collector = sample_process(collector)
                    yield collector
                break
            
            obs_data = _obs_data

