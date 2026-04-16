
import kaiwu_agent
from kaiwu_agent.conf import tree_switch
from kaiwu_agent.conf import yaml_agent
from time import sleep
from kaiwu_agent.agent.manager.mempool_manager import sample_recv, sample_send
from kaiwu_agent.agent.manager.mempool_manager import MempoolManager
from kaiwu_agent.agent.manager.sample_manager import SampleManager
from kaiwu_agent.agent.manager.model_manager import ModelManager
from kaiwu_agent.agent.manager.model_deliver_manager import ModelDeliverManager
from kaiwu_agent.episode import run_episodes
from kaiwu_agent.utils.common_func import wrap_fn_2_process

from kaiwu_agent.zlib.batchihc import batch_worker
from kaiwu_agent.zlib.zhelper import WORKER_FAULT_RESULT, BATCH_SIZE
from kaiwu_agent.conf import yaml_rmt_agent
from kaiwu_agent.agent.manager.remote_act_manager import RemoteActManager
import os, signal

def run_arena(cls_agent, game_name, scene_name='default_scene'):
    if not tree_switch.check("remote_env"):
        kaiwu_agent.setup(run_mode='entity', entity_type='cloak')
    else:
        kaiwu_agent.setup(run_mode='proxy')
    env = kaiwu_agent.make(game_name)
    agent = cls_agent("player")

    # remote_sample打开时, 训练样本发送到mempool, 要求打开mempool, 开关mempool_mgr=true
    if tree_switch.check("remote_sample"):
        sample_mgr = SampleManager(yaml_agent.agent.agent_num, game_name)
        # 单元测试用, 如果没有打开mempool_mgr, 则启动一个用于测试
        if not tree_switch.check("mempool_mgr"):
            run_mempool(game_name, scene_name)
    
    # 单元测试用, 如果需要远程预测但是没有打开mempool_mgr, 则启动一个和actor用于测试
    if tree_switch.check("remote_predict") and not tree_switch.check("remote_act_mgr"):
        run_rmt_act_proxy()
        subproc_run_agent(cls_agent, "actor", game_name)

    if tree_switch.check("remote_model"):
        model_mgr = ModelManager()

    episode_num_every_epoch = yaml_agent.agent.episode_num_every_epoch
    g_data_truncat = yaml_agent.agent.g_data_truncat
    extra_info = {}

    if game_name == "gorge_walk":
        extra_info = {
            'idx_start': 0, 
            'idx_end': 10
        }

    for index in range(yaml_agent.agent.epoch_num):
        epoch_total_rew = 0
        for g_data in run_episodes(episode_num_every_epoch, env, agent, g_data_truncat, extra_info):
            total_rew = sum([i.rew for i in g_data])
            epoch_total_rew += total_rew
            if tree_switch.check("remote_sample"):
                sample_mgr.process_and_send(g_data)
            if tree_switch.check("remote_model"):
                model_mgr.update_model(agent)
            if not tree_switch.check("remote_learn"):
                agent.learn(g_data)
        print(f"total_rew: {epoch_total_rew/episode_num_every_epoch}")

        if game_name == "gorge_walk":
            if epoch_total_rew/episode_num_every_epoch > 200:
                extra_info['idx_start'] = extra_info['idx_start'] + 10
                extra_info['idx_end'] = extra_info['idx_end'] +10
                print("more difficult: ", extra_info['idx_start'])  


def run_mempool(game_name, scene_name='default_scene'):
    mempool_mgr = MempoolManager(game_name, scene_name)
    mempool_mgr.run()

    for i in range(1):
        if tree_switch.check("remote_sample"):
            sample_recv(i, mempool_mgr.mempool)
        if tree_switch.check("receive_sample"):
            sample_send(i, mempool_mgr.batch_mgr)

def run_rmt_act_proxy():
    RemoteActManager.run_batch_proxy()

def run_agent(cls_agent, agent_type, game_name=None, scene_name=None):
    agent = cls_agent(agent_type)
    if agent_type == "actor":
        # 载入远程传来的模型
        model_mgr = ModelManager() if tree_switch.check("receive_model") else None
        # 启动一个worker去服务
        worker = batch_worker(1, yaml_rmt_agent.predict.worker.url_bkend, True, agent.remote_predict, model_mgr, )
        worker.send(None)
        while True:
            byte_list_rsp = worker.send(None)
            if byte_list_rsp == WORKER_FAULT_RESULT:
                continue
            # print(f"run one time, serve {BATCH_SIZE} client")
    if agent_type == "learner":
        sample_mgr = SampleManager(yaml_agent.agent.agent_num, game_name)
        while True:
            list_samples = sample_mgr.recv_and_process()
            agent.learn(list_samples)
            # print("train one step")

def run_workflow(fn=None):
    if fn:
        fn(kaiwu_agent.logger, kaiwu_agent.monitor)
    else:
        kaiwu_agent.agent.protocol.protocol.workflow(kaiwu_agent.logger, kaiwu_agent.monitor)

@wrap_fn_2_process(daemon=False)
def subproc_run_workflow(fn=None):
    run_workflow(fn=fn)

@wrap_fn_2_process(daemon=False)
def subproc_run_arena(cls_agent, game_name, scene_name='default_scene'):
    run_arena(cls_agent, game_name, scene_name)

@wrap_fn_2_process(daemon=False)
def subproc_run_agent(cls_agent, agent_type, game_name=None, scene_name=None):
    run_agent(cls_agent, agent_type, game_name, scene_name)

def run_model_deliver_proxy():
    ModelDeliverManager.run_rps_proxy()

@wrap_fn_2_process(daemon=False)
def subproc_run_model_deliver_send():
    ModelDeliverManager.model_send()

@wrap_fn_2_process(daemon=False)
def subproc_run_model_deliver_recv():
    ModelDeliverManager.model_recv()


def init_agent_runtime(cls_agent, game_name, scene_name='default_scene'):
    # remote_sample打开时, 训练样本发送到mempool, 要求打开mempool, 开关mempool_mgr=true
    if tree_switch.check("remote_sample"):
        sample_mgr = SampleManager(yaml_agent.agent.agent_num, game_name)
        kaiwu_agent.agent.sample_mgr = sample_mgr
        # 单元测试用, 如果没有打开mempool_mgr, 则启动一个用于测试
        if not tree_switch.check("mempool_mgr"):
            run_mempool(game_name, scene_name)
    
    # 单元测试用, 如果需要远程预测但是没有打开mempool_mgr, 则启动一个和actor用于测试
    if tree_switch.check("remote_predict") and not tree_switch.check("remote_act_mgr"):
        run_rmt_act_proxy()
        subproc_run_agent(cls_agent, "actor", game_name)

    if tree_switch.check("remote_model"):
        model_mgr = ModelManager()
        kaiwu_agent.agent.model_mgr = model_mgr
