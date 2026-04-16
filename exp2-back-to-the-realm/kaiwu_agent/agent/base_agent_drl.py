import torch
from framework.common.config.config_control import  CONFIG
torch.set_num_threads(int(CONFIG.torch_num_threads))
import re
import os
import numpy as np
from framework.common.utils.common_func import get_first_last_line_from_file
from framework.common.utils.kaiwudrl_define import KaiwuDRLDefine
from conf.config import Config

def learn_wrapper(func):
    def wrapper(agent, g_data):
        return func(agent, g_data)
    return wrapper

def predict_wrapper(func):
    def wrapper(agent, list_obs_data, *args, **kargs):
        return func(agent, list_obs_data, *args, **kargs)
    return wrapper

def exploit_wrapper(func):
    def wrapper(agent, list_obs_data, *args, **kargs):
        return func(agent, list_obs_data, *args, **kargs)
    return wrapper

def save_model_wrapper(func):
    def wrapper(agent, *args, **kargs):
        return func(agent, *args, **kargs)
    return wrapper

def load_model_wrapper(func):
    def wrapper(agent, *args, **kargs):
        return func(agent, *args, **kargs)
    return wrapper

class BaseAgent:
    """
    Agent 的基类，所有的 Agent 都应该继承自这个类"""

    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        self.file_queue = []

    @learn_wrapper
    def learn(self, list_sample_data) -> dict:
        """
        用于学习的函数，接受一个 SampleData 的列表
        - dqn/ppo: 每个 game_data 是一个 episode 的数据
        - dynamic_programming: 每个 game_data 是一个 step 的数据
        """
        raise NotImplementedError

    @predict_wrapper
    def predict(self, list_obs_data: list) -> list:
        """
        用于获取动作的函数，接受一个 ObsData 的列表, 返回一个动作列表
        """
        raise NotImplementedError

    @exploit_wrapper
    def exploit(self, list_obs_data: list) -> list:
        """
        用于获取动作的函数，接受一个 ObsData 的列表, 返回一个动作列表
        """
        raise NotImplementedError

    def get_action(self, *kargs, **kwargs):
        return self.predict(*kargs, **kwargs)

    @save_model_wrapper
    def save_model(self, path, id='1'):
        pass
        # raise NotImplementedError

    @load_model_wrapper
    def load_model(self, path, id='1'):
        pass
        # raise NotImplementedError

    def should_stop(self):
        return False

    def stop(self):
        return True

    def get_last_new_model_path(self, models_path):
        """
            Description: 根据传入的模型路径，载入最新模型
        """
        checkpoint_id = -1
        checkpoint_file = f'{models_path}/{KaiwuDRLDefine.CHECK_POINT_FILE}'

        last_line = None
        try:
            _, last_line = get_first_last_line_from_file(checkpoint_file)
        except Exception as e:
            pass

        if not last_line or (KaiwuDRLDefine.KAIWU_MODEL_CKPT not in last_line):
            return checkpoint_id

        # 格式形如all_model_checkpoint_paths: "/data/ckpt//sgame_ppo/model.ckpt-4841", 注意不要采用正则匹配, 因为app可能会有具体的数字
        checkpoint_id = re.search(r'(?<={}-)\d+'.format(KaiwuDRLDefine.KAIWU_MODEL_CKPT), last_line)
        if not checkpoint_id:
            return checkpoint_id
        checkpoint_id = int(checkpoint_id.group())

        return checkpoint_id

    def set_dataset(self, dataset):
        self.dataset = dataset

    '''
    按照不同的算法进行保存模型的操作
    1. sarsa, 保存Q内容
    2. q_learning, 保存Q内容
    3. monte_carlo, 保存Q内容
    4. dqn, 保存pytorch的模型内容
    5. ppo, 保存pytorch的模型内容
    6. dynamic_programming, 不需要learner/actor之间传递

    保存模型后的操作:
    1. checkpoint文件里增加checkpoints, 原则上是第一次使用时增加即可
    2. checkpoint文件里增加all_model_checkpoint_paths
    3. 保留最近N个模型的逻辑
    '''
    def save_param_detail(self, path, id):
        """
            Description: 保存模型的方法
            ----------

            Parameters
            ----------
            path: str
                保存模型的路径
        """

        save_file_path = f"{str(path)}/model.ckpt-{str(id)}.pkl"

        file_exist_flag = os.path.exists(f"{str(path)}/checkpoint")
        with open(f"{str(path)}/checkpoint", mode='a') as fp:
            if not file_exist_flag:
                fp.writelines([
                    f"checkpoints list\n"
                ])
            fp.writelines([
                f"all_model_checkpoint_paths: \"{str(path)}/model.ckpt-{str(id)}\"\n"
            ])
        self.add_file_to_queue(save_file_path)

    def add_file_to_queue(self, file_path):
        self.file_queue.append(file_path)
        if len(self.file_queue) > Config.MAX_FILE_KEEP_CNT:
            to_delete_file = self.file_queue.pop(0)
            if os.path.exists(to_delete_file):
                os.remove(to_delete_file)

    '''
    将Q-table的内容写入文件
    '''
    def write_Q_to_file(self, filename, q_table):
        with open(filename, 'w') as file:
            np.savetxt(file, q_table)

    '''
    从文件里读取Q-table的内容
    '''
    def read_Q_from_file(self, filename):
        with open(filename, 'r') as file:
            q_table = np.loadtxt(file)

        return q_table

