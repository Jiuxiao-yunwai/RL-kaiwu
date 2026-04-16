import kaiwu_agent
from kaiwu_agent.conf import tree_switch
from kaiwu_agent.agent.manager.remote_act_manager import RemoteActManager
import random

def learn_wrapper(func):
    def wrapper(agent, g_data):
        # 如果agent是learner则包装不生效
        if agent.agent_type == "learner":
            return func(agent, g_data)
        # 如何agent是player则看情况是否使用sample_mgr, model_mgr, learn
        if tree_switch.check("remote_sample"):
            kaiwu_agent.agent.sample_mgr.process_and_send(g_data)
        if tree_switch.check("remote_model"):
            kaiwu_agent.agent.model_mgr.update_model(agent)
        if not tree_switch.check("remote_learn"):
            # 如agent是player且remote_sample是false, 才使用本地样本池, 或直接使用样本
            if False: # not tree_switch.check("on_policy"):
                for item in g_data:
                    agent.memory.append(item) 
                if len(agent.memory) < agent.memory.maxlen // 2:
                    return
                g_data = random.sample(agent.memory, 512)
            return func(agent, g_data)
        return None
    return wrapper

def predict_wrapper(func):
    def wrapper(agent, list_obs_data, *args, **kargs):
        # 如果agent是actor则包装不生效
        if agent.agent_type == "actor":
            return func(agent, list_obs_data, *args, **kargs)
        # 如何agent是player则看情况调用predict或者远程访问
        if not tree_switch.check("remote_predict"):
            return func(agent, list_obs_data, *args, **kargs)
        else:
            return agent.remote_act_mgr.get_remote_action(list_obs_data)
    return wrapper

def exploit_wrapper(func):
    def wrapper(agent, *args, **kargs):
        return func(agent, *args, **kargs)
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
        self.agent_type = agent_type
        self.device = device
        self.logger = logger if logger else kaiwu_agent.logger
        self.monitor = monitor if monitor else kaiwu_agent.monitor

        if tree_switch.check("remote_predict"):
            self.remote_act_mgr = RemoteActManager()

        if tree_switch.check("receive_model"):
            self.can_update_remote_model = True

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

    @save_model_wrapper
    def save_model(self, path, id='1'):
        raise NotImplementedError

    @load_model_wrapper
    def load_model(self, path, id='1'):
        raise NotImplementedError

    def set_can_update_remote_model(self, value):
        assert isinstance(value, bool)
        self.can_update_remote_model = value

    def remote_predict(self, list_b_obs_list, model_mgr):
        if tree_switch.check("receive_model") and self.can_update_remote_model:
            self.can_update_remote_model = True
            flag = model_mgr.update_model(self)
            if flag:
                print("remote predictor load new model")
        list_list_obs_data = [self.remote_act_mgr.parse_obs_data(b_obs_list) for b_obs_list in list_b_obs_list]
        len_list_obs_data = [len(i) for i in list_list_obs_data]
        stack_obs_data = list()
        for list_obs_data in list_list_obs_data:
            stack_obs_data += list_obs_data
        stack_act_data = self.predict(stack_obs_data)
        list_list_act_data, start_index = list(), 0
        for length in len_list_obs_data:
            list_list_act_data.append(stack_act_data[start_index:start_index+length])
            start_index += length
        return [self.remote_act_mgr.ret_remote_action(list_act_data) for list_act_data in list_list_act_data]

