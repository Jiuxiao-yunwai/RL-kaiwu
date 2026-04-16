import kaiwu_agent
from kaiwu_agent.agent.base_agent import BaseAgent
import numpy as np
import os
from kaiwu_agent.gorge_walk.utils import get_logger
from kaiwu_agent.conf import yaml_metagent
from kaiwu_agent.agent.base_agent import learn_wrapper, save_model_wrapper, load_model_wrapper, check_hasattr


class Agent(BaseAgent):
    @check_hasattr(["Q", "gamma", "learning_rate"])
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        super().__init__(agent_type, device, logger, monitor)
        self.agent_type = agent_type
        self.device = device
        self.logger = logger if logger else kaiwu_agent.logger  
        self.monitor = monitor if monitor else kaiwu_agent.monitor
    
    @learn_wrapper
    def learn(self, list_sample_data):
        """
        Update the Q-table with the given game data: \n
            - list_sample: each sampple is (state, action, reward, next_state, next_action) \n
        Using the following formula to update q value: \n
            - Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * Q(s',a') - Q(s,a)] \n
        If next_state is the end, then next_action is -1
        """
        # TODO: 这里是按照单个 sample 单步更新, 实际上可以按照 batch 更新
        sample = list_sample_data[0]
        state, action, reward = sample.state, sample.action, sample.reward
        next_state, next_action = sample.next_state, sample.next_action
        
        if next_action == -1:
            delta = reward - self.Q[state, action]
        else:
            delta = reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action]
            
        self.Q[state, action] += self.learning_rate * delta
        
        return

    @save_model_wrapper
    def save_model(self, path=None, id='1'):
        path = path if path else yaml_metagent.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
            self.logger.info(f"Create directory: {path}")
            
        np.save(f"{path}/model-{str(id)}", self.Q)
        self.save_checkpoint(path, id)
        os.system(f"touch {path}/model-{str(id)}.done")
        self.logger.info(f"Save model {path}/model-{str(id)}.npy successfully")
    
    @load_model_wrapper
    def load_model(self, path=None, id='1'):
        path = path if path else yaml_metagent.model_load_path
        if not os.path.exists(f"{path}/model-{str(id)}.done"):
            self.logger.error(f"File {path}/model-{str(id)}.done not found")
            exit(1)
        try:
            self.Q = np.load(f"{path}/model-{str(id)}.npy")
        except FileNotFoundError:
            self.logger.info(f"File {path}/model-{str(id)}.npy not found")
            exit(1)
            
    def save_checkpoint(self, path, id):
        file_path = f"{path}/checkpoint"
        model_checkpoint = "model_checkpoint_path: " + f"{path}/model-{id}.npy"
        all_model_checkpoint = f"all_model_checkpoint_path: {path}/model-{id}.npy"
        
        with open(file_path, 'w') as file:
            file.write(model_checkpoint + '\n')
            file.write(all_model_checkpoint + '\n')