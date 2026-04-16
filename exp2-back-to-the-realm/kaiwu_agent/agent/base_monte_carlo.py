import kaiwu_agent
from kaiwu_agent.agent.base_agent import BaseAgent
import numpy as np
from kaiwu_agent.gorge_walk.utils import get_logger
from kaiwu_agent.conf import yaml_metagent
import os
from kaiwu_agent.agent.base_agent import learn_wrapper, save_model_wrapper, load_model_wrapper, check_hasattr


class Agent(BaseAgent):
    @check_hasattr(["Q", "visit", "policy", "gamma", "state_size"])
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        super().__init__(agent_type, device, logger, monitor)
        self.agent_type = agent_type
        self.device = device
        self.logger = logger if logger else kaiwu_agent.logger  
        self.monitor = monitor if monitor else kaiwu_agent.monitor
    
    @learn_wrapper
    def learn(self, list_sample_data):
        """
        Calculate the optimal policy using Monte Carlo Control - fist visit \n
            - list_sample is a list a sample: (state, action, reward) \n
        Return is calculated using the following formula: \n
            - G = R(t+1) + gamma * R(t+2) + ... + gamma^(T-t-1) * R(T)
        """
        # Initialization
        G, state_action_return = 0, []
        
        # Calculate the return for each state-action pair
        for sample in reversed(list_sample_data[:-1]):
            state_action_return.append((sample.state, sample.action, G))
            G = self.gamma * G + sample.reward
            state_action_return.reverse()
        
        # Update the Q-table
        seen_state_action = set()
        for state, action, G in state_action_return:
            if (state, action) not in seen_state_action:
                self.visit[state][action] += 1
                
                # calculate incremental mean
                self.Q[state, action] = self.Q[state, action] + (G - self.Q[state, action]) / self.visit[state, action]
                seen_state_action.add((state, action))

        # Update policy
        for state in range(self.state_size):
            best_action = np.argmax(self.Q[state])
            self.policy[state] = best_action
        
        return

    @save_model_wrapper
    def save_model(self, path=None, id='1'):
        path = path if path else yaml_metagent.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
            self.logger.info(f"Create directory: {path}")
            
        np.save(f"{path}/model-{str(id)}", self.policy)
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
            self.policy = np.load(f"{path}/model-{str(id)}.npy")
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