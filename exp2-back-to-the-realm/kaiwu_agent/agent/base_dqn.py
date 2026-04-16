
import kaiwu_agent
import torch
from collections import deque
import random
import os
from kaiwu_agent.conf import tree_switch, yaml_metagent
from kaiwu_agent.agent.base_agent import learn_wrapper, BaseAgent, check_hasattr, save_model_wrapper, load_model_wrapper

class Agent(BaseAgent):

    @check_hasattr(["model", "_gamma", "optim", "train_step"])
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        super().__init__(agent_type, device, logger, monitor)
        self.agent_type = agent_type
        self.device = device
        self.logger = logger if logger else kaiwu_agent.logger
        self.monitor = monitor if monitor else kaiwu_agent.monitor
        self.memory = deque(maxlen=200)

    @learn_wrapper
    def learn(self, list_sample_data):
        t_data = list_sample_data

        obs = [frame.obs for frame in t_data]
        action = torch.LongTensor([int(frame.act) for frame in t_data]).view(-1,1).long().to(self.model.device)
        ret = torch.tensor([frame.ret for frame in t_data], device=self.model.device)
        _obs = [frame._obs for frame in t_data]
        not_done = torch.tensor([0 if frame.done==1 else 1 for frame in t_data], device=self.model.device)

        model = getattr(self, "model")
        model.eval()
        with torch.no_grad():
            q, h = model(_obs, state=None)
            q_max = q.max(dim=1).values.detach().cpu()

        target_q = ret + self._gamma * q_max * not_done

        self.optim.zero_grad()

        model = getattr(self, "model")
        model.train()
        logits, h = model(obs, state=None)

        loss = torch.square(target_q - logits.gather(1, action).view(-1)).sum()
        loss.backward()
        self.optim.step()

        self.train_step += 1

        if self.train_step % 100 == 0:
            self.save_model(id=str(self.train_step))

    @save_model_wrapper
    def save_model(self, path=None, id='1'):
        # To save the model, it can consist of multiple files, and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id='1'):
        # When loading the model, you can load multiple files, and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path,
                                   map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))
        self.logger.info(f"load model {model_file_path} successfully")
