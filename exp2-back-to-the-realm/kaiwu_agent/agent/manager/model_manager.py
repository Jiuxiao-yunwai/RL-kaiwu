import glob
import os
from kaiwu_agent.conf import yaml_modelpool

class ModelManager():
    def __init__(self):
        self.base_path = yaml_modelpool.RECV_MODEL_PATH # '/data/projects/Metagent/recv_model/model'
        self.current_model_name = None
        self.__update_model_file_list()

    def update_model(self, agent):
        tmp = self.__try_find_newest_model()
        if tmp:
            agent.load_model(path=tmp[0], id=tmp[1])
            return True
        return False

    def __try_find_newest_model(self):
        self.__update_model_file_list()
        if len(self.list_model_file_name) == 0:
            return None
        if self.current_model_name == self.list_model_file_name[-1]:
            return None

        self.current_model_name = self.list_model_file_name[-1]
        return self.__parse_current_model_name()
    
    def __parse_current_model_name(self):
        '''
            将'/data/projects/Metagent/recv_model/model/model.ckpt-40.pkl' 转换成
            ('/data/projects/Metagent/recv_model/model', '40')
        '''
        # 分割文件路径以提取文件名
        file_name = os.path.basename(self.current_model_name)
        # 从文件名中删除基本路径和文件扩展名
        model_id = file_name.replace('.pkl', '')
        # 通过'-'分割model_id以获取所需的输出
        model_id = model_id.split('-')[-1]
        return self.base_path, model_id
    
    def __update_model_file_list(self):
        self.list_model_file_name = sorted(glob.glob(self.base_path + "/*.pkl"), key=lambda x: os.path.getmtime(x))


