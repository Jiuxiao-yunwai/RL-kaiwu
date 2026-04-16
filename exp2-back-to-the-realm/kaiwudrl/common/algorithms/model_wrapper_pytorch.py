#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file model_wrapper_pytorch.py
# @brief
# @author kaiwu
# @date 2023-11-28


import os
import numpy as np
import glob
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.torch_utils import *


class ModelWrapperPytorch:
    """
    ModelWrapperPytorch类, actor和learner都会使用, 主要用于预测, 训练等
    """

    def __init__(self, model, logger, server=None) -> None:

        self.model = model
        self.logger = logger

        # 统计值
        self.train_count = 0
        self.predict_count = 0
        self.save_model_count = 0

        # 主learner
        self.is_chief = CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER

        # 给业务设置下日志接口
        self.set_logger()

        # 因为pytorch需要用户确保保存最多多少model文件
        self.file_queue = []

    def should_stop(self):
        return self.model.should_stop()

    def set_logger(self):
        if hasattr(self.model, "set_logger"):
            self.model.set_logger(self.logger)

    def close(self):
        if hasattr(self.model, "stop"):
            return self.model.stop()

    def before_train(self):
        pass

    def add_file_to_queue(self):
        id = (self.save_model_count - 1) * CONFIG.dump_model_freq

        # 采用模糊匹配的方法来操作
        model_file_names = glob.glob(f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}/model.ckpt-{id}.*")
        if not model_file_names:
            return

        self.file_queue.append(model_file_names)
        if len(self.file_queue) >= CONFIG.max_save_model_file_count:
            model_file_names_to_delete = self.file_queue.pop(0)
            if model_file_names_to_delete:
                for to_model_file_name in model_file_names_to_delete:
                    if os.path.exists(to_model_file_name):
                        os.remove(to_model_file_name)

    def after_train(self):

        # 本次是否执行了更新model文件的操作
        has_model_file_changed = False
        self.train_count += 1
        if self.train_count % CONFIG.dump_model_freq == 0:
            self.save_param()
            has_model_file_changed = True

            # 放入队列控制占用大小以免磁盘无限增加被驱逐
            self.add_file_to_queue()

        return (
            has_model_file_changed,
            (self.save_model_count - 1) * CONFIG.dump_model_freq,
        )

    def save_param(self):
        self.model.save_param(id=self.save_model_count * CONFIG.dump_model_freq)
        self.save_model_count += 1

    def before_predict(self, predict_data):
        return isinstance(predict_data, dict)

    def after_predict(self, batch_size):
        self.predict_count += batch_size

    # train 函数
    def train(self, extra_tensors=None):
        self.before_train()

        # 具体的训练流程
        data = self.get_data_from_reverb()
        if not data:
            return None, False, -1

        values = self.model.learn(data)

        # 返回是否更新了model文件, 更新的model文件的ID
        has_model_file_changed, model_file_id = self.after_train()

        return values, has_model_file_changed, model_file_id

    # predict函数
    def predict(self, predict_data, batch_size):

        # 具体的预测流程
        values = None
        if self.before_predict(predict_data):
            # 部分场景需要更新predict_count
            if hasattr(self.model, "update_predict_count"):
                self.model.update_predict_count(self.predict_count)

            if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_TRAIN:
                values = self.model.predict(predict_data, types="prob")
            elif CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EXAM:
                types = "max" if np.random.rand(1) >= 0.1 else "prob"
                values = self.model.predict(predict_data, types=types)
            else:
                raise ValueError

            self.after_predict(batch_size)

        return values

    def get_global_step(self):
        return self.train_count

    @property
    def train_stat(self):
        return self.train_count

    @property
    def predict_stat(self):
        return self.predict_count

    @property
    def name(self):
        return "ModelWrapperPytorch"

    @property
    def tf_sess(self):
        return self.sess

    # 直接调用业务类的load_last_new_model
    def load_last_new_model(self, models_path):
        if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_TRAIN:
            return self.model.load_last_new_model(models_path)
        elif CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EVAL or CONFIG.run_mode == KaiwuDRLDefine.RUN_MODEL_EXAM:
            self.model.load_specific_model(CONFIG.eval_model_dir)
            self.logger.info(f"eval mode predict load_specific_model from {CONFIG.eval_model_dir} success")

        else:
            pass

    def preload_model_file(self, preload_model_file, id):
        """
        预加载模型文件, 直接调用业务类, 步骤如下:
        1. 删除引擎文件目录下的类似/data/ckpt/gorge_walk_v2_dqn/下的model.ckpt开头的文件
        2. 修改checkpoint文件内容类似/data/ckpt/gorge_walk_v2_dqn/checkpoint
        3. 保存最新的引擎文件
        4. 修改以后计数保存的变量值self.save_model_count
        """
        if not preload_model_file:
            return

        model_path = f"{CONFIG.restore_dir}/{CONFIG.app}_{CONFIG.algo}"
        file_pattern = "model.ckpt*"
        file_paths = glob.glob(os.path.join(model_path, file_pattern))
        for file_path in file_paths:
            os.remove(file_path)

        checkpoint_file = f"{model_path}/checkpoint"
        with open(checkpoint_file, "w") as f:
            # 将文件截断为0字节
            f.truncate(0)
            # 写入checkpoints list\n字符串
            f.writelines([f"checkpoints list\n"])

        self.model.load_specific_model(preload_model_file)

        self.model.save_param(id=id)

        # 按照整数来计数
        self.save_model_count = int(id / CONFIG.dump_model_freq) + 1

    def set_dataset(self, replay_buffer_wrapper):
        self.replay_buffer_wrapper = replay_buffer_wrapper

    def get_data_from_reverb(self):
        # 采用pytorch方法从reverb获取数据
        return self.replay_buffer_wrapper.dataset_from_generator_by_pytorch()
