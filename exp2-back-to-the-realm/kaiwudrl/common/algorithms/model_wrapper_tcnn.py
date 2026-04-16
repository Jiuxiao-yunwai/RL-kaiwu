#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file model_wrapper_tcnn.py
# @brief
# @author kaiwu
# @date 2023-11-28


class ModelWrapperTcnn:
    """
    ModelWrapperTcnn类, actor和learner都会使用, 主要用于预测, 训练等
    """

    def __init__(self, model, logger, server=None) -> None:

        self.model = model
        self.logger = logger

        # 统计值
        self.train_count = 0
        self.predict_count = 0

        # 给业务设置下日志接口
        self.set_logger()

        # 因为tcnn需要用户确保保存最多多少model文件
        self.file_queue = []

    def should_stop(self):
        return self.sess.should_stop()

    def close(self):
        return self.sess.close()

    def before_train(self):
        pass

    def after_train(self):
        # 本次是否执行了更新model文件的操作
        has_model_file_changed = False

        self.train_count += 1

        return has_model_file_changed, -1

    def before_predict(self):
        pass

    def after_predict(self, batch_size):
        self.predict_count += batch_size

    # train 函数
    def train(self, extra_tensors=None):
        self.before_train()

        # 具体的训练流程
        values = None

        has_model_file_changed, model_file_id = self.after_train()

        return values, has_model_file_changed, model_file_id

    # predict函数
    def predict(self, extra_tensors, batch_size):
        self.before_predict()

        # 部分场景需要更新predict_count
        if hasattr(self.model, "update_predict_count"):
            self.model.update_predict_count(self.predict_count)

        # 具体的预测流程
        values = None

        self.after_predict(batch_size)

        return values

    @property
    def train_stat(self):
        return self.train_count

    @property
    def predict_stat(self):
        return self.predict_count

    @property
    def name(self):
        return "ModelWrapperTcnn"
