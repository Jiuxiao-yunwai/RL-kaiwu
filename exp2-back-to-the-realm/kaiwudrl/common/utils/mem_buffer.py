#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file mem_buffer.py
# @brief
# @author kaiwu
# @date 2023-11-28


import random
import copy
import ctypes
from multiprocessing import Value, Array, Queue
import sys
import time
import numpy as np
from kaiwudrl.common.config.config_control import CONFIG


class MemBuffer(object):
    """
    random repeat get sample
    """

    def __init__(self, max_sample_num, sample_size, logger):
        self._maxlen = int(max_sample_num)
        self._sample_size = int(sample_size)
        # 存储float32
        self._c_data_type = ctypes.c_float
        self._data_type = np.float32
        self._data_queue = Array(self._c_data_type, max_sample_num * self._sample_size, lock=False)
        self._data_status = [Value(ctypes.c_bool, False, lock=True) for index in range(max_sample_num)]
        self.next_idx = Value("i", 0)
        self._len = Value("i", 0)
        self.recv_samples = Value("i", 0)
        self.start_time = time.time()
        self.start_sample_num = 0
        self.last_speed = 0

        self.logger = logger

    def __len__(self):
        length = self._len.value
        return length

    def append(self, data):
        with self.next_idx.get_lock():
            idx = self.next_idx.value
            self.next_idx.value = (self.next_idx.value + 1) % self._maxlen
        with self._data_status[idx].get_lock():
            nparray = np.frombuffer(self._data_queue, dtype=self._data_type)
            nparray = nparray.reshape(self._maxlen, self._sample_size)
            nparray[idx] = data

        with self._len.get_lock():
            if self._len.value < self._maxlen:
                self._len.value += 1

        with self.recv_samples.get_lock():
            self.recv_samples.value += 1

    def get_samples(self, N):
        """
        一次性返回N条记录
        """
        error_index = 0

        # 需要满足预加载的样本个数
        while self.__len__() < int(self._maxlen / CONFIG.preload_ratio):
            error_index += 1
            time.sleep(0.05)
            if error_index % 1000 == 0:
                self.logger.info("The sample is less than half the capacity {} {}".format(self.__len__(), self._maxlen))
        # 确保有足够的样本
        while self._len.value < N:
            time.sleep(1)
            self.logger.info("sample_num < N {}".format(self._len.value))

        # 随机选择N个索引
        indices = random.sample(range(self.__len__()), N)
        values = []

        for i in indices:
            if i < 0 or i > self._maxlen:
                self.logger.info("random index is illegal")

            with self._data_status[i].get_lock():
                nparray = np.frombuffer(self._data_queue, dtype=self._data_type)
                nparray = nparray.reshape(self._maxlen, self._sample_size)
                value = copy.deepcopy(nparray[i])
                values.append(value)

        return values

    def get_sample(self):
        """
        一次性返回1条记录
        """
        error_index = 0

        # 需要满足预加载的样本个数
        while self.__len__() < int(self._maxlen / CONFIG.preload_ratio):
            error_index += 1
            time.sleep(0.05)
            if error_index % 1000 == 0:
                self.logger.info("The sample is less than half the capacity {} {}".format(self.__len__(), self._maxlen))
        # 确保有足够的样本
        while self._len.value < 0:
            time.sleep(1)
            self.logger.info("sample_num < 0 {}".format(self._len.value))
        i = random.randint(0, self.__len__() - 1)
        if i < 0 or i > self._maxlen:
            self.logger.info("random index is illegal")

        with self._data_status[i].get_lock():
            nparray = np.frombuffer(self._data_queue, dtype=self._data_type)
            nparray = nparray.reshape(self._maxlen, self._sample_size)
            value = copy.deepcopy(nparray[i])
            return value

    def clear(self):
        with self._len.get_lock():
            self._len.value = 0
        with self.next_idx.get_lock():
            self.next_idx.value = 0
        with self.recv_samples.get_lock():
            self.recv_samples.value = 0

    def get_speed(self):
        total_sample = self.recv_samples.value
        if total_sample < 0:
            with self.recv_samples.get_lock():
                self.recv_samples.value = 0
            return self.last_speed, 0

        end_time = time.time()
        speed = float(total_sample - self.start_sample_num) / float(end_time - self.start_time)
        self.last_speed = speed
        self.start_sample_num = total_sample
        self.start_time = end_time

        # 返回当前速度和总的样本数目
        return speed, total_sample
