#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file zmq_replay_buffer.py
# @brief
# @author kaiwu
# @date 2023-11-28


import numpy as np
import ctypes
from multiprocessing import Value, Array, Queue
from kaiwudrl.common.config.config_control import CONFIG
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.utils.choose_deep_learning_frameworks import *
from kaiwudrl.common.replay_buffer.replay_buffer_base import ReplayBufferBase
from kaiwudrl.common.utils.mem_buffer import MemBuffer


class BatchManager(object):
    """
    因为每条数据是写入到了共享内存里, 但是为了批量的返回数据, 这里采用了类来处理
    """

    def __init__(self, batch_size, sample_size, process_num, logger):
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.process_num = process_num
        self.c_data_type = ctypes.c_float
        self.data_type = np.float32
        self.data = Array(self.c_data_type, process_num * 2 * sample_size * batch_size, lock=False)
        self.state = Array(ctypes.c_int, process_num * 2 + 1, lock=False)
        for index in range(len(self.state)):
            self.state[index] = 1
        self.last_get = Value("i", process_num * 2, lock=False)

        self.logger = logger

    def set_batch_sample(self, sample, batch_index):
        if not (isinstance(sample, np.ndarray) or sample.shape == (1, self.sample_size * self.batch_size)):
            self.logger.info(f"set_batch_sample error batch {sample.shape}")
            return False
        nparray = np.frombuffer(self.data, dtype=self.data_type)
        nparray = nparray.reshape(self.process_num * 2, self.batch_size, self.sample_size)
        nparray[batch_index] = sample
        return True

    def set_one_sample(self, sample, batch_index, sample_index):
        if not (isinstance(sample, np.ndarray) or sample.shape == (1, self.sample_size)):
            self.logger.info(f"set_one_sample error sample {sample.shape}")
            return False
        nparray = np.frombuffer(self.data, dtype=self.data_type)
        nparray = nparray.reshape(self.process_num * 2 * self.batch_size, self.sample_size)
        nparray[batch_index * self.batch_size + sample_index] = sample
        return True

    def get_batch_sample(self, batch_index):
        nparray = np.frombuffer(self.data, dtype=self.data_type)
        nparray = nparray.reshape(self.process_num * 2 * self.batch_size, self.sample_size)
        value = nparray[batch_index * self.batch_size : batch_index * self.batch_size + self.batch_size]
        return value

    def set_state(self, index):
        self.state[self.last_get.value] = 1
        self.last_get.value = index

    def clear(self):
        for index in range(len(self.state)):
            self.state[index] = 1
        self.last_get.value = 2 * self.process_num


class BatchProcess(object):
    def __init__(self, batch_size, sample_size, process_num, logger):
        self.batch_size = batch_size
        self.process_num = process_num

        # batch_idx有效
        self.batch_queue = Queue()
        self.free_queue = Queue()
        self.logger = logger

        self.batch_manager = BatchManager(
            batch_size=batch_size,
            sample_size=sample_size,
            process_num=process_num,
            logger=self.logger,
        )
        self.pids = []
        self.last_get_index = None

    def __process_run(self, process_index, get_sample_func, full_queue, free_queue):
        self.logger.info("[BatchProcess::__process_run] process_index:{} pid:{}".format(process_index, os.getpid()))
        while True:
            batch_index = free_queue.get()
            for sample_index in range(self.batch_size):
                sample = get_sample_func()
                self.batch_manager.set_one_sample(sample, batch_index, sample_index)
            full_queue.put(batch_index)

    def process(self, get_data_func):
        for batch_index in range(self.process_num * 2):
            self.free_queue.put(batch_index)
        for process_index in range(self.process_num):
            pid = Process(
                target=self.__process_run,
                args=(
                    process_index,
                    get_data_func,
                    self.batch_queue,
                    self.free_queue,
                ),
            )
            pid.daemon = True
            pid.start()
            self.pids.append(pid)

    def get_batch_data(self):
        batch_index = self.batch_queue.get()
        sample = self.batch_manager.get_batch_sample(batch_index)
        return batch_index, sample

    def put_free_data(self, batch_index):
        self.free_queue.put(batch_index)

    def exit(self):
        for pid in self.pids:
            pid.join()


class ZmqReplayBuffer(ReplayBufferBase):
    def __init__(self, data_spec, logger):

        # 参数定义
        capacity = CONFIG.replay_buffer_capacity

        self.batch_size = CONFIG.train_batch_size
        self.data_shapes = data_spec
        self.logger = logger

        sample_size = int(self.data_shapes[0].shape[0])

        # MemBuffer
        self.mem_buffer = MemBuffer(capacity, sample_size, self.logger)

        # BatchProcess
        self.batch_process = BatchProcess(
            self.batch_size,
            sample_size,
            CONFIG.batch_process_for_batch_manager,
            self.logger,
        )

        self.last_batch_index = -1

    def add_sample(self, datas):
        if not datas:
            return False

        for data in datas:
            self.mem_buffer.append(data)

        return True

    def init(self):

        # 从MemBuff里获取样本
        self.batch_process.process(self.mem_buffer.get_sample)

    def get_next_batch(self):
        """
        获取下一批的数据记录
        """
        batch_index, sample_buff = self.batch_process.get_batch_data()
        if self.last_batch_index >= 0:
            self.batch_process.put_free_data(self.last_batch_index)
        self.last_batch_index = batch_index

        return sample_buff

    def next(self):
        """
        获取下一条数据记录, 采用预先批处理
        """

        return torch.from_numpy(self.get_next_batch())

    def next_by_batch_size(self):
        """
        获取下一条数据记录, 采用一次性批处理方式
        """
        return [self.mem_buffer.get_samples(CONFIG.train_batch_size)]

    def next_by_for(self):
        """
        获取下一条数据记录, 采用for循环方式
        """
        batch_samples = []
        for _ in range(CONFIG.train_batch_size):
            sample = self.mem_buffer.get_sample()
            batch_samples.append(sample)

        # 增加数组是为了适配已经有的取data[0]的操作
        return [batch_samples]

    def get_insert_speed(self):
        speed, add_sample_count = self.mem_buffer.get_speed()
        return speed, add_sample_count

    def total_size(self):
        return CONFIG.replay_buffer_capacity
