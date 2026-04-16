#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file reverb_dataset_v1.py
# @brief
# @author kaiwu
# @date 2023-11-28


from collections import deque
import torch
import numpy as np
import reverb
from kaiwudrl.common.config.config_control import CONFIG
import threading
import time


class ReverbDataset(torch.utils.data.IterableDataset):
    """
    主动采用reverb_client从reverb_server读取数据
    """

    def __init__(self, logger):
        super().__init__()
        self._table_names = ["{}_{}".format(CONFIG.reverb_table_name, i) for i in range(int(CONFIG.reverb_table_size))]
        self.buffer = deque()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger

        # 双缓冲队列（线程安全）
        self.buffer_lock = threading.Lock()
        # 主缓冲
        self.active_buffer = deque(maxlen=CONFIG.train_batch_size * 4)
        # 后台缓冲
        self.backup_buffer = deque(maxlen=CONFIG.train_batch_size * 4)

        # 线程停止信号
        self._stop_event = threading.Event()

        self.client = None

        # 后台填充线程
        self._fill_thread = None

    def start_background_filler(self):
        """启动后台填充线程"""
        if self._fill_thread is None:
            self._stop_event.clear()

            self.client = reverb.Client(f"localhost:{CONFIG.reverb_svr_port}")
            self._fill_thread = threading.Thread(target=self._fill_buffers_loop, daemon=True)
            self._fill_thread.start()
            self.logger.info(
                f"start_background_filler success, reverb.Client connect at localhost:{CONFIG.reverb_svr_port}"
            )

    def _fill_buffers_loop(self):
        """后台线程持续填充缓冲区"""

        while not self._stop_event.is_set():
            # 填充备用缓冲区
            self._fill_buffer(self.backup_buffer)

            # 交换缓冲区（原子操作）
            with self.buffer_lock:
                self.active_buffer, self.backup_buffer = self.backup_buffer, self.active_buffer

            # 避免高频切换
            time.sleep(CONFIG.idle_sleep_second)

    def _fill_buffer(self, target_buffer):
        """填充指定缓冲区至目标大小"""
        while len(target_buffer) < target_buffer.maxlen and not self._stop_event.is_set():
            try:
                data = self.client.sample(
                    table=self._table_names[0],
                    num_samples=min(CONFIG.train_batch_size, target_buffer.maxlen - len(target_buffer)),
                )
                if data:
                    target_buffer.extend(data)
                else:
                    time.sleep(CONFIG.idle_sleep_second)
            except Exception as e:
                self.logger.error(f"后台填充线程错误: {str(e)}")
                time.sleep(CONFIG.idle_sleep_second)

    def __iter__(self):

        # 启动后台线程从reverb读取数据
        self.start_background_filler()

        while True:
            # 从主缓冲取数据
            with self.buffer_lock:
                if len(self.active_buffer) >= CONFIG.train_batch_size:
                    batch = [self.active_buffer.popleft() for _ in range(CONFIG.train_batch_size)]
                else:
                    batch = None

            if batch is None:
                time.sleep(CONFIG.idle_sleep_second)
                continue

            # 转换为 Tensor
            processed_batch = [self._process_element(sample[0].data[0]) for sample in batch]
            batch_tensor = torch.stack(processed_batch).to(self.device)
            yield [batch_tensor]

    def _process_element(self, element):
        """统一数据处理逻辑"""
        if isinstance(element, torch.Tensor):
            return element.detach().cpu()
        elif isinstance(element, np.ndarray):
            return torch.from_numpy(element).float()
        else:
            raise TypeError(f"不支持的数据类型: {type(element)}")

    def __del__(self):
        self._stop_event.set()
        if self._fill_thread is not None:
            self._fill_thread.join(timeout=5)

    def get_metrics(self):
        """获取性能指标"""
        return {"buffer_utilization": f"{len(self.active_buffer)}/{self.active_buffer.maxlen}"}
