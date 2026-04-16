
import time, copy
from telnetlib import SE
from multiprocessing import Array, Value, Process, Queue, Semaphore
import numpy as np
import ctypes
import copy
from kaiwu_agent.utils.shm.smarray import SMArray, counter_plus
        

class SMBatchMgr(SMArray):
    def __init__(self, dtype_item=ctypes.c_uint32, len_item=2, batch_size=4, slot_per_process=5, num_process=9) -> None:
        super().__init__(dtype_item, len_item, len_array=0)
        self.dtype_item = dtype_item
        self.len_item = len_item
        self.batch_size = batch_size
        self.num_process = num_process
        self.slot_per_process = slot_per_process
        self.array_item = Array(self.dtype_item, self.num_process * self.slot_per_process * self.batch_size * self.len_item, lock=False)
        self.status_item = [Semaphore(1) for _ in range(self.num_process * self.slot_per_process)]
        self.q_read = Queue(maxsize=self.num_process * self.slot_per_process + 1)

    @counter_plus('read')
    def get_one_batch(self, idx_batch):
        nparray = np.frombuffer(self.array_item, dtype=self.dtype_item)
        nparray = nparray.reshape(self.num_process * self.slot_per_process, self.batch_size, self.len_item)
        value = copy.deepcopy(nparray[idx_batch, :, :])
        # self.counter_plus('read')
        return value

    @counter_plus('write')
    def set_one_batch(self, idx_batch, data_batch):
        nparray = np.frombuffer(self.array_item, dtype=self.dtype_item)
        nparray = nparray.reshape(self.num_process * self.slot_per_process, self.batch_size, self.len_item)
        nparray[idx_batch] = data_batch
        # self.counter_plus('write')
    
    @counter_plus('write')
    def set_one_item(self, idx_item, data_item):
        nparray = np.frombuffer(self.array_item, dtype=self.dtype_item)
        nparray = nparray.reshape(self.num_process * self.slot_per_process * self.batch_size, self.len_item)
        nparray[idx_item] = data_item
        # self.counter_plus('write')


class BatchReader(Process):
    def __init__(self, id, bmgr) -> None:
        super().__init__()
        self.bmgr = bmgr
        self.id = id

    def run(self):
        while True:
            time.sleep(0.00001)
            idx_batch = self.bmgr.q_read.get()
            data_batch = self.bmgr.get_one_batch(idx_batch)
            if data_batch[0, 0] != idx_batch:
                print('wrongggggggggggggggggggg')
            self.bmgr.status_item[idx_batch].release()

class BatchWriter(Process):
    def __init__(self, id, bmgr) -> None:
        super().__init__()
        self.id = id
        self.bmgr = bmgr
        
    def run(self):
        wc = self.bmgr.get_write_speed()
        rc = self.bmgr.get_read_speed()
        count = 0

        slot_per_process = self.bmgr.slot_per_process
        while True:
            for i in range(slot_per_process):
                time.sleep(0.00001)
                idx_batch = self.id * slot_per_process + i
                self.bmgr.status_item[idx_batch].acquire(block=False)
                data_batch = np.ones([self.bmgr.batch_size, self.bmgr.len_item]) * idx_batch
                self.bmgr.set_one_batch(idx_batch, data_batch)
                self.bmgr.q_read.put(idx_batch)

            count += 1
            if count % 1000 == 0 and self.id == 0:
                print(next(wc), '   ', next(rc), '   ')
            

if __name__=='__main__':
    bmgr = SMBatchMgr()

    for i in range(9):
        p = BatchWriter(i, bmgr)
        p.start()
     
    for i in range(9):
        reader = BatchReader(i, bmgr)
        reader.start()

    reader.join()


