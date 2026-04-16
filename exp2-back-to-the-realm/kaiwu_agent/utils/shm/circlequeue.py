import time, copy, random, math
from multiprocessing import Array, Value, Process
import numpy as np
import ctypes
import copy
from kaiwu_agent.utils.shm.smarray import SMArray, counter_plus

class SMCircleQueue(SMArray):
    def __init__(self, dtype_item=ctypes.c_uint32, len_item=2, len_array=10) -> None:
        super().__init__(dtype_item, len_item, len_array)
        
        self.idx_next = Value('i', 0)
        self.len_queue = Value('i', 0)
    
    @counter_plus('write')
    def append(self, item):
        # 有可能存在append进程比item还多的情况，在把数据写入内存之前，其他进程不许动idx_next和len_queue
        # 另一方面，尽量在get_lock中少放代码，不可重入代码会让进程间相互等待
        nparray = self.get_sma_2nparray()
        with self.idx_next.get_lock():
            idx = self.idx_next.value
            self.idx_next.value = (self.idx_next.value + 1) % self.len_array
        
            with self.status_item[idx].get_lock():
                nparray[idx] = item

            with self.len_queue.get_lock():
                if self.len_queue.value < self.len_array:
                    self.len_queue.value += 1

    
    def get_random_item(self):
        error_count = 0
        while self.len_queue.value < int(self.len_array / 2):
            error_count += 1
            time.sleep(0.05)
            if error_count % 1000 == 0:
                print("[Debug] The sample is less than half the capacity {} {}".format(self.len_queue.value, self.len_array))

        with self.len_queue.get_lock():
            idx_random = random.randint(0, self.len_queue.value - 1) 
            return self.get_item_by_idx_2nparray(idx_random)



if __name__=='__main__':
    # sma = SMArray()
    sma = SMCircleQueue()

    def writer(i, sma):
        while True:
            time.sleep(0.00001)
            idx = random.randint(0, 9)
            data = np.array([idx * math.pow(10, i), idx * math.pow(10, i)])
            # sma.set_item_by_idx_2nparray(idx, data)
            sma.append(data)
            
    
    def reader(sma):
        wc = sma.get_write_speed()
        rc = sma.get_read_speed()
        while True:
            time.sleep(1)
            print([sma.get_item_by_idx_2nparray(i).tolist() for i in range(10)], next(wc), '   ', next(rc))

    for i in range(9):
        p = Process(target=writer, args=(i, sma, ))
        p.start()
    
    for i in range(9):
        p = Process(target=writer, args=(i, sma, ))
        p.start()
    
    p = Process(target=reader, args=(sma, ))
    p.start()
    

    p.join()
