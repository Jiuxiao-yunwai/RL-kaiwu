
import time, copy, random, math
from multiprocessing import Array, Value, Process
import numpy as np
import ctypes
import copy
        
def counter_plus(name):
    def decorator(func):
        def wrapper(self, *args, **kws):
            result = func(self, *args, **kws)
            # 不是严格的计数，所以可以放在业务with的外面
            if name == 'read':
                with self.counter_read.get_lock():
                    self.counter_read.value += 1
            elif name == 'write':
                with self.counter_write.get_lock():
                    self.counter_write.value += 1
            else:
                print('wrong')
            return result
        return wrapper
    return decorator

class SMArray(object):
    def __init__(self, dtype_item=ctypes.c_uint32, len_item=2, len_array=10) -> None:
        self.dtype_item = dtype_item
        # self.dtype_item = ctypes.c_float   # match to np.float32
        # self.dtype_item = ctypes.c_unit16  # match to np.int16
        self.len_item = len_item
        self.len_array = len_array
        self.array_item = Array(self.dtype_item, self.len_item * self.len_array, lock=False)
        self.status_item = [Value(ctypes.c_bool, False, lock=True) for _ in range(self.len_array)]
        
        self.counter_write = Value('i', 0)
        self.counter_read = Value('i', 0)

        self.time_start = time.time()

    def get_sma_2nparray(self):
        # 获取引用，不用获取锁
        nparray = np.frombuffer(self.array_item, dtype=self.dtype_item)
        nparray = nparray.reshape(self.len_array, self.len_item)
        return nparray
    
    @counter_plus('read')
    def get_item_by_idx_2nparray(self, idx):
        nparray = self.get_sma_2nparray()   
        with self.status_item[idx].get_lock():
            item = copy.deepcopy(nparray[idx])
        # self.counter_plus('read')
        return item

    @counter_plus('write')
    def set_item_by_idx_2nparray(self, idx, item):
        nparray = self.get_sma_2nparray()   
        with self.status_item[idx].get_lock():
            nparray[idx] = item
        # self.counter_plus('write')
    

    # def counter_plus(self, name):
        # # 不是严格的计数，所以可以放在业务with的外面
        # if name == 'read':
            # with self.counter_read.get_lock():
                # self.counter_read.value += 1
        # elif name == 'write':
            # with self.counter_write.get_lock():
                # self.counter_write.value += 1
        # else:
            # print('wrong')
    
    def get_write_speed(self):
        time_last = self.time_start
        count_last = 0
        while True:
            with self.counter_write.get_lock():
                count_end = self.counter_write.value
            time_end = time.time()        
            yield float(count_end - count_last) / float(time_end - time_last)
            
            count_last = count_end
            time_last = time_end
    
    def get_read_speed(self):
        time_last = self.time_start
        count_last = 0
        while True:
            with self.counter_read.get_lock():
                count_end =  self.counter_read.value
            time_end = time.time()
            yield float(count_end - count_last) / float(time_end - time_last)
            
            count_last = count_end
            time_last = time_end
    

if __name__=='__main__':
    sma = SMArray()

    def writer(i, sma):
        while True:
            time.sleep(0.00001)
            idx = random.randint(0, 9)
            data = np.array([idx * math.pow(10, i), idx * math.pow(10, i)])
            sma.set_item_by_idx_2nparray(idx, data)
            
    
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

