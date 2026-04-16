import kaiwu_agent
from kaiwu_agent.agent.protocol.mempoolproto import MemPoolProtocol
from kaiwu_agent.utils.shm.circlequeue import SMCircleQueue
from kaiwu_agent.utils.shm.batchmgr import SMBatchMgr
from kaiwu_agent.zlib.rrrproxy import ReliaRRProxy
from kaiwu_agent.utils.common_func import wrap_fn_2_process
from kaiwu_agent.conf import tree_switch
from kaiwu_agent.conf import yaml_rmt_agent
from multiprocessing import Process
from kaiwu_agent.zlib.rrrworker import worker_rrrproxy
import ctypes
from kaiwu_agent.zlib.zhelper import WORKER_FAULT_RESULT, SAMPLE_REQ
from kaiwu_agent.conf import yaml_mempool

LEN_SAMPLE = yaml_mempool.LEN_SAMPLE 
BATCH_SIZE = yaml_mempool.BATCH_SIZE 
LEN_MEMPOOL = yaml_mempool.LEN_MEMPOOL 
SLOT_PER_PROCESS = yaml_mempool.SLOT_PER_PROCESS 
NUM_PROCESS = yaml_mempool.NUM_PROCESS 
NUM_WORKER = yaml_mempool.NUM_WORKER 

class MempoolManager():
    def __init__(self, game_name, scene_name='default_scene', logger=None, monitor=None):
        self.game_name = game_name
        self.scene_name = scene_name
        self.mempool = SMCircleQueue(dtype_item=ctypes.c_float, len_item=LEN_SAMPLE, len_array=LEN_MEMPOOL)
        self.batch_mgr = SMBatchMgr(dtype_item=ctypes.c_float, len_item=LEN_SAMPLE, batch_size=BATCH_SIZE, slot_per_process=SLOT_PER_PROCESS, num_process=NUM_PROCESS)

    def run(self):
        if tree_switch.check("remote_sample"):
            self.__run_mempool_ftend()
            for id in range(NUM_PROCESS):
                batch_writer(id, self.batch_mgr, self.mempool)
            # 用一个arena进行测试
            # from kaiwu_agent.main import subproc_run_arena
            # subproc_run_arena("dqn", self.game_name)

        if tree_switch.check("receive_sample"):
            self.__run_mempool_bkend()
            # 用一个learner进行测试
            # local_run_learner(self.game_name)

    def __run_mempool_ftend(self):
        # 启动mempool前端     
        rrrproxy = ReliaRRProxy(yaml_rmt_agent.mempool_fe.server.url_ftend, yaml_rmt_agent.mempool_fe.server.url_bkend)
        rrrproxy.run()
    
    def __run_mempool_bkend(self):
        # 启动mempool后端     
        rrrproxy = ReliaRRProxy(yaml_rmt_agent.mempool_be.server.url_ftend, yaml_rmt_agent.mempool_be.server.url_bkend)
        rrrproxy.run()
    

@wrap_fn_2_process(daemon=True)
def sample_recv(id, mempool, ):
    def __sample_recv(req, mempool, ):
        ret = MemPoolProtocol().generate_samples(req)
        for item in ret:
            mempool.append(item)
            # print('learner recv sample: ', item.shape)
        return b'success'

    worker = worker_rrrproxy(id, yaml_rmt_agent.mempool_fe.worker.url_bkend, True, __sample_recv, mempool)
    worker.send(None)
    while True:
        byte_rsp = worker.send(None)
        if byte_rsp == WORKER_FAULT_RESULT:
            continue

@wrap_fn_2_process(daemon=True)
def sample_send(id, batch_mgr, ):
    def __sample_send(req, batch_mgr, ):
        assert req == SAMPLE_REQ
        idx_batch = batch_mgr.q_read.get()
        data_batch = batch_mgr.get_one_batch(idx_batch)
        batch_mgr.status_item[idx_batch].release()

        if len(data_batch) > 0:
            # 这里要求不拆包发送, 正常一个包大小BATCH_SIZE, 这里设置max_sample_num=BATCH_SIZE*2, 返回必须是一个元素的list
            format_samples = MemPoolProtocol().format_batch_samples_array(data_batch, priorities=None, max_sample_num=BATCH_SIZE*2)
            assert len(format_samples) == 1
            return format_samples[0]

    worker = worker_rrrproxy(id, yaml_rmt_agent.mempool_be.worker.url_bkend, True, __sample_send, batch_mgr)
    worker.send(None)
    while True:
        byte_rsp = worker.send(None)
        if byte_rsp == WORKER_FAULT_RESULT:
            continue

@wrap_fn_2_process(daemon=True)
def batch_writer(id, batch_mgr, mempool, ):
    wc = batch_mgr.get_write_speed()
    rc = batch_mgr.get_read_speed()
    count = 0

    slot_per_process = batch_mgr.slot_per_process
    while True:
        for i in range(slot_per_process):
            idx_batch = id * slot_per_process + i
            if not batch_mgr.status_item[idx_batch].acquire(block=False):
                continue
            data_batch = list()  # np.ones([bmgr.batch_size, bmgr.len_item]) * idx_batch
            for i in range(BATCH_SIZE):
                data_batch.append(mempool.get_random_item())
            batch_mgr.set_one_batch(idx_batch, data_batch)
            batch_mgr.q_read.put(idx_batch)

            count += 1
            # print('batch_writer ', id, '   ', count)

            if count % 10 == 0 and id == 0:
                print(next(wc), '   ', next(rc), '   ')

# 以下是单元测试使用

@wrap_fn_2_process(daemon=False)
def local_run_learner(game_name):
    from kaiwu_agent.agent.manager.sample_manager import SampleManager
    from kaiwu_agent.conf import yaml_agent
    if tree_switch.check("receive_sample"):
        sample_mgr = SampleManager(yaml_agent.agent.agent_num, game_name)
        while True:
            res = sample_mgr.recv_and_process()
            print('learner run one step')
