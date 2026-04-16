
import zmq
from threading import Thread
import logging, sys, time
from kaiwu_env.zlib.zhelper import HEARTBEAT_IVL_WORKER, HEARTBEAT_WORKER, HEARTBEAT_PROXY, HEARTBEAT_IVL_PROXY, DEFAULT_LIVENESS_WOKER, DEFAULT_LIVENESS_PROXY, WORKER_FAULT_RESULT, SLEEP_TIME_REREP, zpipe
from kaiwu_env.utils.common_func import wrap_fn_2_process
import asyncio
import zmq.asyncio


class ReliaRRProxy():
    
    def __init__(self, url_ftend, url_bkend, flag_monitor=False) -> None:
        super().__init__()
        self.url_ftend = url_ftend
        self.url_bkend = url_bkend
        self.flag_monitor = flag_monitor
    
    @wrap_fn_2_process(daemon=True)
    def run(self):
        logger = logging.getLogger("%s.%s.%s" % (__name__, __class__.__name__, sys._getframe().f_code.co_name))
        self.ctx = zmq.Context.instance()

        skt_ftend = self.ctx.socket(zmq.ROUTER)
        skt_ftend.bind(self.url_ftend)
        
        skt_bkend = self.ctx.socket(zmq.ROUTER)
        skt_bkend.bind(self.url_bkend)

        poller_bkend = zmq.Poller()
        poller_bkend.register(skt_bkend, zmq.POLLIN)

        poller_both = zmq.Poller()
        poller_both.register(skt_ftend, zmq.POLLIN)
        poller_both.register(skt_bkend, zmq.POLLIN)

        # worker的管理结构，成对出现，成对处理
        id_workers = []
        dict_liveness_workers = dict()

        if self.flag_monitor:
            pipe = zpipe(self.ctx)
            m_thread = Thread(target=self.moniter_thread, args=(pipe[1],))
            m_thread.daemon = True
            m_thread.start()
        
        heartbeat_at = time.time() + HEARTBEAT_IVL_PROXY

        while True:
            # 不管啥情况，心跳都是要发的，Send heartbeats to idle workers
            if time.time() >= heartbeat_at:
                for id_worker in id_workers:
                    skt_bkend.send_multipart([id_worker, HEARTBEAT_PROXY])
                heartbeat_at = time.time() + HEARTBEAT_IVL_PROXY
            
            if id_workers:
                tmp = poller_both
            else:
                tmp = poller_bkend
            socks = dict(tmp.poll(HEARTBEAT_IVL_PROXY * 1000))

            if socks.get(skt_bkend) == zmq.POLLIN:
                msg = skt_bkend.recv_multipart()
                if not msg:
                    break

                if self.flag_monitor:
                    pipe[0].send_multipart(msg)
                
                id_worker = msg[0]
                rep = msg[1:]
                
                if len(rep) == 1 and rep[0] == HEARTBEAT_WORKER:
                    # handle heartbeat
                    logger.debug("recv a heartbeat from: %s " % id_worker.decode('ascii'))
                else:
                    # logger.debug('bkend recv %s %s %s' % (id_worker.decode('ascii'), rep[0].decode('ascii'), rep[1].decode('ascii')))
                    skt_ftend.send_multipart(rep)
                
                if id_worker not in id_workers:
                    id_workers.append(id_worker)
                dict_liveness_workers[id_worker] = DEFAULT_LIVENESS_WOKER
            
            else:
                remove_list = list()
                for index, id_worker in enumerate(id_workers):
                    if id_worker in dict_liveness_workers:
                        dict_liveness_workers[id_worker] -= 1
                        if dict_liveness_workers[id_worker] < 0:
                            remove_list.append(index)
                            del dict_liveness_workers[id_worker]
                if remove_list:
                    remove_list.reverse()
                    for index in remove_list:
                        del id_workers[index]
                # 后端的else会删除超时的worker，有可能不留一个，此时不让前端收请求
                if len(id_workers) == 0:
                    continue

            # 有worker空闲了再去理请求，反正请求会在前端队列里等着
            if socks.get(skt_ftend) == zmq.POLLIN:
                msg = skt_ftend.recv_multipart()
                logger.debug('proxy recv a msg')
                if not msg:
                    break

                if self.flag_monitor:
                    pipe[0].send_multipart(msg)
                
                id_worker = id_workers.pop(0)
                del dict_liveness_workers[id_worker]

                # logger.debug('get a worker %s %s' % (msg[0].decode('ascii'), msg[1].decode('ascii')))
                skt_bkend.send_multipart([id_worker] + msg)   
            
    def moniter_thread(self, pipe):
        logger = logging.getLogger("%s.%s.%s" % (__name__, __class__.__name__, sys._getframe().f_code.co_name))
        while True:
            try:
                logger.debug("%s monitor recv: %s" % (sys._getframe().f_code.co_name, pipe.recv_multipart()))
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    break   

