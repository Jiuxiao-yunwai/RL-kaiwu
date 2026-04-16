
import zmq
from threading import Thread
import logging, sys, time
from kaiwu_env.zlib.zhelper import HEARTBEAT_IVL_WORKER, HEARTBEAT_WORKER, HEARTBEAT_PROXY, HEARTBEAT_IVL_PROXY, DEFAULT_LIVENESS_WOKER, DEFAULT_LIVENESS_PROXY, WORKER_FAULT_RESULT, SLEEP_TIME_REREP, zpipe
from kaiwu_env.utils.common_func import wrap_fn_2_process
import asyncio
import zmq.asyncio
from kaiwu_env.zlib.zhelper import SESS_START, SESS_STOP, SESS_UPDATE, HEARTBEAT_WORKER_KILL


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

        poller_both = zmq.Poller()
        poller_both.register(skt_ftend, zmq.POLLIN)
        poller_both.register(skt_bkend, zmq.POLLIN)

        # worker的管理结构, 一个worker要么在id_workers中要么在dict_client_worker中, dict_liveness_workers中记录worker的剩余心跳
        id_workers = []
        dict_liveness_workers = dict()
        dict_client_worker = dict()

        if self.flag_monitor:
            pipe = zpipe(self.ctx)
            m_thread = Thread(target=self.moniter_thread, args=(pipe[1],))
            m_thread.daemon = True
            m_thread.start()
        
        heartbeat_at = time.time() + HEARTBEAT_IVL_PROXY

        while True:
            # 不管啥情况,心跳都是要发的, 在dict_client_worker中的worker有可能长时间收不到req,此时也需要发心跳不然worker会告警
            if time.time() >= heartbeat_at:
                for id_worker in id_workers:
                    skt_bkend.send_multipart([id_worker, HEARTBEAT_PROXY])
                for id_worker in  dict_client_worker.values():
                    skt_bkend.send_multipart([id_worker, HEARTBEAT_PROXY])
                heartbeat_at = time.time() + HEARTBEAT_IVL_PROXY

                # 超过一次心跳时间, 就将dict_liveness_workers每个worker计时器都-1, 小于0则认为worker死亡, worker一旦活跃则设成默认值
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
                
                remove_list = list()
                for id_client, id_worker in dict_client_worker.items():
                    if id_worker in dict_liveness_workers:
                        dict_liveness_workers[id_worker] -= 1
                        if dict_liveness_workers[id_worker] < 0:
                            remove_list.append(id_client)
                            del dict_liveness_workers[id_worker]
                if remove_list:
                    remove_list.reverse()
                    for id_client in remove_list:
                        del dict_client_worker[id_client]
            
            socks = dict(poller_both.poll(HEARTBEAT_IVL_PROXY * 1000))

            if socks.get(skt_bkend) == zmq.POLLIN:
                msg = skt_bkend.recv_multipart()
                if not msg:
                    break

                if self.flag_monitor:
                    pipe[0].send_multipart(msg)
                
                id_worker = msg[0]
                rep = msg[1:]

                # 只要有收到东西, 则认为与心跳等同
                dict_liveness_workers[id_worker] = DEFAULT_LIVENESS_WOKER
                
                if len(rep) == 1 and rep[0] == HEARTBEAT_WORKER:
                    # handle heartbeat
                    logger.debug("recv a heartbeat from: %s " % id_worker.decode('ascii'))
                    # 这个worker不在dict_client_worker中并且不在id_workers中, 则他是新的或刚空闲, 加入id_workers
                    if id_worker not in dict_client_worker.values() and id_worker not in id_workers:
                        id_workers.append(id_worker)
                # worker可以自杀
                elif len(rep) == 1 and rep[0] == HEARTBEAT_WORKER_KILL:
                    if id_worker in dict_liveness_workers:
                        del dict_liveness_workers[id_worker]
                    if id_worker in dict_client_worker.values():
                        for id_client, value in dict_client_worker.items():
                            if value == id_worker:
                                del dict_client_worker[id_client]
                                break
                    if id_worker in id_workers:
                        id_workers.remove(id_worker)

                else:
                    # logger.debug('bkend recv %s %s %s' % (id_worker.decode('ascii'), rep[0].decode('ascii'), rep[1].decode('ascii')))
                    skt_ftend.send_multipart(rep)
                    
                    # 如果cmd == SESS_STOP, 需要等到worker返回之后删除worker, 因为worker自杀有时延, proxy有可能依然给这个worker分配任务
                    id_client, seq_no, cmd, _ = rep
                    if cmd == SESS_STOP:
                        del dict_liveness_workers[dict_client_worker[id_client]]
                        del dict_client_worker[id_client]
            
            # 不能等有worker空闲了再去理请求
            if socks.get(skt_ftend) == zmq.POLLIN:
                msg = skt_ftend.recv_multipart()
                logger.debug('proxy recv a msg')
                if not msg:
                    break

                if self.flag_monitor:
                    pipe[0].send_multipart(msg)

                id_client, seq_no, cmd, req = msg
                if id_client in dict_client_worker.keys():
                    # cmd == SESS_START在else中处理, 不可能出现在这里, 如果出现, 告警,需要client自己重发
                    if cmd == SESS_START:
                        print("a new client try to connect to an old worker")
                        del dict_client_worker[id_client]
                        continue
                    # 转发到后端
                    skt_bkend.send_multipart([dict_client_worker[id_client]] + msg)   
                    print(id_client, seq_no, cmd, req)
                else:
                    # 如果找不到服务的worker并且cmd不是SESS_START, 则意味着丢失worker, 告警,需要client自己检测重新开始会话
                    if cmd != SESS_START:
                        continue
                    # cmd==SESS_START, 如果没有可用的worker则当没收到过, 告警,需要client自己重发
                    if len(id_workers) == 0:
                        continue
                    # cmd==SESS_START并且有空闲的worker, 则分配worker开始服务
                    id_worker = id_workers.pop(0)
                    dict_client_worker[id_client] = id_worker
                    skt_bkend.send_multipart([id_worker] + msg)   
                    print(id_client, seq_no, cmd, req)
            
    def moniter_thread(self, pipe):
        logger = logging.getLogger("%s.%s.%s" % (__name__, __class__.__name__, sys._getframe().f_code.co_name))
        while True:
            try:
                logger.debug("%s monitor recv: %s" % (sys._getframe().f_code.co_name, pipe.recv_multipart()))
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    break   

