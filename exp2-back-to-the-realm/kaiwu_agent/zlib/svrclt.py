from random import randint, random
import zmq
import logging, sys, time
import zmq.asyncio
from kaiwu_agent.zlib.zhelper import HEARTBEAT_IVL_WORKER, HEARTBEAT_WORKER, HEARTBEAT_PROXY, HEARTBEAT_IVL_PROXY, DEFAULT_LIVENESS_WOKER, DEFAULT_LIVENESS_PROXY, WORKER_FAULT_RESULT, SLEEP_TIME_REREP, zpipe
from kaiwu_agent.zlib.zhelper import SESS_START, SESS_STOP, SESS_UPDATE, HEARTBEAT_WORKER_KILL, B_SEQ_NO_0


REQUEST_TIMEOUT = 1000
REQUEST_RETRIES = 20 
RESPONSE_RETRIES = 20

class client_svrproxy:

    def __init__(self, i, url_ftend):
        self.logger = logging.getLogger("client_for_arena")
        assert isinstance(i, int)

        ctx = zmq.Context.instance()
        self.skt = ctx.socket(zmq.DEALER)
        self.skt.identity = (u"Client-%d-%04X" % (i, randint(0, 0x10000))).encode('ascii')
        self.skt.connect(url_ftend)
        # frame seq no, sync_send_recv调用的开头会自增, 所以初始化是0
        self.seq_no = 0
    
    def start_sess(self, msg=b''):
        # 如果正常启动返回reply, 如果不正常启动返回False, 不正常包括收不到服务端的信息
        self.seq_no = 0
        seq_no, cmd, reply = self.__sync_send_recv(SESS_START, msg)
        if reply == False:
            return reply
        assert seq_no == self.seq_no
        assert cmd == SESS_START
        assert isinstance(reply, bytes)
        return reply

    def stop_sess(self, msg=b''):
        # 如果正常结束返回reply, 如果不正常结束返回False, 不正常包括收不到服务端的信息
        seq_no, cmd, reply = self.__sync_send_recv(SESS_STOP, msg)
        if reply == False:
            return reply
        assert seq_no == self.seq_no
        assert cmd == SESS_STOP
        assert isinstance(reply, bytes)
        self.seq_no = 0
        return reply
    
    def update_sess(self, msg):
        return self.__sync_send_recv(SESS_UPDATE, msg)

    def __sync_send_recv(self, cmd, msg):
        # 如果正常发收返回int_seq_no, cmd, reply, 如果收不到服务端的信息返回None, cmd, False
        assert isinstance(cmd, bytes)
        assert isinstance(msg, bytes)

        self.seq_no += 1
        b_seq_no = self.seq_no.to_bytes(4, byteorder='big')
        self.skt.send_multipart([b_seq_no, cmd, msg])

        retries_left = REQUEST_RETRIES
        try:
            while True:
                if (self.skt.poll(REQUEST_TIMEOUT) & zmq.POLLIN) != 0:
                    b_seq_no, cmd, reply = self.skt.recv_multipart()
                    int_seq_no = int.from_bytes(b_seq_no, byteorder='big')
                    if int_seq_no == self.seq_no:
                        return int_seq_no, cmd, reply
                else:
                    # timeout, proxy loss
                    retries_left -= 1
                    if retries_left == 0:
                        self.logger.warning('client out')
                        return None, cmd, False
                    
                    time.sleep(1)
                    # 即使id一样，重建socket也不会收到前一个socket收到的消息，所以不能重建，直接让zmp负责重连
                    self.skt.send_multipart([b_seq_no, cmd, msg])

        except zmq.ContextTerminated:
            # context terminated so quit silently
            return


# 如果对应的client消失, worker发送自杀心跳被proxy认为死亡, 此时worker应该适当时间 
class worker_svrproxy:
    def __init__(self, i, url_bkend):
        self.logger = logging.getLogger("worker")

        assert isinstance(i, int)

        ctx = zmq.Context.instance()
        self.skt = ctx.socket(zmq.DEALER)
        self.skt.identity = (u"Worker-%d-%04X" % (i, randint(0, 0x10000))).encode('ascii')
            
        self.skt.connect(url_bkend)

        self.poller_skt = zmq.Poller()
        self.poller_skt.register(self.skt, zmq.POLLIN)
        self.__reset()

    def __reset(self):
        self.id_client, self.seq_no, self.cmd, self.rep, self.rsp = None, B_SEQ_NO_0, None, None, None
    
    def start_sess(self, func, *args, **kargs):
        # 如果正常启动返回True, 如果不正常启动返回False, 不正常包括收不到client信息和收到SESS_STOP信息
        self.__reset()
        req = self.recv()
        if req == False:
            return False
        assert int.from_bytes(self.seq_no, byteorder='big') == 1
        assert self.cmd == SESS_START
        assert isinstance(req, bytes)
        return self.send(req, callback=True, func=func, *args, **kargs)

    def recv(self):
        # 如果收到返回bytes, 如果收不到返回False
        heartbeat_at = time.time()
        liveness_proxy = DEFAULT_LIVENESS_PROXY
        sleep_interval_worker = SLEEP_TIME_REREP

        retries_left = RESPONSE_RETRIES
        while True:
            
            # 不管啥情况，心跳到时间都得发
            if time.time() > heartbeat_at:
                heartbeat_at = time.time() + HEARTBEAT_IVL_WORKER
                self.skt.send(HEARTBEAT_WORKER)

            socks = dict(self.poller_skt.poll(HEARTBEAT_IVL_WORKER * 1000))

            if socks.get(self.skt) == zmq.POLLIN:
                msg = self.skt.recv_multipart()
                # logger.debug("%s receive req or heartbeat: %s" % (skt.identity.decode('ascii'), msg[0].decode('ascii')))
                            
                if len(msg) == 1 and msg[0] == HEARTBEAT_PROXY:
                    self.logger.debug("recv a heartbeat ")
                    liveness_proxy = DEFAULT_LIVENESS_PROXY
                    sleep_interval_worker = SLEEP_TIME_REREP
                else:
                    id_client, seq_no, cmd, req = msg
                    assert self.id_client == None or id_client == self.id_client
                    if int.from_bytes(seq_no, byteorder='big') == int.from_bytes(self.seq_no, byteorder='big') + 1:
                        self.id_client, self.seq_no, self.cmd, self.req = id_client, seq_no, cmd, req
                        return req
                        
            else:
                # timeout, proxy loss
                liveness_proxy -= 1
                if liveness_proxy == 0:
                    self.logger.warning("%s report: proxy die" % self.skt.identity.decode('ascii'))
                    # just sleep and wait proxy alive
                    time.sleep(sleep_interval_worker)
                    if sleep_interval_worker < SLEEP_TIME_REREP * 16:
                        sleep_interval_worker *= 2
                    liveness_proxy = DEFAULT_LIVENESS_PROXY
    
                retries_left -= 1
                if retries_left == 0:
                    self.__suicide()
                    return False
                # 如果是第一次收, 没有前序rsp则不重发
                if self.seq_no != B_SEQ_NO_0:
                    time.sleep(1)
                    self.skt.send_multipart([self.id_client, self.seq_no, self.cmd, self.rsp])

    def send(self, rsp, callback=False, func=None, *args, **kargs):
        # 如果是SESS_STOP返回False, 否则返回True
        assert callable(func) if callback else True
        # logger.debug('worker recv %s %s' % (id_client.decode('ascii'), req.decode('ascii')))
        if callback:
            rsp = func(rsp, *args, **kargs)          # time.sleep(randint(0, 1))     # simulate some workload
        
        self.rsp = rsp
        self.skt.send_multipart([self.id_client, self.seq_no, self.cmd, self.rsp])

        # print(self.id_client, self.seq_no, self.cmd, rsp)
        if self.cmd == SESS_STOP:
            self.__suicide()
            return False
        return True
    
    def __suicide(self):
        # 自杀需要reset自己并给proxy发消息
        self.__reset()
        self.skt.send(HEARTBEAT_WORKER_KILL)
