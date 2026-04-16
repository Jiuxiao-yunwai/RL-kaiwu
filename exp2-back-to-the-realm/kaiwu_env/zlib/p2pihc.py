from random import randint
import zmq
import logging, sys, time
import zmq.asyncio
from kaiwu_env.zlib.zhelper import HEARTBEAT_IVL_WORKER, REQUEST_RETRIES, REQUEST_TIMEOUT, RESPONSE_RETRIES, RESPONSE_TIMEOUT
from kaiwu_env.zlib.zhelper import SESS_START, SESS_STOP, SESS_UPDATE, B_SEQ_NO_0
from kaiwu_env.utils.common_func import get_uuid
from kaiwu_env.conf import yaml_arena


class P2PClient:
    """
        一个zmq.DEALER实现的P2P客户端, 定义为同步发收, 根据seq_no和worker进行同步, 非阻塞式
    """
    def __init__(self, i, url_ftend, logger=None):
        self.logger = logger if logger else logging.getLogger("P2PClient_for_arena")
        assert isinstance(i, int)

        ctx = zmq.Context.instance()
        self.skt = ctx.socket(zmq.DEALER)
        self.skt.identity = (u"Client-%d-%s" % (i, get_uuid())).encode('ascii')
        self.skt.connect(url_ftend)
        # 一次会话的帧计数器, worker从0开始, client从1开始保证client发送都是worker.seq_no+1, __sync_send_recv调用的开头会自增, 所以初始化是0
        self.seq_no = 0
    
    def start_sess(self, msg=b''):
        """
            同步会话的开始, 直到worker返回reply, 开始会话
            如果返回False, 表示同步失败, 需要使用方再次调用start_sess进行会话同步
        """
        self.seq_no = 0
        seq_no, cmd, reply = self.__sync_send_recv(SESS_START, msg)
        if reply == False:
            return reply
        assert seq_no == self.seq_no
        assert cmd == SESS_START
        assert isinstance(reply, bytes)
        return reply

    def stop_sess(self, msg=b'STOP_SESS_REQUEST'):
        """
            请求会话结束, 如worker返回reply, 表示worker也正常结束会话, 双方调用start_sess进行新一轮会话同步
            如果返回False, 表示worker结束同步失败, 需要worker容灾后调用start_sess进行新一轮会话同步
        """
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
        """
            一次发收的过程调用, 如果正常发收返回seq_no, cmd, reply, 如果收不到服务端的信息返回None, cmd, False
            调用者如果收到False应该重新进行调用start_sess进行新一轮会话同步
        """
        assert isinstance(cmd, bytes)
        assert isinstance(msg, bytes)

        self.seq_no += 1
        b_seq_no = self.seq_no.to_bytes(4, byteorder='big')
        self.skt.send_multipart([b_seq_no, cmd, msg])

        retries_left = REQUEST_RETRIES
        try:
            while True:
                if (self.skt.poll(REQUEST_TIMEOUT) & zmq.POLLIN) != 0:
                    recv_b_seq_no, recv_cmd, reply = self.skt.recv_multipart()
                    int_seq_no = int.from_bytes(recv_b_seq_no, byteorder='big')
                    if int_seq_no == self.seq_no:
                        return int_seq_no, recv_cmd, reply
                else:
                    retries_left -= 1
                    if retries_left == 0:
                        self.logger.debug(f'try {REQUEST_RETRIES} times, client {self.skt.identity} cannot recv response')
                        return None, cmd, False
                    
                    # 即使id一样，重建socket也不会收到前一个socket收到的消息，所以不能重建，直接让zmp负责重连
                    self.skt.send_multipart([b_seq_no, cmd, msg])

        except Exception as e:
            self.logger.warning(f'raise error e: {e}')
            return None, cmd, False


class P2PWorker:
    """
        一个zmq.DEALER实现的P2Pworker端, 定义为任意收发根据seq_no和client进行同步, 非阻塞式
    """
    def __init__(self, i, url_bkend, logger=None):
        self.logger = logger if logger else logging.getLogger("P2PWorker_for_arena")

        assert isinstance(i, int)

        ctx = zmq.Context.instance()
        self.skt = ctx.socket(zmq.ROUTER)
        self.skt.identity = (u"Worker-%d-%s" % (i, get_uuid())).encode('ascii')
            
        self.skt.bind(url_bkend)

        self.poller_skt = zmq.Poller()
        self.poller_skt.register(self.skt, zmq.POLLIN)
        self.__reset()

    def __reset(self):
        # seq_no一次会话的帧计数器, worker从0开始, client从1开始保证client发送都是worker.seq_no+1
        # worker的send和recv分开调用, 使用self.cmd, self.rep, self.rsp记录当前帧的状态
        self.seq_no, self.cmd, self.rep, self.rsp = B_SEQ_NO_0, None, None, None
        self.client_id = None
    
    def recv_start_sess(self):
        """
            同步会话的开始, 需要收到client发送的SESS_START并返回之后, 开始会话
            如果返回False, 表示同步失败, 需要使用方再次调用recv_start_sess进行会话同步
        """
        # 如果正常启动返回True, 如果不正常启动返回False, 不正常包括收不到client信息和收到SESS_STOP信息
        self.__reset()
        req = self.recv()
        if req == False:
            return False
        assert int.from_bytes(self.seq_no, byteorder='big') == 1
        assert self.cmd == SESS_START
        assert isinstance(req, bytes)
        return req
    
    def does_recv_stop(self):
        """
            判断worker是否在当前帧收到了SESS_STOP消息, worker无法主动判断是否要去新启动一次会话, 如果判断收到SESS_STOP则新启动一次会话
        """
        return self.cmd == SESS_STOP

    def recv(self):
        """
            收client发送的信息, 如果正常发收返回bytes类型, 如果收不到client的信息返回False
            调用者如果收到False应该重新进行调用recv_start_sess进行新一轮会话同步 
        """
        retries_left = RESPONSE_RETRIES
        try:
            while True:
                
                socks = dict(self.poller_skt.poll(HEARTBEAT_IVL_WORKER * RESPONSE_TIMEOUT))

                if socks.get(self.skt) == zmq.POLLIN:
                    msg = self.skt.recv_multipart()
                    client_id, seq_no, cmd, req = msg
                    # self.logger.debug(f"client_id: {client_id}  seq_no: {int.from_bytes(self.seq_no, byteorder='big')} {int.from_bytes(seq_no, byteorder='big')}  cmd: {cmd} ")
                    if self.client_id == None:
                        self.client_id = client_id
                    if client_id != self.client_id:
                        self.logger.warning(f'worker {self.skt.identity} now serve {self.client_id}, recv a new client_id {client_id}, skip')
                        self.__suicide()
                        return False
                    if int.from_bytes(seq_no, byteorder='big') == int.from_bytes(self.seq_no, byteorder='big') + 1:
                        # 更新最新的seq_no, cmd, req
                        self.seq_no, self.cmd, self.req = seq_no, cmd, req
                        return req
                    # 如果client已经开始同步新会话且worker不在同步新会话，应该跳出
                    # 以下代码可以不要，为了支持busy模式暂时保留
                    if cmd == SESS_START and self.cmd != SESS_START and yaml_arena.comm_type == "busy":
                        self.logger.debug(f'worker need to sync start sess')
                        self.__suicide()
                        return False
                else: 
                    retries_left -= 1
                    if retries_left == 0:
                        self.logger.debug(f'try {RESPONSE_RETRIES} times, worker {self.skt.identity} cannot recv request')
                        self.__suicide()
                        return False

                    # 如果是第一次收, 没有前序rsp则不重发
                    if self.seq_no != B_SEQ_NO_0:
                        self.skt.send_multipart([self.client_id, self.seq_no, self.cmd, self.rsp])
        except Exception as e:
            self.logger.warning(f'raise error e: {e}')
            self.__suicide()
            return False

    def send(self, rsp, callback=False, func=None, *args, **kargs):
        """
            发送处理完之后的rsp, 如果是SESS_STOP请求则返回False表示为会话结束需要进行新一轮会话同步 
            调用者如果收到False应该重新进行调用recv_start_sess进行新一轮会话同步 
        """
        # 如果是SESS_STOP返回False, 否则返回True
        assert callable(func) if callback else True
        if callback:
            rsp = func(rsp, *args, **kargs)          # time.sleep(randint(0, 1))     # simulate some workload
        
        self.rsp = rsp
        self.skt.send_multipart([self.client_id, self.seq_no, self.cmd, self.rsp])

        if self.cmd == SESS_STOP:
            self.__suicide()
            return False
        return True
    
    def __suicide(self):
        # 自杀需要reset自己并给proxy发消息
        self.__reset()