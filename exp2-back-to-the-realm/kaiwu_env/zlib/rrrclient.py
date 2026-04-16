
from random import randint, random
import zmq
import logging, sys, time
# from kaiwu_env.common_python.ipc.connection import Connection
import socket
import asyncio
import zmq.asyncio


def client_for_arena(i, url_ftend):
        """
            一个协程实现的客户端，一个进程中可以通过c=coroutine_client_rrrproxy(1), c.send(None)实例化协程
            然后想要发送数据的时候，使用c.send(msg)实现数据发送
        """
        logger = logging.getLogger("client_for_arena")

        assert isinstance(i, int)

        REQUEST_TIMEOUT = 1000
        REQUEST_RETRIES = 3

        ctx = zmq.Context.instance()
        skt = ctx.socket(zmq.DEALER)
        skt.identity = (u"Client-%d-%04X" % (i, randint(0, 0x10000))).encode('ascii')
        skt.connect(url_ftend)

        reply = b'' 
        while True:
            msg  = yield reply

            skt.send(msg)

            retries_left = REQUEST_RETRIES
            # logger.warning('client send')
            try:
                while True:
                    if (skt.poll(REQUEST_TIMEOUT) & zmq.POLLIN) != 0:
                        reply = skt.recv()
                        # logger.error('client %d recv: %s' % (i, reply.decode('ascii')))
                        break 
                    else:
                        # timeout, proxy loss
                        retries_left -= 1
                        if retries_left == 0:
                            logger.warning('client out')
                            # 如果超出最大重试次数, 应该做一些容灾操作
                            # sys.exit()
                    
                        time.sleep(1)
                        # 即使id一样，重建socket也不会收到前一个socket收到的消息，所以不能重建，直接让zmp负责重连
                        skt.send(msg)

            except zmq.ContextTerminated:
                # context terminated so quit silently
                return

async def client_rrrproxy(i, url_ftend):
    """
        一个协程实现的客户端，一个进程中可以通过c=coroutine_client_rrrproxy(1), c.send(None)实例化协程
        然后想要发送数据的时候，使用c.send(msg)实现数据发送
    """
    logger = logging.getLogger("async_client_rrrproxy")

    assert isinstance(i, int)

    REQUEST_TIMEOUT = 2500
    REQUEST_RETRIES = 30000 

    ctx = zmq.asyncio.Context.instance()
    skt = ctx.socket(zmq.DEALER)
    skt.identity = (u"Client-%d-%04X" % (i, randint(0, 0x10000))).encode('ascii')
    skt.connect(url_ftend)

    reply = b'' 
    while True:
        msg  = yield reply

        await skt.send(msg)

        retries_left = REQUEST_RETRIES
        # logger.warning('client send')
        try:
            while True:
                if (await skt.poll(REQUEST_TIMEOUT) & zmq.POLLIN) != 0:
                    reply = await skt.recv()
                    # logger.error('client %d recv: %s' % (i, reply.decode('ascii')))
                    # await asyncio.sleep(random())
                    break 
                else:
                    # timeout, proxy loss
                    retries_left -= 1
                    if retries_left == 0:
                        logger.warning('client out')
                        sys.exit()
                    
                    time.sleep(1)
                    # 即使id一样，重建socket也不会收到前一个socket收到的消息，所以不能重建，直接让zmp负责重连
                    await skt.send(msg)

        except zmq.ContextTerminated:
            # context terminated so quit silently
            return


class client_for_drl:
    def __init__(self, id) -> None:
        aisrv_ip='127.0.0.1'
        aisrv_port='8008'
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        status = sock.connect_ex((aisrv_ip, int(aisrv_port)))
        if status == 0:
            print(f"Socket connect to aisrv: {aisrv_ip}:{aisrv_port} ... Succeed")
        else:
            print(f"Socket connect to aisrv: {aisrv_ip}:{aisrv_port} ... Failed!")
            return
        self.conn = Connection(sock)

    def send(self, msg):
        self.conn.send_msg(msg)
        return self.conn.recv_msg().tobytes()
    