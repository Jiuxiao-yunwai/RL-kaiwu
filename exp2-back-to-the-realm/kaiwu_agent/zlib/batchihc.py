
from random import randint, random
import zmq
import logging, sys, time
from kaiwu_agent.zlib.zhelper import HEARTBEAT_IVL_WORKER, HEARTBEAT_WORKER, HEARTBEAT_PROXY, HEARTBEAT_IVL_PROXY, DEFAULT_LIVENESS_WOKER, DEFAULT_LIVENESS_PROXY, WORKER_FAULT_RESULT, SLEEP_TIME_REREP, zpipe
from kaiwu_agent.zlib.zhelper import BATCH_SIZE, REQUEST_TIMEOUT, REQUEST_RETRIES
from pickle import dumps


def batch_client(i, url_ftend):
        """
            一个协程实现的客户端，一个进程中可以通过c=coroutine_client_rrrproxy(1), c.send(None)实例化协程
            然后想要发送数据的时候，使用c.send(msg)实现数据发送
        """
        logger = logging.getLogger("client_for_arena")

        assert isinstance(i, int)

        ctx = zmq.Context.instance()
        skt = ctx.socket(zmq.DEALER)
        skt.identity = (u"Client-%d-%04X" % (i, randint(0, 0x10000))).encode('ascii')
        skt.connect(url_ftend)

        # 因为client可能会发送很多次, 会收到重复的请求和回复, 需要seq_no标识
        seq_no = 0
        reply = b'' 
        while True:
            msg  = yield reply

            seq_no = (seq_no + 1) % 10000
            b_seq_no = seq_no.to_bytes(4, byteorder='big')
            skt.send_multipart([b_seq_no, msg])

            retries_left = REQUEST_RETRIES
            # logger.warning('client send')
            try:
                while True:
                    if (skt.poll(REQUEST_TIMEOUT) & zmq.POLLIN) != 0:
                        b_seq_no, reply = skt.recv_multipart()
                        int_seq_no = int.from_bytes(b_seq_no, byteorder='big')
                        if int_seq_no == seq_no:
                            break 
                    else:
                        # timeout, proxy loss
                        retries_left -= 1
                        if retries_left == 0:
                            logger.warning('client out')
                            # 如果超出最大重试次数, 返回false表示获取不到了
                            reply = False
                            break
                    
                        time.sleep(1)
                        # 即使id一样，重建socket也不会收到前一个socket收到的消息，所以不能重建，直接让zmp负责重连
                        skt.send_multipart([b_seq_no, msg])

            except zmq.ContextTerminated:
                # context terminated so quit silently
                return
    



def batch_worker(i, url_bkend, callback, func=None, *args, **kargs):
    logger = logging.getLogger("worker")

    assert isinstance(i, int)
    assert callable(func) if callback else True

    ctx = zmq.Context.instance()
    skt = ctx.socket(zmq.DEALER)
    skt.identity = (u"Worker-%d-%04X" % (i, randint(0, 0x10000))).encode('ascii')
        
    skt.connect(url_bkend)

    poller_skt = zmq.Poller()
    poller_skt.register(skt, zmq.POLLIN)

    heartbeat_at = time.time()
    liveness_proxy = DEFAULT_LIVENESS_PROXY
    sleep_interval_worker = SLEEP_TIME_REREP

    quota = BATCH_SIZE
    list_msg = list()
    # 如果超时不管有几个请求都赶紧处理
    time_last_recv_req = time.time()

    result = WORKER_FAULT_RESULT
    while True:
        tmp = yield result
        result = WORKER_FAULT_RESULT

        # 不管啥情况，心跳到时间都得发
        if time.time() > heartbeat_at:
            heartbeat_at = time.time() + HEARTBEAT_IVL_WORKER
            skt.send(HEARTBEAT_WORKER)

        socks = dict(poller_skt.poll(HEARTBEAT_IVL_WORKER * 1000))

        if socks.get(skt) == zmq.POLLIN:
            msg = skt.recv_multipart()
            # logger.debug("%s receive req or heartbeat: %s" % (skt.identity.decode('ascii'), msg[0].decode('ascii')))
                    
            if len(msg) == 1 and msg[0] == HEARTBEAT_PROXY:
                logger.debug("recv a heartbeat ")
            else:
                quota -= 1
                list_msg.append(msg)
                time_last_recv_req = time.time()
            if quota == 0 or (time.time() > time_last_recv_req + HEARTBEAT_IVL_WORKER * BATCH_SIZE * 2 and len(list_msg) > 0):
                # do real work
                # id_client, seq_no, req = msg
                list_id_client = [id_client for id_client, seq_no, req in list_msg]
                list_seq_no    = [seq_no for id_client, seq_no, req in list_msg]
                list_req       = [req for id_client, seq_no, req in list_msg]
                # logger.debug('worker recv %s %s' % (id_client.decode('ascii'), req.decode('ascii')))
                if callback:
                    list_result = func(list_req, *args, **kargs)          # time.sleep(randint(0, 1))     # simulate some workload
                else:
                    list_result = yield list_req
                list_rsp = [list(x) for x in zip(list_id_client, list_seq_no, list_result)]
                result = dumps(list_rsp)
                skt.send(result)
                quota = BATCH_SIZE
                list_msg = list()

            liveness_proxy = DEFAULT_LIVENESS_PROXY
            sleep_interval_worker = SLEEP_TIME_REREP

        else:
            # timeout, proxy loss
            liveness_proxy -= 1
            if liveness_proxy == 0:
                logger.warning("%s report: proxy die" % skt.identity.decode('ascii'))
                # just sleep and wait proxy alive
                time.sleep(sleep_interval_worker)
                if sleep_interval_worker < SLEEP_TIME_REREP * 16:
                    sleep_interval_worker *= 2
                liveness_proxy = DEFAULT_LIVENESS_PROXY
    