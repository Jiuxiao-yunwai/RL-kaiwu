
from random import randint, random
import zmq
import logging, sys, time
from kaiwu_env.zlib.zhelper import HEARTBEAT_IVL_WORKER, HEARTBEAT_WORKER, HEARTBEAT_PROXY, HEARTBEAT_IVL_PROXY, DEFAULT_LIVENESS_WOKER, DEFAULT_LIVENESS_PROXY, WORKER_FAULT_RESULT, SLEEP_TIME_REREP, zpipe
from kaiwu_env.utils.common_func import wrap_fn_2_process
import asyncio
import zmq.asyncio


def worker_rrrproxy(i, url_bkend, callback, func=None, *args, **kargs):
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
                # do real work
                id_client, req = msg
                # logger.debug('worker recv %s %s' % (id_client.decode('ascii'), req.decode('ascii')))
                if callback:
                    result = func(req, *args, **kargs)          # time.sleep(randint(0, 1))     # simulate some workload
                else:
                    result = yield req
                skt.send_multipart([id_client, result])

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
    
