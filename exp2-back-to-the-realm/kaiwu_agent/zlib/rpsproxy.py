
from random import randint
from pickle import dumps, loads
import zmq
from multiprocessing import Process
from threading import Thread
import logging, sys, time
from kaiwu_agent.zlib.zhelper import zpipe, CLOSE_SUB, CONNECT_SUB, TIME_IVL_PUB, HEARTBEAT_PUB, HEARTBEAT_IVL_PUB, HEARTBEAT_SUB, HEARTBEAT_IVL_SUB, TIME_IVL_DETECT, PS_MAX_ALLOWED_DELAY
from kaiwu_agent.utils.common_func import wrap_fn_2_process

class ReliaPSProxy():

    def __init__(self, url_ftend, url_bkend, flag_monitor=True, use_lvc=True) -> None:
        super().__init__()
        self.url_ftend = url_ftend
        self.url_bkend = url_bkend
        self.flag_monitor = flag_monitor
        self.use_lvc = use_lvc
    
    @wrap_fn_2_process(daemon=True)
    def run(self):
        logger = logging.getLogger("%s.%s.%s" % (__name__, __class__.__name__, sys._getframe().f_code.co_name))
        self.ctx = zmq.Context.instance()

        skt_ftend = self.ctx.socket(zmq.XSUB)
        skt_ftend.bind(self.url_ftend)
        
        skt_bkend = self.ctx.socket(zmq.XPUB)
        skt_bkend.bind(self.url_bkend)

        # note: by default, the XPUB socket does not report duplicate subscriptions, you’ll want to use the zmq.XPUB_VERBOSE=True to avoid
        skt_bkend.setsockopt(zmq.XPUB_VERBOSE, True)

        poller_both = zmq.Poller()
        poller_both.register(skt_ftend, zmq.POLLIN)
        poller_both.register(skt_bkend, zmq.POLLIN)

        if self.flag_monitor:
            pipe = zpipe(self.ctx)
            m_thread = Thread(target=self.moniter_thread, args=(pipe[1],))
            m_thread.daemon = True
            m_thread.start()
        
        # 前端订阅所有频道，神奇的是启动proxy后新增的XPUB前端也能订阅到
        # skt_ftend.send(b'\x01')

        if self.use_lvc:
            cache = {}
        
        dict_pubs = dict()
        dict_subs = dict()
        time_detect = time.time()

        while True:
            
            socks = dict(poller_both.poll(TIME_IVL_DETECT * 10000))
            if socks.get(skt_ftend) == zmq.POLLIN:
                # 处理发布的消息，使用recv_multipart
                msg = skt_ftend.recv_multipart()
                if not msg:
                    break
                
                topic, rep = msg

                # 只要是前端发来的信息，都作为XPUB心跳
                dict_pubs[topic] = time.time()

                # 除了心跳，其他简单发送到后端
                if rep != HEARTBEAT_PUB:
                    # logger.debug('proxy ftend %03d recv: %s' % (int(topic.decode('ascii')), rep.decode('ascii')))
                    skt_bkend.send_multipart(msg)

                    # Last Value Caching Mode
                    if self.use_lvc:
                        cache[topic] = rep

            if socks.get(skt_bkend) == zmq.POLLIN:
                # 处理subscriptions，使用recv
                event = skt_bkend.recv()
                if not event:
                    break

                flag, topic = event[:1], event[1:]

                # 只要是后端发来的信息，都作为XSUB的心跳
                dict_subs[topic] = time.time()

                if flag == HEARTBEAT_SUB:
                    print('hahahahahahahaaaaaaaaaaaaaaaaaaa-----------')
                
                # 除了心跳，其他简单发送到后端
                if flag != HEARTBEAT_SUB:
                    skt_ftend.send(event)

                    if self.flag_monitor:
                        pipe[0].send(event)

                    # Last Value Caching Mode
                    if self.use_lvc:
                        # Event is one byte 0=unsub or 1=sub, followed by topic
                        if flag == CONNECT_SUB:
                            # logger.debug('proxy bkend recv subscriptions: %s' % topic.decode('ascii'))
                            if topic in cache:
                                logger.info("Sending cached topic %s  %s" % (topic, cache[topic]))
                                skt_bkend.send_multipart([topic, cache[topic]])
            
            # 做简单的检测
            if time.time() > time_detect + TIME_IVL_DETECT:
                time_detect = time.time()
                for topic, ts in dict_pubs.items():
                    if time.time() - ts > HEARTBEAT_IVL_PUB * 10:
                        logger.error('topic: %s . pub long time no send' % topic.decode('ascii'))
                    if topic in dict_subs.keys() and dict_pubs[topic] - dict_subs[topic] > PS_MAX_ALLOWED_DELAY:
                        print(dict_pubs[topic], dict_subs[topic], dict_pubs[topic] - dict_subs[topic])
                        logger.error('topic: %s . sub long time no recv. time-gap: %4f' % (topic.decode('ascii'), dict_pubs[topic] - dict_subs[topic]))


    def moniter_thread(self, pipe):
        logger = logging.getLogger("%s.%s.%s" % (__name__, __class__.__name__, sys._getframe().f_code.co_name))
        while True:
            try:
                logger.info("%s monitor recv: %s" % (sys._getframe().f_code.co_name, pipe.recv_multipart()))
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    break


if __name__ == '__main__':
    pass