import zmq
import hashlib
import os
import sys
import time
import glob
from collections import OrderedDict
import logging
from kaiwu_agent.zlib.zhelper import CONNECT_SUB, TIME_IVL_PUB, HEARTBEAT_PUB, CLOSE_SUB, HEARTBEAT_IVL_PUB
from kaiwu_agent.agent.protocol.modelpoolproto import ModelProtocol
from kaiwu_agent.zlib.zhelper import CONNECT_SUB, HEARTBEAT_SUB, HEARTBEAT_IVL_SUB
from kaiwu_agent.zlib.rpsproxy import ReliaPSProxy
from kaiwu_agent.conf import yaml_rpssvr
from kaiwu_agent.conf import yaml_modelpool

SIZE_MODELPOOL = yaml_modelpool.SIZE_MODELPOOL

class ModelSendManager(object):
    def __init__(self, url_ftend, is_delete=False, use_bkup=False):
        self.url_ftend = url_ftend
        self.is_delete = is_delete
        # bkup every 30 min.
        self.bkup_model_interval = 60 * 30
        self.use_bkup = use_bkup
        self.step=0

    def run(self, i, sub_sync, num_sub):
        
        logger = logging.getLogger("%s.%s.%s" % (__name__, __class__.__name__, sys._getframe().f_code.co_name))

        assert isinstance(i, int)
        assert isinstance(sub_sync, bool)
        assert isinstance(num_sub, int)

        ctx = zmq.Context.instance()
        skt = ctx.socket(zmq.XPUB)
        skt.connect(self.url_ftend)

        topic_pub = b"%03d" % i
        heartbeat_at = time.time()

        count_sub = 0 
        if sub_sync:
            skt.setsockopt(zmq.XPUB_VERBOSE, True)
            while count_sub < num_sub:
                event = skt.recv()
                flag, topic = event[:1], event[1:]
                if flag == CONNECT_SUB and topic == b"%03d" % i:
                    count_sub += 1
                    logger.debug('xpub %d recv dingyue topic: %s %d/%d stand by' % (i, topic.decode('ascii'), count_sub, num_sub))
                elif topic != b"%03d" % i:
                    logger.debug('xpub recv other subscriptions topic: %s ' % topic.decode('ascii'))
        

        base_path = yaml_modelpool.SEND_MODEL_PATH # '/data/projects/Metagent/send_model/model/'

        ckpt_file_storage_most_count = 50
        cp_file_order_dict = OrderedDict()
        send_file_done_order_dict = OrderedDict()
        the_last_model = None
        last_save_time = None
        protocol = ModelProtocol()

        while True:
            # 这里不处理*.tar而是处理*.pkl
            tmp_cp_file_list = sorted(glob.glob(base_path + "/*.pkl"), key=lambda x: os.path.getmtime(x))
            
            if (skt.poll(TIME_IVL_PUB * 1000) & zmq.POLLIN) != 0:
                event = skt.recv()
                flag, topic = event[:1], event[1:]
                
                # Event is one byte 0=unsub or 1=sub, followed by topic
                if topic == b"%03d" % i:
                    if flag == CONNECT_SUB:
                        count_sub += 1
                        logger.debug('xpub %d recv sub request, topic: %s there are total %d subscribers ever' % (i, topic.decode('ascii'), count_sub))
                    # TODO: 奇怪的事情是，这里永远收不到CLOSE_SUB而PROXY可以收到
                    elif flag == CLOSE_SUB:
                        count_sub -= 1
                        logger.debug('xpub %d recv unsub request, topic: %s there are %d subscribers' % (i, topic.decode('ascii'), count_sub))
                    else:
                        logger.error('receive a undefined flag')
                        exit(1)
                # XPUB有可能收到其他频道的订阅和退订消息，忽略即可
                else:
                    logger.debug('xpub recv other subscriptions topic: %s ' % topic.decode('ascii'))
                
            # 不管啥情况，心跳到时间都得发
            if time.time() > heartbeat_at:
                skt.send_multipart([
                    topic_pub,
                    HEARTBEAT_PUB,
                ])
                heartbeat_at = time.time() + HEARTBEAT_IVL_PUB
                logger.debug('send a heartbeat')

            if (len(tmp_cp_file_list) > 0) and (tmp_cp_file_list[-1] not in cp_file_order_dict):
                logger.debug('if in inhahahah')
                time.sleep(0.2)
                model_file = tmp_cp_file_list[-1]
                model_name = model_file.split('/')[-1]
                if the_last_model is not None and model_name == the_last_model:
                    continue

                model, local_md5 = self._read_model(model_file)
                if model is None:
                    logger.error('fatal error')
                key = model_file.split("/")[-1]
                
                data = protocol.model_encode(key, local_md5, model)
                # do real publish work
                skt.send_multipart([
                    topic_pub, 
                    data,
                ])
                
                print('send: ', type(key), key, local_md5, type(local_md5), type(model))
                self.step += 1
                
                the_last_model = model_name
                # save file info
                for tmp_cp_file in tmp_cp_file_list:
                    cp_file_order_dict[tmp_cp_file] = tmp_cp_file
                    send_file_done_order_dict[tmp_cp_file] = tmp_cp_file+'.done'

                # bkup model
                if self.use_bkup:
                    cur_time = time.time()
                    if last_save_time is None or (cur_time - last_save_time) > self.bkup_model_interval:
                        last_save_time = cur_time
                        need_del_ckpt_file_name = list(cp_file_order_dict.keys())[-1]
                        self._backup_file(need_del_ckpt_file_name, cp_file_order_dict, (base_path + "/model/"))
                        self._backup_file(need_del_ckpt_file_name, send_file_done_order_dict, (base_path + "/model/"))

                if self.is_delete:
                    while(len(cp_file_order_dict) > ckpt_file_storage_most_count):
                        need_del_ckpt_file_name = list(cp_file_order_dict.keys())[0]
                        self._delete_file(need_del_ckpt_file_name, cp_file_order_dict, (base_path+"/model/"))
                        self._delete_file(need_del_ckpt_file_name, send_file_done_order_dict, (base_path+"/model/"))
                        model_name = need_del_ckpt_file_name.split('/')[-1]
                        os.system("cd %s/model/; rm %s*"%(base_path, model_name))

    def _backup_file(self, key_name, order_dict_obj, base_path):
        if key_name in order_dict_obj:
            os.system("cd %s; cp %s /root/model_bkup/" % ((base_path), order_dict_obj[key_name]))

            print("%s has backup to /root/model_bkup/" % (order_dict_obj[key_name]))
            sys.stdout.flush()
            # del order_dict_obj[key_name]

    def _delete_file(self, key_name, order_dict_obj, base_path):
        if key_name in order_dict_obj:
            os.system("cd %s; rm -r %s" % ((base_path), order_dict_obj[key_name]))
            print("%s has deleted" % (order_dict_obj[key_name]))
            sys.stdout.flush()
            del order_dict_obj[key_name]

    def _read_model(self, model_path):
        if not os.path.exists(model_path):
            model = None
            local_md5 = None
            return model, local_md5
        else:
            with open(model_path, "rb") as fin:
                model = fin.read()
            local_md5 = hashlib.md5(model).hexdigest()
        return model, local_md5


def write_model(model_path, file_data):
    with open(model_path, "wb") as fin:
        model = fin.write(file_data)
    return model

def model_recv_mgr(i, url_bkend, ):
    assert isinstance(i, int)
    assert isinstance(url_bkend, str)

    protocol = ModelProtocol()

    ctx = zmq.Context.instance()
    skt = ctx.socket(zmq.XSUB)
    skt.connect(url_bkend)

    sub_topic = b'%03d' % i

    skt.send(CONNECT_SUB + sub_topic)

    heartbeat_at = time.time()

    while True:
        # 不管啥情况，心跳到时间都得发
        if time.time() > heartbeat_at:
            skt.send(HEARTBEAT_SUB + sub_topic)
            heartbeat_at = time.time() + HEARTBEAT_IVL_SUB
            
        # do real subscribe work
        msg = skt.recv_multipart()
        topic, rep = msg
        file_name, remote_md5, message = protocol.model_decode(rep)

        file_path = yaml_modelpool.RECV_MODEL_PATH # '/data/projects/Metagent/recv_model/model/'
        abs_path_file_name = file_path + file_name

        local_md5 = hashlib.md5(message).hexdigest()

        assert local_md5 == remote_md5

        write_model(abs_path_file_name, message)

        # 这里不处理*.tar而是处理*.pkl, 所以不需要解压
        # print("os.system run: tar xvf %s -C %s; rm -rf %s" % (abs_path_file_name, file_path, abs_path_file_name))
        # os.system("tar xvf %s -C %s; rm -rf %s" % (abs_path_file_name, file_path, abs_path_file_name))

        tmp_cp_file_list = sorted(glob.glob(file_path + "*"), key=lambda x: os.path.getmtime(x))
        tmp_cp_file_list.reverse()
        cat_cp_file_list = tmp_cp_file_list[SIZE_MODELPOOL:]
        for file_name in cat_cp_file_list:
            os.system("rm -r %s " % file_name)


class ModelDeliverManager():

    @staticmethod
    def run_rps_proxy():
        rpsproxy = ReliaPSProxy(
            yaml_rpssvr.deliver_model.server.url_ftend, 
            yaml_rpssvr.deliver_model.server.url_bkend, 
            flag_monitor=True if yaml_rpssvr.deliver_model.server.flag_monitor == 'true' else False,)
        rpsproxy.run()
    
    @staticmethod
    def model_send():
        model_send_mgr = ModelSendManager(yaml_rpssvr.deliver_model.publisher.url_ftend)
        model_send_mgr.run(1, True, 1)
    
    @staticmethod
    def model_recv():
        model_recv_mgr(i=1, url_bkend=yaml_rpssvr.deliver_model.subscriber.url_bkend)


if __name__ == '__main__':
    psproxy = ReliaPSProxy("ipc://rps_ftend.ipc", "ipc://rps_bkend.ipc", flag_monitor=True, mode_ftend='tcp', mode_bkend='tcp')
    psproxy.run()

    model_recv_mgr(i=1, url_bkend="ipc://rps_bkend.ipc")

# if __name__ == '__main__':
    check_and_send = ModelSendManager('ipc://rps_ftend.ipc')
    check_and_send.run(1, True, 1)
    


    
