from pyexpat import model
import numpy as np
import struct
import socket
import lz4.block
from enum import Enum

class CmdType(Enum):
    KModelPubRequest = 3000001


class ModelProtocol():
    def __init__(self):
        pass

    def model_encode(self, s_key, s_md5, b_model):

        KModelPubRequest = int(CmdType.KModelPubRequest.value)
        
        model_str = b""
        b_key = s_key.encode('utf-8')
        model_str += struct.pack("<I", int(len(b_key))) + \
                        struct.pack("<%ss" % (len(b_key)), b_key)
        b_md5 = s_md5.encode('utf-8')
        model_str += struct.pack("<I", int(len(b_md5))) + \
                        struct.pack("<%ss" % (len(b_md5)), b_md5)
        model_str += struct.pack("<I", int(len(b_model))) + \
                        struct.pack("<%ss" % (len(b_model)), b_model)
        

        # 3.header info
        # total, seq, cmd, num, data
        total_len = 4 + 4 + 4 + 4 + int(len(model_str))
        seq_no = 0
        num_item = 3

        return struct.pack("<I", socket.htonl(total_len)) \
                        + struct.pack("<I", int(seq_no)) \
                        + struct.pack("<I", int(KModelPubRequest)) \
                        + struct.pack("<I", int(num_item)) \
                        + model_str

    def model_decode(self, data):
        # 3.header info
        has_deal = 12
        num_item = struct.unpack("I", data[has_deal:has_deal+4])[0]
        has_deal += 4

        # 2.unpackage models
        data_len = struct.unpack("I", data[has_deal:has_deal+4])[0]
        has_deal += 4
        b_key = data[has_deal:has_deal+data_len]
        s_key = b_key.decode('utf-8')

        has_deal += data_len
        data_len = struct.unpack("I", data[has_deal:has_deal+4])[0]
        has_deal += 4
        b_md5 = data[has_deal:has_deal+data_len]
        s_md5 = b_md5.decode('utf-8')

        has_deal += data_len
        data_len = struct.unpack("I", data[has_deal:has_deal+4])[0]
        has_deal += 4
        s_model = data[has_deal:has_deal+data_len]
            
        return s_key, s_md5, s_model

    
if __name__ == "__main__":

    protocol = ModelProtocol()
    b_model = protocol.model_encode('checkpoints_20220727_213615.257571.tar', 'd266890e56d87b75ab2f1de9804fd5f9', b'model_data')
    s_model = protocol.model_decode(b_model)
    print(s_model)