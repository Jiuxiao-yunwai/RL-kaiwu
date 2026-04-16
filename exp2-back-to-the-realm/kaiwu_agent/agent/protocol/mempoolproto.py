import numpy as np
import logging.config
import struct
import socket
import lz4.block
from enum import Enum

class CmdType(Enum):
    KMemSetBatchRequest = 1000001
    KMemGetRequest = 2000000
    KMemGetBatchRequest = 2000001
    KMemGetBatchCompressRequest = 2000002


class MemPoolProtocol():
    def __init__(self):
        pass

    def format_batch_samples_array(self, samples, priorities=None, max_sample_num=128):
        # 如果传入数据是b''组成的列表，无需序列化
        samples = self._serialization(samples)
        if priorities is None:
            priorities = list([0.0] * len(samples))

        # 0.split big list to spec-size sub-list
        start_idx = 0
        send_samples = []
        while (start_idx < len(samples)):
            sample_num = min(len(samples) - start_idx, max_sample_num)
            send_sample = self.format_set_batch_request(
                samples[start_idx:start_idx + sample_num], 
                priorities[start_idx:start_idx + sample_num])
            send_samples.append(send_sample)
            start_idx = start_idx + sample_num
        return send_samples
    
    def format_set_batch_request(self, samples, priorities=None):
        if priorities is None:
            priorities = list([0.0] * len(samples))

        KMemSetBatchRequest = int(CmdType.KMemSetBatchRequest.value)

        # 1.compress each sample
        samples = self._compress_sample(samples)

        # 2.package samples
        sample_str = b""
        for frame_idx in range(0, len(samples)):
            sample = samples[frame_idx]
            sample_len = len(sample)
            priority = priorities[frame_idx]
            sample_str += struct.pack("<I", int(sample_len)) + \
                            struct.pack("<f", float(priority)) + \
                            struct.pack("<%ss" % (sample_len), sample)

        # 3.header info
        # total, seq, cmd, num, data
        total_len = 4 + 4 + 4 + 4 + int(len(sample_str))
        seq_no = 0
        sample_num = len(samples)
        #print ("sample num %s sample_str %s total_len %s" %(sample_num, len(sample_str), total_len))

        return struct.pack("<I", socket.htonl(total_len)) \
                        + struct.pack("<I", int(seq_no)) \
                        + struct.pack("<I", int(KMemSetBatchRequest)) \
                        + struct.pack("<I", int(sample_num)) \
                        + sample_str

    def generate_samples(self, data):
        # 3.header info
        sample_list = []
        has_deal = 12
        sample_num = struct.unpack("I", data[has_deal:has_deal+4])[0]
        has_deal += 4

        # 2.unpackage samples
        for _ in range(sample_num):
            data_len = struct.unpack("I", data[has_deal:has_deal+4])[0]
            has_deal += 4
            priority = struct.unpack("f", data[has_deal:has_deal+4])[0]
            has_deal += 4

            sample = data[has_deal:has_deal+data_len]
            # 1. decompress each sample
            decompress_data = self._decompress_sample(sample)
            # _deserialization
            deserialization_data = self._deserialization(decompress_data)
            sample_list.append(deserialization_data)
            
            # 准备处理下一个sample
            has_deal += data_len

        return sample_list
    
    def _compress_sample(self, samples):
        compress_samples = []
        for sample in samples:
            if isinstance(sample, str):
                sample = bytes(sample, encoding='utf8')
            if not isinstance(sample, bytes):
                return None

            compress_sample = lz4.block.compress(sample, store_size=False)
            compress_samples.append(compress_sample)
        return compress_samples
    
    def _decompress_sample(self, sample):
        if isinstance(sample, str):
            sample = bytes(sample, encoding='utf8')
        if not isinstance(sample, bytes):
            return None

        return lz4.block.decompress(sample, uncompressed_size=3*1024*1024)
    
    def _serialization(self, list_samples):
        seri_samples = []
        for sample in list_samples:
            seri_samples.append(sample.tobytes())
        return seri_samples

    def _deserialization(self, receive_data):
        return np.frombuffer(receive_data, 'f4')   # f4对应dtype=np.float32 


if __name__ == "__main__":
    logging.config.dictConfig(logging_conf)
    logger = logging.getLogger(__name__)    

    samples = [np.ones(shape=(15552, ), dtype=np.float32) * (i+1) for i in range(300)]
    protocol = MemPoolProtocol()
    format_samples = protocol.format_batch_samples_array(samples, priorities=None, max_sample_num=128)
    for format_sample in format_samples:
        ret = protocol.generate_samples(format_sample)
        print(ret[0], len(ret[1]))