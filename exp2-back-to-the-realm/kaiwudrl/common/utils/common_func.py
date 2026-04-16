#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file common_func.py
# @brief
# @author kaiwu
# @date 2023-11-28


import datetime
import shutil
import time
import hashlib
import os
import tarfile
import uuid
import socket
import re
import base64
import warnings

# need pip install schedule
import schedule
import lz4.block
import zstd
import subprocess
import psutil
import signal
import random
import json
import fcntl
import zipfile
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.common.config.config_control import CONFIG
from concurrent.futures import ThreadPoolExecutor

try:
    import _pickle as pickle
except ImportError:
    import pickle

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
import json
import chardet


# TimeIt
class TimeIt:
    def __enter__(self):
        self.start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        self.end = time.monotonic()
        self.interval = self.end - self.start


# Context
class Context:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(self, other):
        self.__dict__.update(other)


# ip:port
def str_to_addr(addr_str):
    fields = addr_str.split(":")
    assert len(fields) == 2, "addr_str format must be ip:port"

    return fields[0], int(fields[1])


# hash
def hashlib_md5(data):
    return hashlib.md5(data.encode(encoding="UTF-8")).hexdigest()


def md5sum(file_name):
    """
    读取某个文件, 前4096个字节计算下md5sum值
    """

    if not file_name:
        return ""

    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


# 获取GPU
def get_local_rank():
    local_rank = os.getenv("OMPI_COMM_WORLD_LOCAL_RANK", "0")
    return int(local_rank)


def get_local_size():
    local_size = os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE", "1")
    return int(local_size)


def get_uuid():
    """
    Python中没有基于DCE的, 所以uuid2可以忽略
    uuid4存在概率性重复, 由无映射性, 最好不用
    若在Global的分布式计算环境下, 最好用uuid1
    若有名字的唯一性要求,最好用uuid3或uuid5
    """

    uuid_value = uuid.uuid1()
    # 获取的数据控制在int32范围内(-2147483648, 2147483648), 即取字符串形式的后8位
    return int(str(uuid_value.int)[-8:])


def set_schedule_event(time_interval, run_func, op_gap="minutes"):
    """
    定时器, 支持按照间隔时间, 执行操作, 支持秒, 分钟, 小时, 天等单位
    """

    if op_gap == "minutes":
        schedule.every(time_interval).minutes.do(run_func)
    elif op_gap == "seconds":
        schedule.every(time_interval).seconds.do(run_func)
    elif op_gap == "hour":
        schedule.every(time_interval).hour.do(run_func)
    elif op_gap == "day":
        schedule.every(time_interval).day.do(run_func)
    else:
        # 需要按照需求, 添加功能
        pass


def get_first_last_line_from_file(file_name):
    """
    获取某个文件第一行和最后一行
    注意存在多个进程操作同一个文件的场景, 比如M个进程读操作N个进程写操作, 这样就需要加锁
    """

    first_line = None
    last_line = None

    if not os.path.exists(file_name):
        return first_line, last_line

    file_size = os.path.getsize(file_name)
    if file_size == 0:
        return first_line, last_line

    blocksize = 1024
    try:
        with open(file_name, "rb") as dat_file:
            # 获取文件锁
            fcntl.flock(dat_file.fileno(), fcntl.LOCK_SH)

            headers = dat_file.readline().strip()
            if file_size > blocksize:
                maxseekpoint = file_size // blocksize
                dat_file.seek(maxseekpoint * blocksize)
            else:
                maxseekpoint = 0
                dat_file.seek(0)
            lines = dat_file.readlines()
            if lines:
                last_line = lines[-1].strip()

                # 检查最后一行是否完整
                if last_line[-1:] != b"\n":
                    dat_file.seek(0, os.SEEK_END)
                    while dat_file.read(1) != b"\n":
                        dat_file.seek(-2, os.SEEK_CUR)
                    last_line = dat_file.readline().strip()

            # 释放文件锁
            fcntl.flock(dat_file.fileno(), fcntl.LOCK_UN)

    except FileNotFoundError:
        # 文件在读取过程中被删除
        return first_line, last_line
    except Exception as e:
        # 处理其他可能的异常
        return first_line, last_line

    return (
        headers.decode() if headers else None,
        last_line.decode() if last_line else None,
    )


def get_first_line_and_last_line_from_file(file_name):
    """
    获取文件首行和末行的内容, 实际运行起来在文件内容小于2行时报错
    """

    first_line = None
    last_line = None

    if not os.path.exists(file_name):
        return first_line, last_line

    with open(file_name, "rb") as f:  # 打开文件
        # 在文本文件中，没有使用b模式选项打开的文件，只允许从文件头开始,只能seek(offset,0)
        first_line = f.readline()  # 取第一行
        offset = -50  # 设置偏移量
        while True:
            """
            file.seek(off, whence=0), 从文件中移动off个操作标记(文件指针)，正往结束方向移动，负往开始方向移动。
            如果设定了whence参数,就以whence设定的起始位为准,0代表从头开始,1代表当前位置,2代表文件最末尾位置。
            """
            f.seek(offset, 2)  # seek(offset, 2)表示文件指针：从文件末尾(2)开始向前50个字节(-50)
            lines = f.readlines()  # 读取文件指针范围内所有行
            if len(lines) >= 2:  # 判断是否最后至少有两行，这样保证了最后一行是完整的
                last_line = lines[-1]  # 取最后一行
                break
            # 如果off为50时得到的readlines只有一行内容，那么不能保证最后一行是完整的
            # 所以off翻倍重新运行，直到readlines不止一行
            offset *= 2

    return first_line.decode(), last_line.decode()


def get_random_line_from_file(file_name):
    """
    获取文件里从第2行到文件末尾中的随机一行
    """

    random_line = None

    # 判断文件不存在则返回
    if not os.path.exists(file_name):
        return random_line

    with open(file_name, "r", encoding="utf-8") as f:
        # 获取文件总行数
        total_lines = sum(1 for _ in f)

        # 随机生成一个整数，表示要获取的行数
        line_num = random.randint(1, total_lines)

        # 重置文件指针
        f.seek(0)
        # 遍历文件，获取指定行的内容
        for i, line in enumerate(f):
            if i == line_num - 1:
                random_line = line.strip()

    # 如果文件为空或者n大于文件总行数，返回空字符串
    return random_line.decode()


def get_last_two_line_from_file(file_name):
    """
    获取文件最后2行
    """

    last_two_line = None

    # 判断文件不存在则返回
    if not os.path.exists(file_name):
        return last_two_line

    with open(file_name, "rb") as f:  # 打开文件
        f.seek(0, 0)
        lines = f.readlines()
        if len(lines) < 3:
            # 在文本文件中，没有使用b模式选项打开的文件，只允许从文件头开始,只能seek(offset,0)
            offset = -50  # 设置偏移量
            while True:
                """
                file.seek(off, whence=0), 从文件中移动off个操作标记(文件指针)，正往结束方向移动，负往开始方向移动。
                如果设定了whence参数,就以whence设定的起始位为准,0代表从头开始,1代表当前位置,2代表文件最末尾位置。
                """
                f.seek(offset, 2)  # seek(offset, 2)表示文件指针：从文件末尾(2)开始向前50个字节(-50)
                lines = f.readlines()  # 读取文件指针范围内所有行
                if len(lines) >= 2:  # 判断是否最后至少有两行，这样保证了最后一行是完整的
                    last_two_line = lines[-1]  # 取最后一行
                    break
                # 如果off为50时得到的readlines只有一行内容，那么不能保证最后一行是完整的
                # 所以off翻倍重新运行，直到readlines不止一行
                offset *= 2
            # 在文本文件中，没有使用b模式选项打开的文件，只允许从文件头开始,只能seek(offset,0)
            offset = -50  # 设置偏移量

        else:
            last_two_line = lines[-2]  # 取倒数第二行
    return last_two_line.decode()


# 加载旧模型需要修改checkpoint文件
def fix_checkpoint_file(ckpt_file, checkpoint_id):
    with open(ckpt_file, "rb") as f:
        lines = f.readlines()
    if len(lines) <= 2:
        return

    os.remove(ckpt_file)
    with open(ckpt_file, "wb") as f:
        for i, line in enumerate(lines):
            line = line.decode()
            if i == 0:
                old_checkpoint_id = line.split(f"{KaiwuDRLDefine.KAIWU_MODEL_CKPT}-")[1]
                old_checkpoint_id = re.findall(r"\d+\.?\d*", old_checkpoint_id)[0]

                line = line.replace(old_checkpoint_id, checkpoint_id)

            elif i == len(lines) - 1:
                break
            f.write(line.encode())


# tar包压缩
def make_tar_file(output_file_name, source_dir):
    if not output_file_name or not source_dir:
        return

    with tarfile.open(output_file_name, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))


# tar包解压, 注意参数传递, 比如有zip则参数为r:zg
def tar_file_extract(input_file_name, destination_dir):
    if not input_file_name or not destination_dir:
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            with tarfile.open(input_file_name, "r") as tar:
                # 安全检查：确保没有文件是绝对路径或包含父目录引用
                safe_files = [
                    member
                    for member in tar.getmembers()
                    if member.isreg() and not member.name.startswith("/") and ".." not in member.name
                ]
                for member in safe_files:
                    # 清除文件模式位
                    member.mode = member.mode & 0o777
                    tar.extract(member, destination_dir)
        except Exception as e:
            raise e


# zip包压缩
def make_zip_file(output_file_name, source_dir):
    if not output_file_name or not source_dir:
        return

    try:
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)  # 创建目标文件夹

        with zipfile.ZipFile(output_file_name, "w") as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, source_dir))

    except Exception as e:
        raise e


# zip包解压缩
def zip_file_extract(input_file_name, destination_dir):
    if not input_file_name or not destination_dir:
        return

    with zipfile.ZipFile(input_file_name, "r") as zipf:
        zipf.extractall(destination_dir)


# 清空某个文件夹, 采用先删除, 再新增文件夹的方法
def clean_dir(dir_path):
    if not dir_path:
        return

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)


# 创建文件夹, 目录存在则跳过
def make_single_dir(dir_path):
    if not dir_path:
        return

    folder = os.path.exists(dir_path)
    if not folder:
        os.makedirs(dir_path, exist_ok=True)


# 根据函数名, 注意传递的是函数不是字符串, 返回函数内容
def get_fun_content_by_name(fun_name):
    import inspect

    if not fun_name or not inspect.isfunction(fun_name):
        print("func_name is None or not function")
        return

    return inspect.getsource(fun_name)


def insert_any_string(old_str, to_insert_str, to_find_str, before_or_after):
    """
    在特定的字符串前或者后增加字符串
    old_str: 原有的字符串
    to_insert_str: 需要插入的字符串
    to_find_str: 查找的字符串
    before_or_after: 在查找字符串to_find_str前或者后插入to_insert_str
    """

    if not old_str or not to_insert_str or not to_find_str:
        return

    idx = old_str.find(to_find_str)
    if before_or_after == "before":
        final_string = old_str[:idx] + to_insert_str + old_str[idx:]
    else:
        pass

    return final_string


# python 获取本机IP
def get_host_ip():
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        st.connect(("10.255.255.255", 1))
        IP = st.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        st.close()

    return IP


# 判断2个list是否相等, 主要是先排序再判断
def is_list_eq(listA, listB):
    if not listA or not listB:
        return False

    listA.sort()
    listB.sort()

    return listA == listB


def list_diff(listA, listB):
    """
    2个list的差集
    返回的数据包括list_A_have_B_not_have, list_B_have_A_not_have
    """

    if not listA or not listB:
        return []

    # listB中有但是listA中没有
    list_A_have_B_not_have = list(set(listB).difference(set(listA)))
    # listA中有但是listB中没有
    list_B_have_A_not_have = list(set(listA).difference(set(listB)))

    return list_A_have_B_not_have, list_B_have_A_not_have


def get_any_time_from_now(delta, ways="day"):
    """
    获取当前时间的前后多少秒, 多少分钟, 前后多少小时, 前后多少天
    ways: 支持天、小时、分钟、秒
    delta: 差量, 如果是正则是当前时间往后, 如果是负则当前时间往前, 如果是0则就是当前时间
    """

    now_time = datetime.datetime.now()
    end_time = None

    if "day" == ways:
        end_time = now_time + datetime.timedelta(days=delta)
    elif "hour" == ways:
        end_time = now_time + datetime.timedelta(hours=delta)
    elif "min" == ways:
        end_time = now_time + datetime.timedelta(minutes=delta)
    elif "second" == ways:
        end_time = now_time + datetime.timedelta(seconds=delta)
    else:
        pass

    return end_time


def python_exec_shell(shell_content):
    """
    python 执行shell语句命令
    0表示执行成功, 非0表示执行失败
    """

    if not shell_content:
        return 1, None

    result_code, result_str = subprocess.getstatusoutput(shell_content)
    return result_code, result_str


# actor/learner增加来自aisrv的TCP连接数目统计
def actor_learner_aisrv_count(host, srv_name):
    if not srv_name or not host:
        return 0

    port = 0
    if KaiwuDRLDefine.SERVER_ACTOR == srv_name:
        port = CONFIG.zmq_server_port
    elif KaiwuDRLDefine.SERVER_LEARNER == srv_name:
        port = CONFIG.reverb_svr_port
    elif KaiwuDRLDefine.SERVER_AISRV == srv_name:
        port = CONFIG.aisrv_server_port
    else:
        pass

    # 建立连接的TCP数目
    cmd = f"ss -ano  | grep {host}:{port} | grep ESTAB | wc -l"

    result_code, result_str = python_exec_shell(cmd)
    if result_code != 0:
        return 0

    count = 0
    try:
        count = int(result_str)
    except Exception as e:
        print(f"result_str is {result_str}, exception is {str(e)}")

    return count


def get_gpu_machine_type_by_shell():
    """
    获取GPU机器的型号, 支持异构GPU

    shell命令返回的结果形如:
    GPU 0: GRID T4-8C (UUID: GPU-cdd63ee4-0be7-11ed-9562-0c1a84aad0c2)

    函数返回的结果形如:
    GRID T4
    """

    cmd = "nvidia-smi -L"

    result_code, result_str = python_exec_shell(cmd)

    # 解析结果
    if result_code or not result_str:
        return None

    try:
        gpu_machine_type = result_str.split("GPU 0:")[1].split("(")[0]
    except Exception as e:
        return None

    return gpu_machine_type


def get_gpu_machine_type():
    """
    1. get_gpu_machine_type_by_shell返回空时当CPU处理
    2. 其余看属于哪个GPU场景
    """

    gpu_machine_type = KaiwuDRLDefine.GPU_MACHINE_CPU

    gpu_machine_type_shell = get_gpu_machine_type_by_shell()
    if not gpu_machine_type_shell:
        pass
    else:
        if KaiwuDRLDefine.GPU_MACHINE_A100 in gpu_machine_type_shell:
            gpu_machine_type = KaiwuDRLDefine.GPU_MACHINE_A100
        elif KaiwuDRLDefine.GPU_MACHINE_V100 in gpu_machine_type_shell:
            gpu_machine_type = KaiwuDRLDefine.GPU_MACHINE_V100
        elif KaiwuDRLDefine.GPU_MACHINE_T4 in gpu_machine_type_shell:
            gpu_machine_type = KaiwuDRLDefine.GPU_MACHINE_T4
        else:
            pass

    return gpu_machine_type


def compress_data(data, serialize=True):
    """
    压缩, 主要是为了扩展, 以后方便多种算法
    流程:
    1. 进行序列化, 可选方法有pickle/protobuf
    2. 进行压缩, 可选方法有lz4/zstd
    """

    if not data:
        return data

    if not CONFIG.use_compress_decompress:
        return data

    try:
        # 采用pickle序列化
        if serialize:
            if CONFIG.aisrv_actor_protocol == KaiwuDRLDefine.PROTOCOL_PICKLE:
                data = pickle.dumps(data)
            elif CONFIG.aisrv_actor_protocol == KaiwuDRLDefine.PROTOCOL_PROTOBUF:
                data = data.SerializeToString()
            else:
                pass

        if CONFIG.compress_decompress_algorithms == KaiwuDRLDefine.COMPRESS_DECOMPRESS_ALGORITHMS_LZ4:
            compress_msg = lz4.block.compress(data, mode="fast", store_size=False)
        elif CONFIG.compress_decompress_algorithms == KaiwuDRLDefine.COMPRESS_DECOMPRESS_ALGORITHMS_ZSTD:
            # 设置为1时, 压缩耗时较小
            compress_msg = zstd.compress(data, 1)
        else:
            compress_msg = data

    # 失败场景下, 返回原始数据
    except Exception as e:
        print(f"compress_data error {str(e)}")
        compress_msg = data

    return compress_msg


def compress_data_parallel(datas, serialize=True):
    """
    开启线程来处理多个data, 每个线程调用compress_data
    """

    if not datas:
        return datas

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(lambda data: compress_data(data, serialize), datas)

    return list(results)


def decompress_data(data, deserialize=True, uncompressed_size=None):
    """
    解压缩, 主要是为了扩展, 以后方便多种算法
    流程:
    1. 进行反序列化, 可选方法有pickle/protobuf
    2. 进行解压缩, 可选方法有lz4/zstd
    """

    if not data:
        return data

    if not CONFIG.use_compress_decompress:
        return data

    try:
        if CONFIG.compress_decompress_algorithms == KaiwuDRLDefine.COMPRESS_DECOMPRESS_ALGORITHMS_LZ4:
            current_uncompressed_size = CONFIG.lz4_uncompressed_size
            if uncompressed_size:
                current_uncompressed_size = uncompressed_size
            decompress_msg = lz4.block.decompress(data, uncompressed_size=current_uncompressed_size)

        elif CONFIG.compress_decompress_algorithms == KaiwuDRLDefine.COMPRESS_DECOMPRESS_ALGORITHMS_ZSTD:
            decompress_msg = zstd.decompress(data)

        else:
            decompress_msg = data

        if deserialize:
            # 采用pickle反序列化
            if CONFIG.aisrv_actor_protocol == KaiwuDRLDefine.PROTOCOL_PICKLE:
                decompress_msg = pickle.loads(decompress_msg, encoding="bytes")
            # 采用protobuf序列化, 因为需要特定的对象才能反序列化故原样返回数据
            elif CONFIG.aisrv_actor_protocol == KaiwuDRLDefine.PROTOCOL_PROTOBUF:
                decompress_msg = decompress_msg
            else:
                pass

    except Exception as e:
        print(f"decompress_data error {str(e)}")
        decompress_msg = data

    return decompress_msg


# 开启线程来处理多个data, 每个线程调用decompress_data
def decompress_data_parallel(datas, deserialize=True):
    if not datas:
        return datas

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(lambda data: decompress_data(data, deserialize), datas)

    return list(results)


# CPU 绑核操作, 规避因为CPU调度引起的时延大问题
def cpu_affinity(pid, cpu_idx):
    # 如果设置的idx小于0, 则默认绑定在CPU 核1上
    if not isinstance(cpu_idx, list):
        cpu_idx = [cpu_idx]

    p = psutil.Process(pid)
    if not p:
        return False

    # 如果CPU列表为空, 意味着绑定到所有可用核上
    p.cpu_affinity(cpu_idx)
    return True


# 根据输入的list, 生成平均值和最大值
def get_mean_and_max(data):
    mean_value = 0
    max_value = 0

    if not data:
        return mean_value, max_value

    max_value = max(data)
    mean_value = sum(data) / len(data)

    return mean_value, max_value


# 获取文件大小
def get_file_size(file_path):
    if not file_path or not os.path.exists(file_path):
        return 0

    return os.path.getsize(file_path)


# 将一个字典写入文件
def write_json_to_file(data_dict, file_name, target_dir):
    with open(target_dir + "/" + file_name + ".json", "w") as json_file:
        json_file.write(json.dumps(data_dict))


def get_sort_file_list(dir_path, reverse):
    """
    对某个目录下的文件按照创建时间升序或者降序排序
    dir_path: 文件夹
    reverse: reverse = True 降序, reverse = False 升序（默认）
    """

    dir_list = os.listdir(dir_path)
    if not dir_list:
        return []

    dir_list = sorted(
        dir_list,
        key=lambda x: os.path.getmtime(os.path.join(dir_path, x)),
        reverse=reverse,
    )

    return dir_list


# 按照进程启动命令形式停掉进程
def stop_process_by_cmdline(cmdlines, not_to_kill_pid=-1):
    if not cmdlines:
        return

    # 按照进程启动命令形式, 采用遍历方式
    processes = psutil.process_iter()
    for process in processes:
        try:
            # 获取进程的命令行参数
            cmdline = process.cmdline()

            # 判断进程是否为目标进程
            if cmdlines in cmdline and process.pid != not_to_kill_pid:
                # 找到目标进程，杀死进程
                os.kill(process.pid, signal.SIGKILL)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # 忽略异常进程
            pass


# 按照进程名字停掉进程
def stop_process_by_name(process_name, not_to_kill_pid=-1):
    if not process_name:
        return

    # 根据进程名获取进程ID, 采用遍历方式
    pids = psutil.process_iter()
    for pid in pids:
        if pid.name() == process_name and pid.pid != not_to_kill_pid:
            try:
                os.kill(pid.pid, signal.SIGKILL)
            except OSError as e:
                print(f"process_name {process_name} pid {pid} not exist")


# 按照进程ID停掉进程
def stop_process_by_pid(pid_list):
    if not pid_list:
        return

    for pid in pid_list:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError as e:
            print(f" pid {pid} not exist")


# 根据进程的运行字符串来获取其pid
def find_pids_by_cmdline(keyword):
    pids = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            if keyword in " ".join(proc.info["cmdline"]):
                pids.append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    return pids


def replace_file_content(file, old_content, new_content):
    """
    替换文件内容, 步骤:
    1. 读取文件内容
    2. 替换文本
    3. 写回文件
    """

    if not file:
        return

    # 读文件内容
    def read_file(file):
        with open(file, "r", encoding="UTF-8") as f:
            read_all = f.read()

        return read_all

    # 写内容到文件
    def rewrite_file(file, data):
        with open(file, "w", encoding="UTF-8") as f:
            f.write(data)

    content = read_file(file)
    content = content.replace(old_content, new_content)
    rewrite_file(file, content)


# 随机生成某个区间的值
def get_random(start_idx, end_idx):
    return random.randint(start_idx, end_idx)


# 对一个map按照key排序后输出键值对, 格式为key1|value1|key2|value2|key3|value3|
def get_map_content(target_dict):
    if not target_dict or not isinstance(target_dict, dict):
        return None

    output = ""
    sorted_map = {k: target_dict[k] for k in sorted(target_dict.keys())}
    for key, value in sorted_map.items():
        output += f"{str(key)}|{str(value)}|"

    return output.encode("utf-8")


# 读取某个文件并且将这个文件的内容设置为环境变量
def set_env_variable(file_path, variable_name):
    if not file_path or not os.path.exists(file_path) or not variable_name:
        return False

    with open(file_path, "r") as file:
        file_content = file.read().strip()
        os.environ[variable_name] = file_content

    return True


def base64_encode(data) -> str:
    """
    base64编码
    """
    if not isinstance(data, bytes):
        return None

    return base64.b64encode(data).decode("utf-8")


def base64_decode(data) -> bytes:
    """
    base64解码
    """
    if not isinstance(data, str):
        return None
    try:
        return base64.b64decode(data)
    except Exception as e:
        return None


# 从private_key_content内容里加载私钥
def load_private_key_by_data(private_key_content):
    if not private_key_content:
        return

    # 将 PEM 格式的私钥反序列化为私钥对象
    private_key = serialization.load_pem_private_key(
        private_key_content.encode(), password=None, backend=default_backend()
    )

    return private_key


# 从private_key_content内容里加载私钥
def load_public_key_by_data(public_key_content):
    if not public_key_content:
        return

    # 将 PEM 格式的私钥反序列化为私钥对象
    public_key = serialization.load_pem_public_key(public_key_content.encode(), backend=default_backend())

    return public_key


# 从private_key_pem_file文件里加载私钥
def load_private_key_by_file(private_key_pem_file):
    if not private_key_pem_file or not os.path.exists(private_key_pem_file):
        return None

    # 从文件中读取 PEM 格式的私钥
    with open(private_key_pem_file, "rb") as f:
        pem_private_key = f.read()

    return load_private_key_by_data(pem_private_key)


# 从public_key_pem_file文件里加载公钥
def load_public_key_by_file(public_key_pem_file):
    if not public_key_pem_file or not os.path.exists(public_key_pem_file):
        return None

    # 从文件中读取 PEM 格式的私钥
    with open(public_key_pem_file, "rb") as f:
        pem_public_key = f.read()

    return load_public_key_by_data(pem_public_key)


def generate_private_key(target_dir="./"):
    """
    生成密钥对, 包括私钥和公钥, 私钥放在private_key.pem文件里, 公钥放在public_key.pem文件里
    target_dir目标目录, 默认为当前目录
    """

    # 设置相同的随机数生成器种子
    private_key = rsa.generate_private_key(
        public_exponent=CONFIG.public_exponent,
        key_size=CONFIG.key_size,
        backend=default_backend(),
    )

    """
    处理私钥步骤:
    1. 将私钥序列化为 PEM 格式
    2. 将PEM格式的私钥写入文件
    """
    pem_private_key = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    with open(f"{target_dir}/private_key.pem", "wb") as f:
        f.write(pem_private_key)

    """
    处理公钥步骤:
    1. 将公钥序列化为 PEM 格式
    2. 将PEM格式的公钥写入文件
    """
    public_key = private_key.public_key()
    pem_public_key = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    with open(f"{target_dir}/public_key.pem", "wb") as f:
        f.write(pem_public_key)

    return private_key, public_key


def generate_signature_by_data_common(data, key):
    """
    生成数字签名, data为数据, key为公钥或者私钥, 返回的公钥的签名或者私钥的签名, 其中:
    1. key为公钥, 返回的是公钥的签名
    2. key为私钥, 返回的是私钥的签名
    """
    if not data or not key:
        return None

    # 生成签名
    signature = key.sign(
        data,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )

    return signature


def generate_public_signature_by_data(data, public_key):
    """
    针对小数据量生成数字签名, data为数据, 返回的是公钥
    1. key为公钥, 返回的是公钥的签名
    2. key为私钥, 返回的是私钥的签名
    """
    return generate_signature_by_data_common(data, public_key)


def generate_private_signature_by_data(data, private_key):
    """
    针对小数据量生成数字签名, data为数据, 返回的是私钥
    """
    return generate_signature_by_data_common(data, private_key)


def generate_signature_by_dir_common(dir_path, key):
    """
    针对文件夹生成数字签名, dir_path为文件目录, 返回的是公钥
    """
    if not dir_path or not key:
        return None

    hasher = hashes.Hash(hashes.SHA256())
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, "rb") as f:
                buf = f.read(1024)  # 读取文件的一部分
                while buf:  # 当文件没有结束时
                    hasher.update(buf)  # 更新哈希值
                    buf = f.read(1024)  # 读取文件的下一部分

    # 生成签名
    signature = key.sign(
        hasher.finalize(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )

    return signature


def generate_public_signature_by_dir(dir_path, private_key):
    """
    针对文件夹生成数字签名, dir_path为文件目录, 返回的是公钥签名
    """
    return generate_signature_by_dir_common(dir_path, public_key)


def generate_private_signature_by_dir(dir_path, private_key):
    """
    针对文件夹生成数字签名, dir_path为文件目录, 返回的是私钥签名
    """
    return generate_signature_by_dir_common(dir_path, private_key)


def compute_file_hash(file_path):
    """
    计算文件的hash值
    """
    if not file_path:
        return None

    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def compute_directory_hash(dir_path, exclude_files=None):
    """
    计算文件夹下的hash值
    """
    if not dir_path:
        return None

    sha256_hash = hashlib.sha256()
    model_files = []

    # 遍历目录下的所有文件
    for root, dirs, files in os.walk(dir_path):
        for file in files:

            # 如果这个文件是包含了model.ckpt的则需要返回
            if KaiwuDRLDefine.KAIWU_MODEL_CKPT in file:
                model_files.append(file)

            # 如果文件名在排除列表中，跳过
            if exclude_files and file in exclude_files:
                continue

            file_path = os.path.join(root, file)

            # 计算每个文件的哈希值
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)

    return sha256_hash.hexdigest(), model_files


def signature_verify_by_data(key, data, signature):
    """
    针对数据数字签名校验, 其中:
    1. key是私钥, signature是公钥签名
    2. key是公钥, signature是私钥签名
    """
    if not key or not data or not signature:
        return False

    try:
        key.verify(
            signature,
            data,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return True

    except InvalidSignature:
        return False


def private_signature_verify_by_data(public_key, data, private_signature):
    """
    针对数据数字签名校验, public_key是公钥, private_signature是私钥签名
    """
    return signature_verify_by_data(public_key, data, private_signature)


def public_signature_verify_by_data(private_key, data, public_signature):
    """
    针对数据数字签名校验, private_key是私钥, public_signature是公钥签名
    """
    return signature_verify_by_data(private_key, data, public_signature)


def public_signature_verify_by_dir(public_key, data, private_signature):
    """
    针对文件夹数字签名校验, public_key是公钥, private_signature是私钥签名
    """
    return signature_verify_by_data(public_key, data, private_signature)


def private_signature_verify_by_dir(private_key, data, public_signature):
    """
    针对文件夹数字签名校验, private_key是私钥, public_signature是公钥签名
    """
    return signature_verify_by_data(private_key, data, public_signature)


def get_machine_device_by_config(framework_name, process_name):
    """
    根据强化学习框架类型和进程类型来获取对应的device
    """
    if not framework_name or not process_name:
        return None

    if framework_name == KaiwuDRLDefine.MODEL_PYTORCH:
        from kaiwudrl.common.utils.torch_utils import get_machine_device_by_config

        return get_machine_device_by_config(process_name)

    elif (
        framework_name == KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX
        or framework_name == KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE
    ):

        return None

    elif framework_name == KaiwuDRLDefine.MODEL_TCNN:
        return None

    elif framework_name == KaiwuDRLDefine.MODEL_TENSORRT:
        return None

    else:
        return None


def check_machine_device_valid(framework_name, device_name):
    """
    检测对应的device容器是否支持
    """
    if not framework_name or not device_name:
        return False

    if framework_name == KaiwuDRLDefine.MODEL_PYTORCH:
        from kaiwudrl.common.utils.torch_utils import (
            torch_is_gpu_available,
            torch_is_npu_available,
        )

        if device_name == KaiwuDRLDefine.MACHINE_DEVICE_CPU:
            return True
        elif device_name == KaiwuDRLDefine.MACHINE_DEVICE_GPU:
            return torch_is_gpu_available()
        elif device_name == KaiwuDRLDefine.MACHINE_DEVICE_NPU:
            return torch_is_npu_available()

        # 未来扩展
        else:
            return False

    elif (
        framework_name == KaiwuDRLDefine.MODEL_TENSORFLOW_COMPLEX
        or framework_name == KaiwuDRLDefine.MODEL_TENSORFLOW_SIMPLE
    ):
        from kaiwudrl.common.utils.tf_utils import (
            tensorflow_is_gpu_available,
            tensorflow_is_npu_available,
        )

        if device_name == KaiwuDRLDefine.MACHINE_DEVICE_CPU:
            return True
        elif device_name == KaiwuDRLDefine.MACHINE_DEVICE_GPU:
            return tensorflow_is_gpu_available()
        elif device_name == KaiwuDRLDefine.MACHINE_DEVICE_NPU:
            return tensorflow_is_npu_available()

        # 未来扩展
        else:
            return False

    elif framework_name == KaiwuDRLDefine.MODEL_TCNN:
        return True

    elif framework_name == KaiwuDRLDefine.MODEL_TENSORRT:
        return True

    else:
        return True


def machine_device_check(process_name):
    """
    检测强化学习配置使用的device使用是否合理, 规则如下:
    1. 如果配置的是CPU, 合理
    2. 如果配置的是GPU, 本机支持GPU, 合理
    3. 如果配置的是GPU, 本机不支持GPU, 不合理
    4. 如果配置的是NPU, 本机支持NPU, 合理
    5. 如果配置的是NPU, 本机不支持NPU, 不合理
    """
    if process_name not in [
        KaiwuDRLDefine.SERVER_ACTOR,
        KaiwuDRLDefine.SERVER_AISRV,
        KaiwuDRLDefine.SERVER_LEARNER,
    ]:
        return False

    # 配置的device
    configure_device = getattr(CONFIG, f"{process_name}_device_type")
    if configure_device == KaiwuDRLDefine.MACHINE_DEVICE_CPU:
        return True
    elif configure_device == KaiwuDRLDefine.MACHINE_DEVICE_GPU or configure_device == KaiwuDRLDefine.MACHINE_DEVICE_NPU:
        return check_machine_device_valid(CONFIG.use_which_deep_learning_framework, configure_device)

    # 未来扩展
    else:
        return True


def detect_encoding(filepath):
    with open(filepath, "rb") as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result["encoding"]


def scan_for_errors(directory, error_indicator="ERROR"):
    """
    扫描某个文件夹下的错误日志
    """
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                encoding = detect_encoding(filepath)
                with open(filepath, "r", encoding=encoding, errors="ignore") as file:
                    for line_number, line in enumerate(file, 1):
                        if error_indicator in line:
                            print(f"Error found in file: {filepath}")
                            print(f"Error content (line {line_number}): {line.strip()}")
                            return True
            except (UnicodeDecodeError, PermissionError) as e:
                print(f"Error reading file {filepath}: {e}")
                continue

    return False


def register_sigterm_handler(sigterm_handler, sigterm_pids_file):
    """
    注册SIGTERM信号处理函数和当前进程pid
    """
    signal.signal(signal.SIGTERM, sigterm_handler)
    with open(sigterm_pids_file, "a") as file:
        file.write(f"{os.getpid()} ")
