# encoding: utf-8
"""
Helper module for example applications. Mimics ZeroMQ Guide's zhelpers.h.
"""
from __future__ import print_function

import binascii
import os
from random import randint

import zmq

def socket_set_hwm(socket, hwm=-1):
    """libzmq 2/3/4 compatible sethwm"""
    try:
        socket.sndhwm = socket.rcvhwm = hwm
    except AttributeError:
        socket.hwm = hwm


def dump(msg_or_socket):
    """Receives all message parts from socket, printing each frame neatly"""
    if isinstance(msg_or_socket, zmq.Socket):
        # it's a socket, call on current message
        msg = msg_or_socket.recv_multipart()
    else:
        msg = msg_or_socket
    print("----------------------------------------")
    for part in msg:
        print("[%03d]" % len(part), end=' ')
        is_text = True
        try:
            print(part.decode('ascii'))
        except UnicodeDecodeError:
            print(r"0x%s" % (binascii.hexlify(part).decode('ascii')))


def set_id(zsocket):
    """Set simple random printable identity on socket"""
    identity = u"%04x-%04x" % (randint(0, 0x10000), randint(0, 0x10000))
    zsocket.setsockopt_string(zmq.IDENTITY, identity)


def zpipe(ctx):
    """build inproc pipe for talking to threads

    mimic pipe used in czmq zthread_fork.

    Returns a pair of PAIRs connected via inproc
    """
    a = ctx.socket(zmq.PAIR)
    b = ctx.socket(zmq.PAIR)
    a.linger = b.linger = 0
    a.hwm = b.hwm = 1
    iface = "inproc://%s" % binascii.hexlify(os.urandom(8))
    a.bind(iface)
    b.connect(iface)
    return a,b


def close_socket(skt):
    skt.setsockopt(zmq.LINGER, 0)
    skt.close()


CLOSE_SUB              = b"\x00"      # Signals worker is ready
CONNECT_SUB            = b"\x01"      # Signals worker is ready
TIME_IVL_PUB           = 2
HEARTBEAT_PUB          = b"\x02"
HEARTBEAT_IVL_PUB      = 1
HEARTBEAT_SUB          = b"\x03"
HEARTBEAT_IVL_SUB      = 1
TIME_IVL_DETECT        = 5
PS_MAX_ALLOWED_DELAY   = TIME_IVL_PUB + 0.1


READY_WORKER           = b"\x04"      # Signals worker is ready

HEARTBEAT_WORKER       = b"\x05"  # Signals worker heartbeat
HEARTBEAT_WORKER_KILL  = b"\x08"  # woker tell proxy suicide
HEARTBEAT_IVL_WORKER   = 1
HEARTBEAT_PROXY        = b"\x06"  # Signals worker heartbeat
HEARTBEAT_IVL_PROXY    = 1

DEFAULT_LIVENESS_WOKER = 3
DEFAULT_LIVENESS_PROXY = 3

SLEEP_TIME_REREQ = 1
SLEEP_TIME_REREP = 3


REGISTER_PUSH = b"\x01"
REGISTER_PULL = b"\x02"
TIMEOUT_PROXY = 10

WORKER_FAULT_RESULT   = b"\x07"  # Signals worker heartbeat


SESS_START = b"\x1101"
SESS_UPDATE = b"\x1102"
SESS_STOP = b"\x1103"

B_SEQ_NO_0 = int(0).to_bytes(4, byteorder='big')

REQUEST_TIMEOUT = 1000
REQUEST_RETRIES = 10
RESPONSE_TIMEOUT = 1000
RESPONSE_RETRIES = 30
RESET_RETRIES = 100000

BATCH_SIZE = 2