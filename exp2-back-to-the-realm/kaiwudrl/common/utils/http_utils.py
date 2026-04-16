#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file http_utils.py
# @brief
# @author kaiwu
# @date 2023-11-28


# need install urllib3
import json
import traceback
from urllib3 import *
import warnings

warnings.simplefilter("ignore", ResourceWarning)


# http request请求
def http_utils_request(url, fields=None, print_error_msg=True):
    """
    通用 HTTP POST请求
    :param url: 请求地址
    :param fields: 字段
    :param print_error_msg: 是否打印错误信息
    :return: 返回响应数据
    """
    if not url:
        return None

    data = None
    try:
        http = PoolManager(timeout=Timeout(connect=10.0, read=10.0))

        if fields:
            r = http.request("GET", url, fields=fields)
        else:
            r = http.request("GET", url)

        # 提前判断status的值
        if r.status != 200:
            return None

        # 如果是json数据格式, 则返回json数据; 否则返回原始数据
        try:
            data = json.loads(r.data.decode("utf-8"), strict=False)
        except json.JSONDecodeError:
            data = r.data.decode("utf-8")

    # urllib3.exceptions的比较多, 故采用Exception作为兜底的, 并且不做处理, finally返回data
    except Exception as e:
        if print_error_msg:
            print(f"http_utils_request error as {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
        else:
            pass
    finally:
        return data


# http post请求, fields支持json格式
def http_utils_post(url, fields=None, print_error_msg=True):
    """
    通用 HTTP POST请求
    :param url: 请求地址
    :param fields: 字段
    :param print_error_msg: 是否打印错误信息
    :return: 返回响应数据
    """
    if not url:
        return None

    data = None
    try:
        http = PoolManager(timeout=Timeout(connect=10.0, read=10.0))

        if fields:
            encode_data = json.dumps(fields).encode("utf-8")
            r = http.request(
                "POST",
                url,
                body=encode_data,
                headers={"Content-Type": "application/json"},
            )
        else:
            r = http.request("POST", url)

        # 提前判断status的值
        if r.status != 200:
            return None

        # 如果是json数据格式, 则返回json数据; 否则返回原始数据
        try:
            data = json.loads(r.data.decode("utf-8"), strict=False)
        except json.JSONDecodeError:
            data = r.data.decode("utf-8")

    # urllib3.exceptions的比较多, 故采用Exception作为兜底的, 并且不做处理, finally返回data
    except Exception as e:
        if print_error_msg:
            print(f"http_utils_request error as {str(e)}, traceback.print_exc() is {traceback.format_exc()}")
        else:
            pass
    finally:
        return data


# http delete请求
def http_utils_delete(url, auth=None, print_error_msg=True):
    """
    通用 HTTP DELETE请求
    :param url: 请求地址
    :param auth: 认证信息 (username, password)
    :param print_error_msg: 是否打印错误信息
    :return: 返回操作是否成功 (True/False)
    """
    if not url:
        return 404, False

    headers = {}
    # 处理 Basic Auth
    if auth and isinstance(auth, tuple) and len(auth) == 2:
        from base64 import b64encode

        username, password = auth
        auth_header = b64encode(f"{username}:{password}".encode()).decode("utf-8")
        headers["Authorization"] = f"Basic {auth_header}"

    try:
        http = PoolManager(timeout=Timeout(connect=10.0, read=10.0))

        # 发送请求
        response = http.request("DELETE", url=url, fields=None, body=None, headers=headers)

        # DELETE 操作成功判断, 204 No Content 是常见成功状态
        return response.status, response.status in [200, 202, 204]

    except Exception as e:
        if print_error_msg:
            print(f"http_utils_delete error: {str(e)}\n{traceback.format_exc()}")
        return 404, False
