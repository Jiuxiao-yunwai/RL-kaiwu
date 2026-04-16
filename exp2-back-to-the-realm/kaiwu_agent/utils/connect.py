
import time
import requests
import socket
import json
from io import BytesIO
from http.server import SimpleHTTPRequestHandler
from kaiwu_agent.utils.logging import ArenaLogger
from kaiwu_agent.conf import yaml_http_ctrl as GW_CONFIG


class ProxyHttpController:
    """
    ProxyHttpController通过http请求给ugc_game_core_server启动/停止gamecore
    """
    def __init__(self):
        self.game_core_server_endpoint = f"{GW_CONFIG.skylarena.client.endpoint}:{GW_CONFIG.skylarena.client.port}"
        self.retry_times = GW_CONFIG.skylarena.client["retry_times"]
        self.retry_times_sleep_seconds = GW_CONFIG.skylarena.client["retry_sleep_seconds"]
        self.logger = ArenaLogger()

    def start_game(self, new_game_req):
        """
            Description: 发送http请求给ugc_game_core_server启动gamecore
            ----------

            Return: 成功返回True, 重试超时失败返回False
            ----------
        """
        endpoint_url = f"http://{self.game_core_server_endpoint}/ugc/newGame"

        try:
            # gamecore确保返回ok时是一定启动成功的, 增加重试次数
            resp = requests.post(url=endpoint_url, json=new_game_req)
            retry_times = 0
            while retry_times < self.retry_times and resp.status_code != 200:
                resp = requests.post(url=endpoint_url, json=new_game_req)
                time.sleep(self.retry_times_sleep_seconds)
                retry_times += 1

            if retry_times >= self.retry_times:
                self.logger.error(f"start_game response code: {resp.status_code}, response content: {resp.content}")
                return False

            self.logger.debug(f'Succeed to start Game using SGame.exe, new_game_req: {new_game_req}')
            return True

        except ConnectionResetError as ex:
            self.logger.error(
                f'failed to start Game using SGame.exe, new_game_req: {new_game_req}, error is {str(ex)}')
            return False

    def stop_game(self, stop_game_req):
        """
            Description: 发送http请求给ugc_game_core_server停止gamecore
            ----------

            Return: 成功返回True, 重试超时失败返回False
            ----------
        """
        endpoint_url = f"http://{self.game_core_server_endpoint}/ugc/stopGame"

        try:
            resp = requests.post(url=endpoint_url, json=stop_game_req)
            retry_times = 0
            while retry_times < self.retry_times and resp.status_code != 200:
                resp = requests.post(url=endpoint_url, json=stop_game_req)
                time.sleep(self.retry_times_sleep_seconds)
                retry_times += 1

            if retry_times >= self.retry_times:
                self.logger.error(f"stop_game response code: {resp.status_code}, response content: {resp.content}")
                return False

            self.logger.info(f'Succeed to stop Game using SGame.exe, stop_game_req: {stop_game_req}')
            return True

        except Exception as ex:
            self.logger.error(
                f'failed to stop Game using SGame.exe, stop_game_req: {stop_game_req}, error is {str(ex)}')
            return False
    
class SkylarenaHttpController:
    def __init__(self) -> None:
        server_address = (GW_CONFIG.skylarena.server.host, GW_CONFIG.skylarena.server.port)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(server_address)
        self.server_socket.listen(1)
        print(f'Listening on port {GW_CONFIG.skylarena.server.host} {GW_CONFIG.skylarena.server.port}...')
        
    @staticmethod
    def __handle_request(request):
        # 处理 HTTP 请求的函数
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((GW_CONFIG.entity.client.host, GW_CONFIG.entity.client.port))
            client_socket.sendall(request)
            response = client_socket.recv(1024)
        finally:
            client_socket.shutdown(socket.SHUT_RDWR)
            client_socket.close()
            return response

    def run_once(self):
        try:
            client_socket, client_address = self.server_socket.accept()
            request = client_socket.recv(1024)
            response = SkylarenaHttpController.__handle_request(request)
            client_socket.sendall(response)
        finally:
            client_socket.shutdown(socket.SHUT_RDWR)
            client_socket.close()


class CustomRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, request_text):
        self.rfile = BytesIO(request_text)
        self.raw_requestline = self.rfile.readline()
        self.error_code = self.error_message = None
        self.parse_request()

class EntityHttpController:
    def __init__(self) -> None:
        server_address = (GW_CONFIG.entity.server.host, GW_CONFIG.entity.server.port)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(server_address)
        self.server_socket.listen(1)
        print(f'Listening on port {GW_CONFIG.entity.server.host} {GW_CONFIG.entity.server.port}...')

    @staticmethod
    def __handle_request(request):
        # 处理 HTTP 请求的函数
        request_handler = CustomRequestHandler(request)

        # 获取请求行
        method = request_handler.command
        path = request_handler.path
        # print(f"Method: {method}, Path: {path}")

        # 获取 JSON 数据
        content_length = int(request_handler.headers.get("Content-Length"))
        try:
            json_data = json.loads(request[-content_length:])
        except json.decoder.JSONDecodeError as e:
            json_data = {}
        # print("JSON data:", json_data)

        response = b'HTTP/1.1 200 OK\r\nContent-Length: 13\r\n\r\nsuccess!'
        return response, json_data

    def run_once(self):
        try: 
            client_socket, client_address = self.server_socket.accept()
            request = client_socket.recv(1024)
            response, json_data = EntityHttpController.__handle_request(request)
            client_socket.sendall(response)
        finally:
            client_socket.shutdown(socket.SHUT_RDWR)
            client_socket.close()
            return json_data

    

