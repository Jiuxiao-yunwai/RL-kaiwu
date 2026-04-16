#!/usr/bin/env python3
# -*- coding:utf-8 -*-


# @file tcp_util.py
# @brief
# @author kaiwu
# @date 2023-11-28

import socket
import threading


class TCPServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

    def start(self):
        try:
            while True:
                client_socket, addr = self.server_socket.accept()
                # 使用线程来处理每个客户端连接
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket, addr))
                client_thread.start()
        except KeyboardInterrupt:
            print("\nServer is shutting down...")
        finally:
            self.server_socket.close()

    def handle_client(self, client_socket, addr):
        try:
            # 业务逻辑处理类的实例
            client_handler = ClientHandler(client_socket)
            client_handler.run()
        except ConnectionResetError:
            print(f"Connection reset by {addr}")
        finally:
            client_socket.close()
            print(f"Connection closed with {addr}")


class ClientHandler:
    def __init__(self, client_socket):
        self.client_socket = client_socket

    def run(self):
        while True:
            data = self.recv()
            if not data:
                break
            print(f"Received: {data}")
            response = self.process_data(data)
            self.send(response)

    def send(self, data):
        self.client_socket.send(data.encode())

    def recv(self):
        data = self.client_socket.recv(1024)
        return data.decode() if data else None

    def process_data(self, data):
        # 这里是业务逻辑处理
        return f"Echo: {data}"


class TCPClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        try:
            self.client_socket.connect((self.host, self.port))
            print(f"Connected to server at {self.host}:{self.port}")
        except socket.error as e:
            print(f"Failed to connect to server: {e}")

    def send(self, message):
        try:
            self.client_socket.sendall(message.encode())
            print(f"Sent: {message}")
        except socket.error as e:
            print(f"Failed to send message: {e}")

    def receive(self):
        try:
            response = self.client_socket.recv(1024)
            print(f"Received: {response.decode()}")
            return response.decode()
        except socket.error as e:
            print(f"Failed to receive message: {e}")
            return None

    def close(self):
        self.client_socket.close()
        print("Connection closed.")
