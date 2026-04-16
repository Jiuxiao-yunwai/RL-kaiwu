#!/bin/bash

# 采用hostname -i在有些场景下获取的结果是::1 127.0.0.1 127.0.0.1这种输出, 故需要调整获取方法
# ip=`hostname -i`
ip=`ip addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '127.0.0.1' | head -n 1`

cd ../config && rm trpc_go.yaml
cd ../config && cp trpc_go.yaml.gpu trpc_go.yaml

sed -i "s/__MODELPOOL_IP_HERE__/${ip}/g" ../config/trpc_go.yaml
