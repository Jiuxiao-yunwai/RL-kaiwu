#!/bin/bash

if [ $# -lt 1 ];then
echo "usage $0 master_ip"
exit -1
fi

#MODELPOOL_ADDR=$1
MODELPOOL_ADDR=$1":10013"

# 采用hostname -i在有些场景下获取的结果是::1 127.0.0.1 127.0.0.1这种输出, 故需要调整获取方法
# ip=`hostname -i`
ip=`ip addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '127.0.0.1' | head -n 1`
TVMEC_DOCKER_ID=`hostname`
CLUSTER_CONTEXT='default'

cd ../config && rm trpc_go.yaml
cd ../config && cp trpc_go.yaml.cpu trpc_go.yaml

sed -i "s/__TARGET_TRPC_ADDRESS_HERE__/${MODELPOOL_ADDR}/g" ../config/trpc_go.yaml
sed -i "s/__MODELPOOL_CLUSTER_HERE__/${CLUSTER_CONTEXT}/g" ../config/trpc_go.yaml
sed -i "s/__MODELPOOL_IP_HERE__/${ip}/g" ../config/trpc_go.yaml
sed -i "s/__MODELPOOL_NAME_HERE__/${TVMEC_DOCKER_ID}/g" ../config/trpc_go.yaml
