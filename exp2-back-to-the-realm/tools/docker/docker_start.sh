#!/bin/bash

# 机器上已经启动的容器, 因为机器被停止再重新启动了后启动容器的脚本

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/docker_start.sh name name \033[0m"
    exit -1
fi

name=$1

docker ps -a | grep $name | awk '{print $1}' | xargs -I {} docker start {}
