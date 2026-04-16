#!/bin/bash

# 对于批量启动的容器, 执行命令

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 2 ];
then
    echo -e "\033[31m useage: sh tools/docker_execute_cmd.sh name cmd \033[0m"
    exit -1
fi

name=$1
cmd=$2

for container_id in $(docker ps -a | grep $name | awk '{print $1}');
do
  docker exec $container_id /bin/sh -c "$cmd"
done
