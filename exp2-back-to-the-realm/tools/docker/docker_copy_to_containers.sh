#!/bin/bash

# 对于批量启动的容器, 执行命令

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 3 ];
then
    echo -e "\033[31m useage: sh tools/docker_copy_to_containers.sh name file_to_copy destination_path \033[0m"
    exit -1
fi

name=$1
file_to_copy=$2
destination_path=$3

# 获取所有名为 kaiwudrl_new_ 的容器 ID
container_ids=$(docker ps -a | grep $name | awk '{print $1}')

# 循环遍历容器 ID 并复制文件
for container_id in $container_ids
do
    docker cp "$file_to_copy" "$container_id":"$destination_path"
done

echo "文件已复制到所有容器。"
