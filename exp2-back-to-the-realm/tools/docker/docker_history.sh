#!/bin/bash

# 分析Docker镜像的历史记录，显示构建镜像时每个层的详细信息

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/docker_history.sh [image_id]\033[0m"
    exit -1
fi

image_id=$1

docker history $image_id