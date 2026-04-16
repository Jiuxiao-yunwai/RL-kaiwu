#!/bin/bash

# 查看镜像的详细信息，包括每个层的大小

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/docker_inspect.sh [image_id]\033[0m"
    exit -1
fi

image_id=$1

docker inspect $image_id