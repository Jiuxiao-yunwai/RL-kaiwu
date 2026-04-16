#!/bin/bash

# 直接将容器打包成镜像

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 2 ];
then
    echo -e "\033[31m useage: sh tools/docker_commit.sh container_id image_id\033[0m"
    exit -1
fi

container_id=$1
image_id=$2

docker commit  $container_id $image_id
