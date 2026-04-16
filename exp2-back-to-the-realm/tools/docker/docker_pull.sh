#!/bin/bash

# 机器上docker pull镜像

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/docker_pull.sh mirror_url \033[0m"
    exit -1
fi

mirror_url=$1

docker pull $mirror_url
