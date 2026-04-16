#!/bin/bash

# 机器上docker 启动起来以后台方式执行

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 2 ];
then
    echo -e "\033[31m useage: sh tools/docker_start_demon.sh name docker_image_id \033[0m"
    exit -1
fi

name=$1
docker_image_id=$2

docker run -d -it -u root \
        --name $name \
        $docker_image_id \
        /bin/bash
