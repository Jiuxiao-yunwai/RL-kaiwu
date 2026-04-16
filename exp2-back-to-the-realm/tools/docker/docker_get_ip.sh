#!/bin/bash

# 根据docker的ID获取对应的IP

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/docker_get_ip.sh docker_name \033[0m"
    exit -1
fi

docker_name=$1

docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $docker_name
