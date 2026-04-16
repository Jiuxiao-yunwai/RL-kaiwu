#!/bin/bash

# 机器上登录docker用户名和密码

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 3 ];
then
    echo -e "\033[31m useage: sh tools/docker_login.sh mirror_url user_name pass_word\033[0m"
    exit -1
fi

mirror_url=$1
user_name=$2
pass_word=$3

docker login $mirror_url --username $user_name --password $pass_word

