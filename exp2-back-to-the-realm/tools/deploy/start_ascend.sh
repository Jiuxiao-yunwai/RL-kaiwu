#!/bin/bash

# 机器上docker 启动华为的Ascend

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 2 ];
then
    echo -e "\033[31m useage: sh tools/start_ascend.sh name docker_image_id \033[0m"
    exit -1
fi

name=$1
docker_image_id=$2

# 华为的ascend机器上启动
docker run -it -u root \
        --name $name \
        --device=/dev/davinci2 \
        --device=/dev/davinci_manager \
        --device=/dev/devmm_svm \
        --device=/dev/hisi_hdc \
        -v /usr/local/dcmi:/usr/local/dcmi \
        -v /var/log/npu:/var/log/npu \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/slog:/usr/slog \
        -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
        -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
        -v /usr/local/Ascend/driver/tools/:/usr/local/Ascend/driver/tools/ \
        -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
        -v /data/projects:/root//data/projects \
        $docker_image_id \
        /bin/bash
