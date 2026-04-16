#!/bin/bash

# eval模式下, 一键修改配置的功能

chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 2 ];
then
    echo -e "\033[31m usage: sh tools/eval_model_dir_change.sh eval_model_dir eval_model_id such as: sh tools/eval_model_dir_change.sh /data/projects/kaiwu-fwk/ckpt/ 0 \033[0m"
    exit -1
fi

configure_file=conf/configure_app.toml
eval_model_dir=$1
eval_model_id=$2

sed -i 's/run_mode = .*/run_mode = "eval"/g' $configure_file
sed -i "s|eval_model_dir = .*|eval_model_dir = \"$eval_model_dir\"|g" $configure_file
sed -i "s/eval_model_id = .*/eval_model_id = $eval_model_id/g" $configure_file

judge_succ_or_fail $? "$configure_file change eval_model_dir $eval_model_dir eval_model_id $eval_model_id"
