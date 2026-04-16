#!/bin/bash
# 利用此脚本生产数字签名的私钥和公钥, 注意是采用source命令, 这样父shell和子shell都能用到环境变量


chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 0 ] && [ $# -ne 1 ];
then
    echo -e "\033[31m useage: source tools/generate_private_key.sh target_dir, such as source tools/generate_private_key.sh \033[0m"

    exit -1
fi

target_dir=${1:-'./'}

python3 -c "
import sys
sys.path.append('./tools')
from generate_and_verify_private_key import generate_private_and_public_key
generate_private_and_public_key('$target_dir')
"

# 设置环境变量
export private_key_content=$(cat $target_dir/private_key.pem)
export public_key_content=$(cat $target_dir/public_key.pem)

judge_succ_or_fail $? "generate_private_and_public_key"
