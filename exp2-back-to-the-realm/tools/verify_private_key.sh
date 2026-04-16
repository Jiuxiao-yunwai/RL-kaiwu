#!/bin/bash
# 利用此脚本验证数字签名的正确性


chmod +x tools/common.sh
. tools/common.sh

if [ $# -ne 1 ];
then
    echo -e "\033[31m useage: sh tools/verify_private_key.sh zip_file_path,
    such as sh tools/verify_private_key.sh /workspace/train/backup_model/back_to_the_realm-uuid100-target_dqn-2999-2024_04_17_10_37_01-1.1.1.zip \033[0m"

    exit -1
fi

zip_file_path=$1

python3 -c "
import sys
sys.path.append('./tools')
from generate_and_verify_private_key import model_file_public_signature_verify
model_file_public_signature_verify('$zip_file_path')
"

judge_succ_or_fail $? "model_file_public_signature_verify"
