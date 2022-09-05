#!/bin/bash
source test_tipc/common_func.sh

FILENAME=$1
MODE=$2

# MODE be one of ['lite_train_lite_infer']          

dataline=$(cat ${FILENAME})


# parser params
IFS=$'\n'
lines=(${dataline})

# The training params
model_name=$(func_parser_value "${lines[1]}")

trainer_list=$(func_parser_value "${lines[14]}")

if [ ${MODE} = "lite_train_lite_infer" ];then
    # prepare lite data
    rm -rf ./test_tipc/data/CBSD68
    cd ./test_tipc/data/ && unzip CBSD68.zip && cd ../../
fi