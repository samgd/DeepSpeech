#!/bin/bash
STORE='/home/ubuntu/efs/'
EXPERIMENT='0-epoch_basic_full-precision/'
LOG_DIR=$STORE$EXPERIMENT
TF_DIR=$LOG_DIR'tf_dir'
OUT_LOG=$LOG_DIR'out_log.txt'

mkdir $LOG_DIR
mkdir $TF_DIR
export COMPUTE_KEEP_DIR=$TF_DIR
./test.sh 2>&1 | tee $OUT_LOG
touch $LOG_DIR'finished'
