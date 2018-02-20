#!/bin/bash
STORE='/home/ubuntu/efs/'
EXPERIMENT='test/'
LOG_DIR=$STORE$EXPERIMENT
TF_DIR=$LOG_DIR'tf_dir'
TRAIN_LOG=$LOG_DIR'train_log.txt'
TEST_LOG=$LOG_DIR'test_log.txt'

mkdir $LOG_DIR
mkdir $TF_DIR
export COMPUTE_KEEP_DIR=$TF_DIR
./train.sh 2>&1 | tee $TRAIN_LOG
./test.sh 2>&1 | tee $TEST_LOG
touch $LOG_DIR'finished'
sudo shutdown
