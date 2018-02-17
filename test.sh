#!/bin/sh
set -xe

if [ ! -f DeepSpeech.py ]; then
    echo "Please make sure you run this from DeepSpeech's top level directory."
    exit 1
fi;

if [ ! -d "${COMPUTE_DATA_DIR}" ]; then
    COMPUTE_DATA_DIR="data/librivox"
fi;

# Warn if we can't find the train files
if [ ! -f "${COMPUTE_DATA_DIR}/librivox-train-clean-100.csv" ]; then
    echo "Warning: It looks like you don't have the LibriSpeech corpus"       \
         "downloaded and preprocessed. Make sure \$COMPUTE_DATA_DIR points to the" \
         "folder where the LibriSpeech data is located, and that you ran the" \
         "importer script at bin/import_librivox.py before running this script."
fi;

if [ -d "${COMPUTE_KEEP_DIR}" ]; then
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("deepspeech/librivox"))')
fi

#./bin/run-cluster.sh 1:1:1 \
python -u DeepSpeech.py \
  --notrain \
  --test \
  --train_files "$COMPUTE_DATA_DIR/librivox-train-clean-100.csv,$COMPUTE_DATA_DIR/librivox-train-clean-360.csv,$COMPUTE_DATA_DIR/librivox-train-other-500.csv" \
  --dev_files "$COMPUTE_DATA_DIR/librivox-dev-clean.csv" \
  --test_files "$COMPUTE_DATA_DIR/librivox-test-clean.csv" \
  --limit_test 64  \
  --test_batch_size 32 \
  --dropout_rate 0.0 \
  --n_hidden 2048 \
  --wer_log_pattern "GLOBAL LOG: logwer('${COMPUTE_ID}', '%s', '%s', %f)" \
  --lstm_type cudnn \
  --initialize_from_checkpoint "$HOME/Code/DeepSpeech/models/cudnn_checkpoint/model.ckpt-0" \
  --log_level 0 \
  --display_step 0 \
  "$@"
  #--train_batch_size 32 \
  #--dev_batch_size 24 \
  #--loss_scale 16 \
  #--early_stop 0 \
  #--epoch 0 \
  #--learning_rate 0.0001 \
  #--validation_step 1 \
  #--dropout_rate 0.2367 \
  #--default_stddev 0.046875 \
  #--half_precision \
  #--checkpoint_step 1 \
  #--checkpoint_dir "$checkpoint_dir" \
  #  --log_placement \
