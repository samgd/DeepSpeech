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

python -u DeepSpeech.py \
  --train_files "$COMPUTE_DATA_DIR/librivox-train-clean-100.csv,$COMPUTE_DATA_DIR/librivox-train-clean-360.csv,$COMPUTE_DATA_DIR/librivox-train-other-500.csv" \
  --dev_files "$COMPUTE_DATA_DIR/librivox-dev-clean.csv" \
  --test_files "$COMPUTE_DATA_DIR/librivox-test-clean.csv" \
  --test_batch_size 32 \
  --notrain \
  --test \
  --n_hidden 2048 \
  --wer_log_pattern "GLOBAL LOG: logwer('${COMPUTE_ID}', '%s', '%s', %f)" \
  --lstm_type cudnn \
  --half_precision \
  --log_level 0 \
  --display_step 0 \
  --initialize_from_checkpoint "$HOME/efs/3-epoch_cudnn_mixed-precision/tf_dir/checkpoints/model.ckpt-6591" \
  --report_count 100 \
  --dropout_rate 0.0 \
  --early_stop 0 \
  "$@"
