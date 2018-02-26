#!/bin/bash
PREV_SPARSITY='56.8'
SPARSITY='61.12'
EPOCH=4

STORE='/home/ubuntu/efs/'
EXPERIMENT=$SPARSITY'_fine-tune/'
LOG_DIR=$STORE$EXPERIMENT
TF_DIR=$LOG_DIR'tf_dir'
TRAIN_LOG=$LOG_DIR'train_log.txt'
VALIDATE_LOG=$LOG_DIR'validate_log.txt'

# Fine-tune
rm -r ~/.local/share/deepspeech
mkdir $LOG_DIR
mkdir $TF_DIR
export COMPUTE_KEEP_DIR=$TF_DIR
./train.sh --epoch $EPOCH \
           --initialize_from_checkpoint $STORE$PREV_SPARSITY'_fine-tune/'$SPARSITY"_base/model.ckpt-$((2197 * ($EPOCH - 1)))" \
           --dropout_rate `python -c "print(0.2367 * (1.0 - ($SPARSITY / 100.0))**0.5)"` \
	   2>&1 | tee $TRAIN_LOG
./validate.sh  $TF_DIR'/checkpoints/model.ckpt-'$((2197 * $EPOCH)) 2>&1 | tee $VALIDATE_LOG
touch $LOG_DIR'finished'

# Generate next base(s)
unset COMPUTE_KEEP_DIR
SPARSITY_LOSS=$LOG_DIR'/sparsity_loss.csv'
echo "Sparsity (%),Validation Loss" > $SPARSITY_LOSS

for i in `seq 1.0 -0.1 0.1`; do
    rm -r ~/.local/share/deepspeech
    NEXT_SPARSITY=`python -c "s = float($SPARSITY); print('%.2f' % (s + ((100. - s) * $i)))"`
    BASE_DIR=$LOG_DIR$NEXT_SPARSITY'_base/'
    mkdir $BASE_DIR
    echo $NEXT_SPARSITY > $BASE_DIR'sparsity.txt'
    OUT_CKPT=$BASE_DIR'model.ckpt-'$((2197 * $EPOCH))
    python $HOME"/Code/DeepSpeech/util/checkpoint_sparsity.py" \
        --out_ckpt $OUT_CKPT \
        --in_ckpt $TF_DIR'/checkpoints/model.ckpt-'$((2197 * $EPOCH)) \
        --to_mask $HOME'/Code/DeepSpeech/util/cudnn_to_mask.txt' \
        --sparsity $NEXT_SPARSITY
    ./validate.sh --initialize_from_checkpoint $OUT_CKPT 2>&1 | tee $BASE_DIR'/validate_log.txt'
    echo -n "$NEXT_SPARSITY," >> $SPARSITY_LOSS
    grep 'loss: ' $BASE_DIR'/validate_log.txt' | awk -F ' ' '{print $8}' >> $SPARSITY_LOSS
done

sudo shutdown
