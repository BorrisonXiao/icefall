#!/usr/bin/env bash

# (See qsub section for explanation on these flags.)
#$ -cwd
#$ -N train_hklegco
#$ -j y -o /home/cxiao7/research/icefall/egs/hklegco/ASR/conformer_ctc/exp/log/qlogs/$JOB_NAME-$JOB_ID.out
#$ -m e

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=16G,mem_free=16G,gpu=2,hostname="c0[2356789]|c1*|c2*"

# Submit to GPU queue
#$ -q g.q

# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
# for _ in $(seq 2); do source /home/gqin2/scripts/acquire-gpu; done
source /home/gqin2/scripts/acquire-gpus 2
# or, less safely:
# export CUDA_VISIBLE_DEVICES=$(free-gpu -n 1)

. ./path.sh
# ./conformer_ctc/train.py \
#      --exp-dir ./conformer_ctc/exp \
#      --world-size 2 \
#      --max-duration 160 \
#      --start-epoch 10 \
#      --num-epochs 15 \
#      --lang-dir data/lang_phone \
#      --att-rate 0 \
#      --num-decoder-layers 0 \
#      --lr-factor 5.0

./conformer_ctc/train.py \
     --exp-dir ./conformer_ctc/exp \
     --world-size 2 \
     --max-duration 160 \
     --start-epoch 10 \
     --num-epochs 15 \
     --lang-dir data/lang_phone \
     --att-rate 0 \
     --num-decoder-layers 0 \
     --lr-factor 5.0
