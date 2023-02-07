#!/usr/bin/env bash

# (See qsub section for explanation on these flags.)
#$ -cwd
#$ -N train
#$ -j y -o /home/cxiao7/research/icefall/egs/hklegco/ASR/conformer_ctc/exp/log/qlogs/$JOB_NAME-$JOB_ID.out
#$ -m e

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=32G,mem_free=32G,gpu=1,hostname="b1*|c0[2356789]|c1*"

# Submit to GPU queue
#$ -q g.q

# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
for _ in $(seq 1); do source /home/gqin2/scripts/acquire-gpu; done
# or, less safely:
# export CUDA_VISIBLE_DEVICES=$(free-gpu -n 1)

. ./path.sh
./conformer_ctc/train.py \
     --exp-dir ./conformer_ctc/exp \
     --world-size 1 \
     --max-duration 150 \
     --num-epochs 2 \
     --lang-dir data/lang_phone \
     --att-rate 0 \
     --num-decoder-layers 0
