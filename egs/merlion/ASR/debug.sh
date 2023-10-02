#!/usr/bin/env bash

# (See qsub section for explanation on these flags.)
#$ -cwd
#$ -N debug
#$ -j y -o /home/cxiao7/research/merlion_ASR/conformer_mtl/qlogs/$JOB_NAME-$JOB_ID.out
#$ -m e

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=16G,mem_free=16G,gpu=1,hostname="b1*|b2*|c0[2356789]|c1[0123456789]|c2[0123456789]"

# Submit to GPU queue
#$ -q g.q

# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
source /home/gqin2/scripts/acquire-gpus 1
# or, less safely:
# export CUDA_VISIBLE_DEVICES=$(free-gpu -n 1)

. ./path.sh
./conformer_mtl/finetune.py \
     --exp-dir ./conformer_mtl/exp \
     --world-size 1 \
     --max-duration 39 \
     --start-epoch 11 \
     --num-epochs 15 \
     --lang-dir data/lang_phone \
     --att-rate 0 \
     --num-decoder-layers 0 \
     --lr-factor 0.2