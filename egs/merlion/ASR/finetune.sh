#!/usr/bin/env bash

# (See qsub section for explanation on these flags.)
#$ -cwd
#$ -N finetune_mtl
#$ -j y -o /home/cxiao7/research/merlion_ASR/conformer_mtl/qlogs/$JOB_NAME-$JOB_ID.out
#$ -m e

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=16G,mem_free=16G,gpu=2,hostname="b19"

# Submit to GPU queue
#$ -q g.q

# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
source /home/gqin2/scripts/acquire-gpus 2
# or, less safely:
# export CUDA_VISIBLE_DEVICES=$(free-gpu -n 1)

. ./path.sh
./conformer_mtl/finetune2.py \
     --exp-dir ./conformer_mtl/exp \
     --world-size 2 \
     --max-duration 120 \
     --start-epoch 15 \
     --num-epochs 20 \
     --lang-dir data/lang_phone \
     --att-rate 0 \
     --num-decoder-layers 0 \
     --m 3 \
     --n-factor 0.0 \
     --lr-factor 0.2