#!/usr/bin/env bash

# (See qsub section for explanation on these flags.)
#$ -cwd
#$ -N pretrain_merlion
#$ -j y -o /home/cxiao7/research/merlion_ASR/conformer_mtl/qlogs/$JOB_NAME-$JOB_ID.out
#$ -m e

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=16G,mem_free=16G,gpu=4,hostname="c0[2356789]|c1*|c2[012356789]"

# Submit to GPU queue
#$ -q g.q

# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
# for _ in $(seq 2); do source /home/gqin2/scripts/acquire-gpu; done
source /home/gqin2/scripts/acquire-gpus 4
# or, less safely:
# export CUDA_VISIBLE_DEVICES=$(free-gpu -n 1)

. ./path.sh
./conformer_mtl/pretrain.py \
     --exp-dir ./conformer_mtl/exp \
     --world-size 4 \
     --max-duration 60 \
     --start-epoch 2 \
     --num-epochs 5 \
     --lang-dir data/lang_phone \
     --att-rate 0 \
     --lid-rate 0.2 \
     --lid-scale 100 \
     --num-decoder-layers 0