#!/usr/bin/env bash

# (See qsub section for explanation on these flags.)
#$ -cwd
#$ -N decode
#$ -j y -o /home/cxiao7/research/merlion_ASR/conformer_mtl/qlogs/$JOB_NAME-$JOB_ID.out
#$ -m e

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=16G,mem_free=16G,gpu=1,hostname="c0[2356789]|c1*|c2*"

# Submit to GPU queue
#$ -q g.q

# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
# source /home/gqin2/scripts/acquire-gpus 1
# or, less safely:
# export CUDA_VISIBLE_DEVICES=$(free-gpu -n 1)

epoch=6
batch=2000
method=nbest-rescoring
num_paths=2

. ./path.sh
./conformer_mtl/decode.py \
    --epoch ${epoch} \
    --batch ${batch} \
    --avg 1 \
    --max-duration 60 \
    --exp-dir conformer_mtl/exp_2 \
    --lang-dir data/lang_phone \
    --method ${method} \
    --num-paths ${num_paths} \
    --lm-dir data/lm_phone

# eval_dir=conformer_mtl/exp/evals/${method}-epoch-${epoch}
# mkdir -p ${eval_dir}
# mv conformer_mtl/exp/*.txt ${eval_dir}