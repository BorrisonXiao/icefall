#!/usr/bin/env bash
# Cihan Xiao 2022

set -eou pipefail

nj=16
stage=5
stop_stage=5
python=python3

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/LibriSpeech
#      You can find BOOKS.TXT, test-clean, train-clean-360, etc, inside it.
#      You can download them from https://www.openslr.org/12
dl_dir=$PWD/download

. shared/parse_options.sh || exit 1

# vocab size for sentence piece models.
# It will generate data/lang_bpe_xxx,
# data/lang_bpe_yyy if the array contains xxx, yyy
vocab_sizes=(
  7000
  6000
  4000
  2500
)

# All files generated by this script are saved in "data".
# You can safely remove "data" and rerun this script to regenerate it.
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

log "dl_dir: $dl_dir"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"

  # If you have pre-downloaded it to /path/to/FLEURS,
  # you can create a symlink
  #
  #   ln -sfv /path/to/FLEURS $dl_dir/FLEURS
  #
  mkdir -p "$dl_dir"
  ${python} <<EOF
from datasets import load_dataset
fleurs = load_dataset("google/fleurs", "yue_hant_hk", cache_dir="$dl_dir/FLEURS")
EOF
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare FLEURS manifest"
  # We assume that you have downloaded the FLEURS corpus
  # to $dl_dir/FLEURS
  mkdir -p data/manifests
  if [ ! -e data/manifests/.fleurs.done ]; then
    lhotse prepare fleurs -j $nj "$dl_dir/FLEURS" data/manifests
    touch data/manifests/.fleurs.done
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Compute fbank for fleurs"
  mkdir -p data/fbank
  if [ ! -e data/fbank/.fleurs.done ]; then
    ./local/compute_fbank_fleurs.py
    touch data/fbank/.fleurs.done
  fi

  if [ ! -e data/fbank/.fleurs-validated.done ]; then
    log "Validating data/fbank for FLEURS"
    parts=(
      "train"
      "validation"
      "test"
    )
    for part in "${parts[@]}"; do
      python3 ./local/validate_manifest.py \
        data/fbank/fleurs_cuts_${part}.jsonl.gz
    done
    touch data/fbank/.fleurs-validated.done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare phone based lang"
  lang_dir=data/lang_phone
  mkdir -p $lang_dir

  # Using hklegco lexicon.txt for inference
  #
  ln -sfv /home/cxiao7/research/icefall/egs/hklegco/ASR/data/lang_phone/lexicon.txt $lang_dir/lexicon.txt
  #
  # if [ ! -f $lang_dir/lexicon.txt ]; then
  #   ./local/generate_lexicon.py \
  #     --lang-dir $lang_dir \
  #     --lexicon $lang_dir/lexicon.raw.txt
  # fi

  # (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |
  #   cat - $lang_dir/lexicon.raw.txt |
  #   sort | uniq > $lang_dir/lexicon.txt

  if [ ! -f $lang_dir/L_disambig.pt ]; then
    ./local/prepare_lang.py --lang-dir $lang_dir
  fi
fi

# if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
#   log "Stage 4: Prepare BPE based lang"

#   for vocab_size in "${vocab_sizes[@]}"; do
#     lang_dir=data/lang_bpe_${vocab_size}
#     mkdir -p $lang_dir

#     if [ ! -f $lang_dir/bpe.model ]; then
#       ./local/train_bpe_model.py \
#         --lang-dir $lang_dir \
#         --vocab-size $vocab_size \
#         --supervision-set data/manifests/fleurs_supervisions_train.jsonl.gz
#     fi

#     if [ ! -f $lang_dir/L_disambig.pt ]; then
#       ./local/prepare_lang_bpe.py --lang-dir $lang_dir

#       log "Validating $lang_dir/lexicon.txt"
#       ./local/validate_bpe_lexicon.py \
#         --lexicon $lang_dir/lexicon.txt \
#         --bpe-model $lang_dir/bpe.model
#     fi
#   done
# fi

# if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
#   log "Stage 5: Prepare bigram P"

#   for vocab_size in "${vocab_sizes[@]}"; do
#     lang_dir=data/lang_bpe_${vocab_size}

#     if [ ! -f $lang_dir/transcript_tokens.txt ]; then
#       ./local/convert_transcript_words_to_tokens.py \
#         --lexicon $lang_dir/lexicon.txt \
#         --bpe-model $lang_dir/bpe.model \
#         --transcript $lang_dir/transcript_words.txt \
#         --oov "<UNK>" \
#         > $lang_dir/transcript_tokens.txt
#     fi

#     if [ ! -f $lang_dir/P.arpa ]; then
#       ./shared/make_kn_lm.py \
#         -ngram-order 2 \
#         -text $lang_dir/transcript_tokens.txt \
#         -lm $lang_dir/P.arpa
#     fi

#     if [ ! -f $lang_dir/P.fst.txt ]; then
#       python3 -m kaldilm \
#         --read-symbol-table="$lang_dir/tokens.txt" \
#         --disambig-symbol='#0' \
#         --max-order=2 \
#         $lang_dir/P.arpa > $lang_dir/P.fst.txt
#     fi
#   done
# fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare G"
  # We assume you have install kaldilm, if not, please install
  # it using: pip install kaldilm
  lang_dir=data/lang_phone
  lm_dir=data/lm_phone
  # rm -rf $lm_dir # TODO: Remove
  mkdir -p $lm_dir

  if [ ! -f $lm_dir/transcript_tokens.txt ]; then
    ./local/transcript_tokens.py \
      --manifests-dir data/manifests \
      --output $lm_dir/transcript_tokens.txt
  fi

<<<<<<< HEAD
  if [ ! -f $lm_dir/transcript_tokens_validation.txt ]; then
    ./local/transcript_tokens.py \
      --manifests-dir data/manifests \
      --output $lm_dir/transcript_tokens_validation.txt \
      --part "validation"
  fi

  if [ ! -f $lm_dir/transcript_tokens_test.txt ]; then
    ./local/transcript_tokens.py \
      --manifests-dir data/manifests \
      --output $lm_dir/transcript_tokens_test.txt \
      --part "test"
  fi

=======
>>>>>>> 8b96e5edcb5894cc5ce5ee14c3800c1e4dac653c
  if [ ! -f $lm_dir/G_3_gram.arpa ]; then
    ./shared/make_kn_lm.py \
      -ngram-order 3 \
      -text $lm_dir/transcript_tokens.txt \
      -lm $lm_dir/G_3_gram.arpa
  fi

  if [ ! -f $lm_dir/G_3_gram.fst.txt ]; then
    log "Making kaldilm for $lm_dir/G_3_gram.arpa"
    python3 -m kaldilm \
      --read-symbol-table="$lang_dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      $lm_dir/G_3_gram.arpa >$lm_dir/G_3_gram.fst.txt
  fi

  if [ ! -f $lm_dir/G_4_gram.arpa ]; then
    ./shared/make_kn_lm.py \
      -ngram-order 4 \
      -text $lm_dir/transcript_tokens.txt \
      -lm $lm_dir/G_4_gram.arpa
  fi

  if [ ! -f $lm_dir/G_4_gram.fst.txt ]; then
    log "Making kaldilm for $lm_dir/G_4_gram.arpa"
    python3 -m kaldilm \
      --read-symbol-table="$lang_dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      $lm_dir/G_4_gram.arpa >$lm_dir/G_4_gram.fst.txt
  fi

  # for vocab_size in "${vocab_sizes[@]}"; do
  #   lang_dir=data/lang_bpe_${vocab_size}
  #   lm_dir=data/lm_${vocab_size}
  #   mkdir -p $lm_dir

  #   if [ ! -f $lm_dir/transcript_words.processed.txt ]; then
  #     ./local/transcript_words_bpe.py \
  #       --transcript $lang_dir/transcript_tokens.txt \
  #       --output $lm_dir/transcript_words.processed.txt
  #   fi

  #   if [ ! -f $lm_dir/G_3_gram.arpa ]; then
  #     ./shared/make_kn_lm.py \
  #       -ngram-order 3 \
  #       -text $lm_dir/transcript_words.processed.txt \
  #       -lm $lm_dir/G_3_gram.arpa
  #   fi

  #   if [ ! -f $lm_dir/G_3_gram.fst.txt ]; then
  #     log "Making kaldilm for $lm_dir/G_3_gram.arpa"
  #     python3 -m kaldilm \
  #       --read-symbol-table="$lang_dir/words.txt" \
  #       --disambig-symbol='#0' \
  #       --max-order=3 \
  #       $lm_dir/G_3_gram.arpa >$lm_dir/G_3_gram.fst.txt
  #   fi

  #   if [ ! -f $lm_dir/G_4_gram.arpa ]; then
  #     ./shared/make_kn_lm.py \
  #       -ngram-order 4 \
  #       -text $lm_dir/transcript_words.processed.txt \
  #       -lm $lm_dir/G_4_gram.arpa
  #   fi

  #   if [ ! -f $lm_dir/G_4_gram.fst.txt ]; then
  #     log "Making kaldilm for $lm_dir/G_4_gram.arpa"
  #     python3 -m kaldilm \
  #       --read-symbol-table="$lang_dir/words.txt" \
  #       --disambig-symbol='#0' \
  #       --max-order=4 \
  #       $lm_dir/G_4_gram.arpa >$lm_dir/G_4_gram.fst.txt
  #   fi
  # done
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Compile HLG"
  ./local/compile_hlg.py --lang-dir data/lang_phone --lm-dir data/lm_phone

  # for vocab_size in ${vocab_sizes[@]}; do
  #   lang_dir=data/lang_bpe_${vocab_size}
  #   ./local/compile_hlg.py --lang-dir $lang_dir
  # done
fi

# Compile LG for RNN-T fast_beam_search decoding
if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
  log "Stage 10: Compile LG"
  ./local/compile_lg.py --lang-dir data/lang_phone --lm-dir data/lm_phone

  # for vocab_size in ${vocab_sizes[@]}; do
  #   lang_dir=data/lang_bpe_${vocab_size}
  #   ./local/compile_lg.py --lang-dir $lang_dir
  # done
fi

<<<<<<< HEAD
if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
  log "Interpolating the LegiCoST LM with the fleurs LM"

  lang_dir=/home/cxiao7/research/hklegco_icefall/data/lang_phone
  interpolate_dir=data/lm_phone_interpolate
  ppl_dir=${interpolate_dir}/ppl
  mkdir -p $ppl_dir
  weights_dir=${interpolate_dir}/weights
  mkdir -p $weights_dir

  lm1_3gram=data/lm_phone/G_3_gram.arpa
  lm2_3gram=/home/cxiao7/research/hklegco_icefall/data/lm_phone/G_3_gram.arpa
  lm1_4gram=data/lm_phone/G_4_gram.arpa
  lm2_4gram=/home/cxiao7/research/hklegco_icefall/data/lm_phone/G_4_gram.arpa
  dev_text=data/lm_phone/transcript_tokens_validation.txt

  # Compute the interpolation weights based on the validation transcripts
  ngram -debug 2 -order 3 -unk -lm ${lm1_3gram} -ppl $dev_text >$ppl_dir/lm1-3gram.ppl
  ngram -debug 2 -order 3 -unk -lm ${lm2_3gram} -ppl $dev_text >$ppl_dir/lm2-3gram.ppl
  compute-best-mix $ppl_dir/*-3gram.ppl >$weights_dir/best-mix-3gram.ppl

  ngram -debug 2 -order 4 -unk -lm ${lm1_4gram} -ppl $dev_text >$ppl_dir/lm1-4gram.ppl
  ngram -debug 2 -order 4 -unk -lm ${lm2_4gram} -ppl $dev_text >$ppl_dir/lm2-4gram.ppl
  compute-best-mix $ppl_dir/*-4gram.ppl >$weights_dir/best-mix-4gram.ppl

  # Interpolate the two LMs
  ngram \
    -order 3 \
    -lm ${lm1_3gram} \
    -mix-lm ${lm2_3gram} \
    -lambda $(head -n 1 $weights_dir/best-mix-3gram.ppl | awk '{print substr($(NF-1),2)}') \
    -write-lm $interpolate_dir/G_3_gram.arpa

  ngram \
    -order 4 \
    -lm ${lm1_4gram} \
    -mix-lm ${lm2_4gram} \
    -lambda $(head -n 1 $weights_dir/best-mix-4gram.ppl | awk '{print substr($(NF-1),2)}') \
    -write-lm $interpolate_dir/G_4_gram.arpa

  if [ ! -f $interpolate_dir/G_3_gram.fst.txt ]; then
    log "Making kaldilm for $interpolate_dir/G_3_gram.arpa"
    python3 -m kaldilm \
      --read-symbol-table="$lang_dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      $interpolate_dir/G_3_gram.arpa >$interpolate_dir/G_3_gram.fst.txt
  fi

  if [ ! -f $interpolate_dir/G_4_gram.fst.txt ]; then
    log "Making kaldilm for $interpolate_dir/G_4_gram.arpa"
    python3 -m kaldilm \
      --read-symbol-table="$lang_dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      $interpolate_dir/G_4_gram.arpa >$interpolate_dir/G_4_gram.fst.txt
  fi

  inter_lang_dir=data/lang_phone_interpolate
  mkdir -p $inter_lang_dir
  cp $lang_dir/*.txt $inter_lang_dir
  cp $lang_dir/L.pt $inter_lang_dir
  cp $lang_dir/L_disambig.pt $inter_lang_dir

  ./local/compile_hlg.py --lang-dir $inter_lang_dir --lm-dir $interpolate_dir
  ./local/compile_lg.py --lang-dir $inter_lang_dir --lm-dir $interpolate_dir
fi

if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
  log "Analyze the OOV situation"

  # Using word.txt from the hklegco corpus
  lang_dir=/home/cxiao7/research/hklegco_icefall/data/lang_phone
  test_scp=data/lm_phone/transcript_tokens_test.txt
  train_scp=data/lm_phone/transcript_tokens.txt

  python3 local/analyze_oov.py \
    --lang-dir $lang_dir \
    --scp $test_scp

  python3 local/analyze_oov.py \
    --lang-dir $lang_dir \
    --scp $train_scp
fi

if [ $stage -le 13 ] && [ $stop_stage -ge 13 ]; then
  log "Concatenate the LegiCoST lexicon with the fleurs lexicon"

  tgt_lang_dir=data/lang_phone_concat
  mkdir -p $tgt_lang_dir
  src_lang_dir=/home/cxiao7/research/hklegco_icefall/data/lang_phone
  tgt_lexicon=$tgt_lang_dir/lexicon.tgt.txt
  src_lexicon=$src_lang_dir/lexicon.raw.txt

  # Generate the lexicon for the target text
  python3 local/generate_lexicon.py \
    --manifests-dir data/manifests \
    --lexicon $tgt_lexicon \
    --tokens $src_lang_dir/tokens.txt \
    --words $src_lang_dir/words.txt

  (
    echo '!SIL SIL'
    echo '<SPOKEN_NOISE> SPN'
    echo '<UNK> SPN'
  ) |
    cat - $src_lexicon $tgt_lexicon |
    sort | uniq >$tgt_lang_dir/lexicon.txt

  if [ ! -f $tgt_lang_dir/L_disambig.pt ]; then
    ./local/prepare_lang.py --lang-dir $tgt_lang_dir
  fi
fi

if [ $stage -le 14 ] && [ $stop_stage -ge 14 ]; then
  log "Perform LM interpolation using the concatenated lexicon"

  lang_dir=data/lang_phone_concat
  lmdir1=data/lm_phone
  lmdir2=/home/cxiao7/research/hklegco_icefall/data/lm_phone
  interpolate_dir=data/lm_phone_concat
  ppl_dir=${interpolate_dir}/ppl
  mkdir -p $ppl_dir
  weights_dir=${interpolate_dir}/weights
  mkdir -p $weights_dir

  lm1_3gram=$lmdir1/G_3_gram.arpa
  lm2_3gram=$lmdir2/G_3_gram.arpa
  lm1_4gram=$lmdir1/G_4_gram.arpa
  lm2_4gram=$lmdir2/G_4_gram.arpa
  dev_text=$lmdir1/transcript_tokens_validation.txt

  # Compute the interpolation weights based on the validation transcripts
  ngram -debug 2 -order 3 -unk -lm ${lm1_3gram} -ppl $dev_text >$ppl_dir/lm1-3gram.ppl
  ngram -debug 2 -order 3 -unk -lm ${lm2_3gram} -ppl $dev_text >$ppl_dir/lm2-3gram.ppl
  compute-best-mix $ppl_dir/*-3gram.ppl >$weights_dir/best-mix-3gram.ppl

  ngram -debug 2 -order 4 -unk -lm ${lm1_4gram} -ppl $dev_text >$ppl_dir/lm1-4gram.ppl
  ngram -debug 2 -order 4 -unk -lm ${lm2_4gram} -ppl $dev_text >$ppl_dir/lm2-4gram.ppl
  compute-best-mix $ppl_dir/*-4gram.ppl >$weights_dir/best-mix-4gram.ppl

  # Interpolate the two LMs
  ngram \
    -order 3 \
    -lm ${lm1_3gram} \
    -mix-lm ${lm2_3gram} \
    -lambda $(head -n 1 $weights_dir/best-mix-3gram.ppl | awk '{print substr($(NF-1),2)}') \
    -write-lm $interpolate_dir/G_3_gram.arpa

  ngram \
    -order 4 \
    -lm ${lm1_4gram} \
    -mix-lm ${lm2_4gram} \
    -lambda $(head -n 1 $weights_dir/best-mix-4gram.ppl | awk '{print substr($(NF-1),2)}') \
    -write-lm $interpolate_dir/G_4_gram.arpa

  if [ ! -f $interpolate_dir/G_3_gram.fst.txt ]; then
    log "Making kaldilm for $interpolate_dir/G_3_gram.arpa"
    python3 -m kaldilm \
      --read-symbol-table="$lang_dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      $interpolate_dir/G_3_gram.arpa >$interpolate_dir/G_3_gram.fst.txt
  fi

  if [ ! -f $interpolate_dir/G_4_gram.fst.txt ]; then
    log "Making kaldilm for $interpolate_dir/G_4_gram.arpa"
    python3 -m kaldilm \
      --read-symbol-table="$lang_dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      $interpolate_dir/G_4_gram.arpa >$interpolate_dir/G_4_gram.fst.txt
  fi

  ./local/compile_hlg.py --lang-dir $lang_dir --lm-dir $interpolate_dir
  ./local/compile_lg.py --lang-dir $lang_dir --lm-dir $interpolate_dir
fi

=======
>>>>>>> 8b96e5edcb5894cc5ce5ee14c3800c1e4dac653c
# if [ $stage -le 11 ] && [ $stop_stage -ge 11 ]; then
#   log "Stage 11: Generate LM training data"

#   for vocab_size in ${vocab_sizes[@]}; do
#     log "Processing vocab_size == ${vocab_size}"
#     lang_dir=data/lang_bpe_${vocab_size}
#     out_dir=data/lm_training_bpe_${vocab_size}
#     mkdir -p $out_dir

#     ./local/prepare_lm_training_data.py \
#       --bpe-model $lang_dir/bpe.model \
#       --lm-data $dl_dir/lm/librispeech-lm-norm.txt \
#       --lm-archive $out_dir/lm_data.pt
#   done
# fi

# if [ $stage -le 12 ] && [ $stop_stage -ge 12 ]; then
#   log "Stage 12: Generate LM validation data"

#   for vocab_size in ${vocab_sizes[@]}; do
#     log "Processing vocab_size == ${vocab_size}"
#     out_dir=data/lm_training_bpe_${vocab_size}
#     mkdir -p $out_dir

#     if [ ! -f $out_dir/valid.txt ]; then
#       files=$(
#         find "$dl_dir/LibriSpeech/dev-clean" -name "*.trans.txt"
#         find "$dl_dir/LibriSpeech/dev-other" -name "*.trans.txt"
#       )
#       for f in ${files[@]}; do
#         cat $f | cut -d " " -f 2-
#       done > $out_dir/valid.txt
#     fi

#     lang_dir=data/lang_bpe_${vocab_size}
#     ./local/prepare_lm_training_data.py \
#       --bpe-model $lang_dir/bpe.model \
#       --lm-data $out_dir/valid.txt \
#       --lm-archive $out_dir/lm_data-valid.pt
#   done
# fi

# if [ $stage -le 13 ] && [ $stop_stage -ge 13 ]; then
#   log "Stage 13: Generate LM test data"

#   for vocab_size in ${vocab_sizes[@]}; do
#     log "Processing vocab_size == ${vocab_size}"
#     out_dir=data/lm_training_bpe_${vocab_size}
#     mkdir -p $out_dir

#     if [ ! -f $out_dir/test.txt ]; then
#       files=$(
#         find "$dl_dir/LibriSpeech/test-clean" -name "*.trans.txt"
#         find "$dl_dir/LibriSpeech/test-other" -name "*.trans.txt"
#       )
#       for f in ${files[@]}; do
#         cat $f | cut -d " " -f 2-
#       done > $out_dir/test.txt
#     fi

#     lang_dir=data/lang_bpe_${vocab_size}
#     ./local/prepare_lm_training_data.py \
#       --bpe-model $lang_dir/bpe.model \
#       --lm-data $out_dir/test.txt \
#       --lm-archive $out_dir/lm_data-test.pt
#   done
# fi

# if [ $stage -le 14 ] && [ $stop_stage -ge 14 ]; then
#   log "Stage 14: Sort LM training data"
#   # Sort LM training data by sentence length in descending order
#   # for ease of training.
#   #
#   # Sentence length equals to the number of BPE tokens
#   # in a sentence.

#   for vocab_size in ${vocab_sizes[@]}; do
#     out_dir=data/lm_training_bpe_${vocab_size}
#     mkdir -p $out_dir
#     ./local/sort_lm_training_data.py \
#       --in-lm-data $out_dir/lm_data.pt \
#       --out-lm-data $out_dir/sorted_lm_data.pt \
#       --out-statistics $out_dir/statistics.txt

#     ./local/sort_lm_training_data.py \
#       --in-lm-data $out_dir/lm_data-valid.pt \
#       --out-lm-data $out_dir/sorted_lm_data-valid.pt \
#       --out-statistics $out_dir/statistics-valid.txt

#     ./local/sort_lm_training_data.py \
#       --in-lm-data $out_dir/lm_data-test.pt \
#       --out-lm-data $out_dir/sorted_lm_data-test.pt \
#       --out-statistics $out_dir/statistics-test.txt
#   done
# fi

log "Successfully finished. [elapsed=${SECONDS}s]"
