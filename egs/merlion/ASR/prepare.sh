#!/usr/bin/env bash
# Cihan Xiao 2022

set -eou pipefail

nj=16
stage=0
stop_stage=100
python=python3

# We assume dl_dir (download dir) contains the following
# directories and files. If not, they will be downloaded
# by this script automatically.
#
#  - $dl_dir/LibriSpeech
#      You can find BOOKS.TXT, test-clean, train-clean-360, etc, inside it.
#      You can download them from https://www.openslr.org/12
dl_dir=/export/b14/cxiao7/data/download

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
parts=(
  "SEAME"
  "LibriSpeech"
  "NSC"
  "AISHELL"
  "dev"
  "test"
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

  # Link to the pre-downloaded LibriSpeech corpus
  mkdir -p $dl_dir
  if [ ! -d $dl_dir/LibriSpeech/train-clean-100 ]; then
    ln -sfv /export/corpora5/LibriSpeech $dl_dir/LibriSpeech
  fi

  # Link to the pre-downloaded SEAME corpus
  if [ ! -d $dl_dir/SEAME/data ]; then
    ln -sfv /export/corpora5/LDC/LDC2015S04 $dl_dir/SEAME
  fi

  if [ ! -d $dl_dir/aishell/data_aishell/wav/train ]; then
    mkdir -p $dl_dir/aishell/data_aishell/wav
    mkdir -p $dl_dir/aishell/data_aishell/transcript
    # c07 was so slow so the data is copied to b14
    rsync -avz --bwlimit=200000 --progress --partial /export/c07/sli136/merlion/aishell/train/* $dl_dir/aishell/data_aishell/wav/train
    rsync -avz --bwlimit=200000 --progress --partial /export/c07/sli136/merlion/aishell/aishell_transcript_v0.8.txt $dl_dir/aishell/data_aishell/transcript/aishell_transcript_v0.8.txt

    # lhotse download aishell $dl_dir
  fi

  if [ ! -d $dl_dir/nsc ]; then
    # Link to the pre-downloaded NSC corpus
    mkdir -p $dl_dir/nsc
    rsync -avz --bwlimit=200000 --progress --partial /export/c07/sli136/merlion/national-speech/* $dl_dir/nsc
  fi

  if [ ! -d $dl_dir/dev ]; then
    # Copy the dev set
    mkdir -p $dl_dir/dev
    rsync -avz --bwlimit=200000 --progress --partial /export/c07/sli136/merlion/merlion-devset/* $dl_dir/dev
  fi

  if [ ! -d $dl_dir/test ]; then
    # Copy the test set
    mkdir -p $dl_dir/test
    rsync -avz --bwlimit=200000 --progress --partial /export/b14/sli136/merlion/Task_1_Eval/* $dl_dir/test
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Prepare the manifests"
  # if [ ! -f data/manifests/.manifests.done ]; then
  mkdir -p data/manifests
  lhotse prepare merlion $dl_dir data/manifests -p devs
  touch data/manifests/.manifests.done
  # fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Compute fbank for merlion sets"
  mkdir -p data/fbank
  # if [ ! -e data/fbank/.merlion.done ]; then
  ./local/compute_fbank_merlion.py
  touch data/fbank/.merlion.done
  # fi

  # if [ ! -e data/fbank/.merlion-validated.done ]; then
  #   log "Validating data/fbank for Merlion"
  #   for part in "${parts[@]}"; do
  #     python3 ./local/validate_manifest.py \
  #       data/fbank/cuts_${part}.jsonl.gz
  #   done
  #   touch data/fbank/.merlion-validated.done
  # fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Prepare phone based lang"
  lang_dir=data/lang_phone
  mkdir -p $lang_dir

  # If you have a pre-processed /path/to/lexicon.txt,
  # you can create a symlink
  #
  #   ln -sfv /path/to/lexicon.txt $lang_dir/lexicon.txt
  #
  # if [ ! -f $lang_dir/lexicon.txt ]; then
  ./local/generate_lexicon.py \
    --manifests-dir data/manifests \
    --lexicon $lang_dir/lexicon.raw.txt
  # fi

  (
    echo '!SIL SIL'
    echo '<SPOKEN_NOISE> SPN'
    echo '<UNK> SPN'
  ) |
    cat - $lang_dir/lexicon.raw.txt |
    sort | uniq >$lang_dir/lexicon.txt

  # if [ ! -f $lang_dir/L_disambig.pt ]; then
  ./local/prepare_lang.py --lang-dir $lang_dir
  # fi
fi

# if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
#   log "Stage 4: Prepare BPE based lang"

#   lang_dir=data/lang_phone

#   # First generate the English word transcripts
#   ./local/prepare_english_transcripts.py \
#     --manifests-dir data/manifests \
#     --output ${lang_dir}/transcript_words_en.txt

#   # for vocab_size in "${vocab_sizes[@]}"; do
#   #   lang_dir=data/lang_bpe_${vocab_size}
#   #   mkdir -p $lang_dir

#   #   if [ ! -f $lang_dir/bpe.model ]; then
#   #     ./local/train_bpe_model.py \
#   #       --lang-dir $lang_dir \
#   #       --vocab-size $vocab_size \
#   #       --supervision-set data/manifests/fleurs_supervisions_train.jsonl.gz
#   #   fi

#   #   if [ ! -f $lang_dir/L_disambig.pt ]; then
#   #     ./local/prepare_lang_bpe.py --lang-dir $lang_dir

#   #     log "Validating $lang_dir/lexicon.txt"
#   #     ./local/validate_bpe_lexicon.py \
#   #       --lexicon $lang_dir/lexicon.txt \
#   #       --bpe-model $lang_dir/bpe.model
#   #   fi
#   # done
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
  mkdir -p $lm_dir

  # if [ ! -f $lm_dir/transcript_tokens.txt ]; then
  ./local/transcript_tokens.py \
    --manifests-dir data/manifests \
    --output $lm_dir/transcript_tokens.txt
  # fi

  # if [ ! -f $lm_dir/G_3_gram.arpa ]; then
    ./shared/make_kn_lm.py \
      -ngram-order 3 \
      -text $lm_dir/transcript_tokens.txt \
      -lm $lm_dir/G_3_gram.arpa
  # fi

  # if [ ! -f $lm_dir/G_3_gram.fst.txt ]; then
    log "Making kaldilm for $lm_dir/G_3_gram.arpa"
    python3 -m kaldilm \
      --read-symbol-table="$lang_dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      $lm_dir/G_3_gram.arpa >$lm_dir/G_3_gram.fst.txt
  # fi

  # if [ ! -f $lm_dir/G_4_gram.arpa ]; then
    ./shared/make_kn_lm.py \
      -ngram-order 4 \
      -text $lm_dir/transcript_tokens.txt \
      -lm $lm_dir/G_4_gram.arpa
  # fi

  # if [ ! -f $lm_dir/G_4_gram.fst.txt ]; then
    log "Making kaldilm for $lm_dir/G_4_gram.arpa"
    python3 -m kaldilm \
      --read-symbol-table="$lang_dir/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      $lm_dir/G_4_gram.arpa >$lm_dir/G_4_gram.fst.txt
  # fi

#   for vocab_size in "${vocab_sizes[@]}"; do
#     lang_dir=data/lang_bpe_${vocab_size}
#     lm_dir=data/lm_${vocab_size}
#     mkdir -p $lm_dir

#     if [ ! -f $lm_dir/transcript_words.processed.txt ]; then
#       ./local/transcript_words_bpe.py \
#         --transcript $lang_dir/transcript_tokens.txt \
#         --output $lm_dir/transcript_words.processed.txt
#     fi

#     if [ ! -f $lm_dir/G_3_gram.arpa ]; then
#       ./shared/make_kn_lm.py \
#         -ngram-order 3 \
#         -text $lm_dir/transcript_words.processed.txt \
#         -lm $lm_dir/G_3_gram.arpa
#     fi

#     if [ ! -f $lm_dir/G_3_gram.fst.txt ]; then
#       log "Making kaldilm for $lm_dir/G_3_gram.arpa"
#       python3 -m kaldilm \
#         --read-symbol-table="$lang_dir/words.txt" \
#         --disambig-symbol='#0' \
#         --max-order=3 \
#         $lm_dir/G_3_gram.arpa > $lm_dir/G_3_gram.fst.txt
#     fi

#     if [ ! -f $lm_dir/G_4_gram.arpa ]; then
#       ./shared/make_kn_lm.py \
#         -ngram-order 4 \
#         -text $lm_dir/transcript_words.processed.txt \
#         -lm $lm_dir/G_4_gram.arpa
#     fi

#     if [ ! -f $lm_dir/G_4_gram.fst.txt ]; then
#       log "Making kaldilm for $lm_dir/G_4_gram.arpa"
#       python3 -m kaldilm \
#         --read-symbol-table="$lang_dir/words.txt" \
#         --disambig-symbol='#0' \
#         --max-order=4 \
#         $lm_dir/G_4_gram.arpa > $lm_dir/G_4_gram.fst.txt
#     fi
#   done
fi

# if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
#   log "Stage 9: Compile HLG"
#   ./local/compile_hlg.py --lang-dir data/lang_phone

#   for vocab_size in ${vocab_sizes[@]}; do
#     lang_dir=data/lang_bpe_${vocab_size}
#     ./local/compile_hlg.py --lang-dir $lang_dir
#   done
# fi

# # Compile LG for RNN-T fast_beam_search decoding
# if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
#   log "Stage 10: Compile LG"
#   ./local/compile_lg.py --lang-dir data/lang_phone

#   for vocab_size in ${vocab_sizes[@]}; do
#     lang_dir=data/lang_bpe_${vocab_size}
#     ./local/compile_lg.py --lang-dir $lang_dir
#   done
# fi

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
