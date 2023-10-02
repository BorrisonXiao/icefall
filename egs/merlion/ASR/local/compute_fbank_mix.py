#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#              2022  Johns Hopkins University.        (authors: Cihan Xiao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
This file computes fbank features of the MERLion dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import logging
import os
from pathlib import Path
import math
import copy
import argparse

import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def compute_fbank_merlion(n_factor: float = 1.0, m: int = 1, cut_name: str ="finetune"):
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    num_jobs = min(15, os.cpu_count())
    num_mel_bins = 80

    dataset_parts = (
        "SEAME",
        "dev",
    )
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    # Step 0: Split the subset to trim off the last 5% of the data (they are used for validation)
    cut_set = CutSet.from_manifests(
        recordings=manifests["dev"]["recordings"],
        supervisions=manifests["dev"]["supervisions"],
    )
    sublen = math.floor(len(cut_set) / 100 * 95)
    cut_set = cut_set.subset(first=sublen)
    cut_set = cut_set.to_eager()

    # Step 1: Add data from SEAME whose duration is n times longer than that of the merlion set while the number of English utts are the same as the Chinese utts
    dev_duration = sum([cut.duration for _, cut in cut_set.cuts.items()])
    ensubsup = manifests["SEAME"]["supervisions"].filter(
        lambda r: r.language == "English")
    en_dur = 0
    for i, sup in enumerate(ensubsup):
        en_dur += sup.duration
        if en_dur >= dev_duration * n_factor / 2:
            ensubsup = ensubsup.subset(first=max(i, 1))
            break
    cn_len = len(manifests["SEAME"]["supervisions"].filter(
        lambda r: r.language == "Chinese"))
    cnsubsup = manifests["SEAME"]["supervisions"].filter(
        lambda r: r.language == "Chinese").subset(first=max(i, 1))
    cnrest = manifests["SEAME"]["supervisions"].filter(
        lambda r: r.language == "Chinese").subset(last=cn_len - max(i, 1))
    subsup = ensubsup + cnsubsup
    seame_set = CutSet.from_manifests(
        recordings=manifests["SEAME"]["recordings"],
        supervisions=subsup,
    )
    seame_set = seame_set.to_eager()

    # Step 2: Duplicate Chinese utts in the merlion set by a factor of m
    cut_sup = cut_set.decompose()[1]
    cnsubsup = copy.deepcopy(cut_sup.filter(lambda r: r.language == "Chinese"))
    for j in range(m - 1):
        for sup in cnsubsup:
            sup.id = sup.id + f"_dup{j}"
    cn_len = len(cnsubsup)
    devsup = cut_sup + cnsubsup
    dup_dev_set = CutSet.from_manifests(
        recordings=manifests["dev"]["recordings"],
        supervisions=devsup,
    )
    dup_dev_set = dup_dev_set.to_eager()

    # Step 3: Add data from SEAME to balance the label distribution
    eng_dur = sum([sup.duration for sup in dup_dev_set.decompose()[
                  1].filter(lambda r: r.language == "English")])
    cn_dur = sum([sup.duration for sup in dup_dev_set.decompose()
                 [1].filter(lambda r: r.language == "Chinese")])

    dur_diff = eng_dur - cn_dur
    dur = 0
    for i, sup in enumerate(cnrest):
        dur += sup.duration
        if dur >= dur_diff:
            cnrest = cnrest.subset(first=i)
            break
    cn_pad_seame_set = CutSet.from_manifests(
        recordings=manifests["SEAME"]["recordings"],
        supervisions=cnrest,
    )
    cn_pad_seame_set = cn_pad_seame_set.to_eager()

    # Combine the duplicated dev set, the padded SEAME set and the additional SEAME set
    cut_set = dup_dev_set + cn_pad_seame_set + seame_set
    cut_set.describe()
    cn_dur = sum([sup.duration for sup in cut_set.decompose()[
                 1].filter(lambda r: r.language == "Chinese")])
    eng_dur = sum([sup.duration for sup in cut_set.decompose()[
                  1].filter(lambda r: r.language == "English")])
    logging.info(f"Chinese duration: {cn_dur}, English duration: {eng_dur}")

    # Step 4: Feature extraction
    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:  # Initialize the executor only once.
        cuts_filename = f"cuts_{cut_name}_{n_factor}_{m}.jsonl.gz"
        if (output_dir / cuts_filename).is_file():
            logging.info(f"File already exists - skipping.")
        logging.info(f"Processing...")
        logging.info("Applying speed perturbation")
        cut_set = (
            cut_set
            + cut_set.perturb_speed(0.9)
            + cut_set.perturb_speed(1.1)
        )
        storage_path = f"{output_dir}/feats_{cut_name}_{n_factor}_{m}"
        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=storage_path,
            # when an executor is specified, make more partitions
            num_jobs=num_jobs if ex is None else 80,
            executor=ex,
            storage_type=LilcomChunkyWriter,
        )
        # Split long cuts into many short and un-overlapping cuts
        cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
        cut_set.to_file(output_dir / cuts_filename)


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Create different finetuning split and extract the features.')
    parser.add_argument('--m', type=int, default=1,
                        help='The factor of duplication of the Chinese utts in the dev set.')
    parser.add_argument('--n-factor', type=float, default=1.0,
                        help='The ratio of the additional SEAME data to add, maximum is 1 due to size.')
    parser.add_argument('--cut-name', type=str, default="finetune",
                        help='The name of the cut.')
    
    args = parser.parse_args()

    compute_fbank_merlion(n_factor=args.n_factor, m=args.m, cut_name=args.cut_name)
