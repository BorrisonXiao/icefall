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
This file computes fbank features of the FLEURS dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import logging
import os
from pathlib import Path
import math

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


def compute_fbank_merlion():
    parti = True
    no_perturb = False
    src_dir = Path("data/manifests")
    output_dir = Path("data/fbank")
    num_jobs = min(15, os.cpu_count())
    num_mel_bins = 80

    dataset_parts = (
        # "SEAME",
        # "LibriSpeech",
        # "NSC",
        # "AISHELL",
        "dev",
        # "test",
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

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:  # Initialize the executor only once.
        for partition, m in manifests.items():
            # cuts_filename = f"cuts_{partition}.{suffix}"
            if parti:
                cuts_filename = f"cuts_dev_part.jsonl.gz"
                raw_cuts_filename = f"cuts_dev_part_raw.jsonl.gz"
            else:
                cuts_filename = f"cuts_{partition}.jsonl.gz" if not no_perturb else f"cuts_{partition}_raw.jsonl.gz"
            if (output_dir / cuts_filename).is_file():
                logging.info(f"{partition} already exists - skipping.")
                continue
            logging.info(f"Processing {partition}")
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )
            if parti:
                sublen = math.floor(len(cut_set) / 100 * 95)
                raw_set = cut_set.subset(last=len(cut_set) - sublen)
                cut_set = cut_set.subset(first=sublen)
                raw_set.describe()
                cut_set.describe()
            if "AISHELL" in partition or "LibriSpeech" in partition or "NSC" in partition or "SEAME" in partition or "dev" in partition and not no_perturb:
                logging.info("Applying speed perturbation")
                cut_set = (
                    cut_set
                    + cut_set.perturb_speed(0.9)
                    + cut_set.perturb_speed(1.1)
                )

            if parti:
                storage_path = f"{output_dir}/feats_dev_part"
                raw_storage_path = f"{output_dir}/feats_dev_part_raw"
            else:
                storage_path = f"{output_dir}/feats_{partition}" if not no_perturb else f"{output_dir}/feats_{partition}_raw"
            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=storage_path,
                # when an executor is specified, make more partitions
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )
            raw_set = raw_set.compute_and_store_features(
                extractor=extractor,
                storage_path=raw_storage_path,
                # when an executor is specified, make more partitions
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )
            # Split long cuts into many short and un-overlapping cuts
            cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
            cut_set.to_file(output_dir / cuts_filename)
            
            raw_set = raw_set.trim_to_supervisions(keep_overlapping=False)
            raw_set.to_file(output_dir / raw_cuts_filename)


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)

    compute_fbank_merlion()
