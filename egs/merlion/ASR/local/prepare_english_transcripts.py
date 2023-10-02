#!/usr/bin/env python3
# Copyright    2022  Johns Hopkins University.        (authors: Cihan Xiao)
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
This file generates lexicon.txt from the HKLEGCO's training dataset.
It looks for manifests in the directory data/manifests.
"""

import argparse
from pathlib import Path

import torch
from lhotse.recipes.utils import read_manifests_if_cached
from tqdm import tqdm
import re

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def transcribe_tokens(manifests_dir: Path, output: Path):
    prefix = ""
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=["NSC", "LibriSpeech"],
        output_dir=manifests_dir,
        suffix=suffix,
        prefix=prefix,
    )
    assert manifests is not None

    assert len(manifests) == 2, (
        len(manifests),
        list(manifests.keys())
    )

    # First generate a token list from the training set
    nsc_supervisions = manifests["NSC"]["supervisions"]
    librispeech_supervisions = manifests["LibriSpeech"]["supervisions"]

    # Then generate a lexicon from the token list
    with open(output, "w") as f:
        for s in tqdm(nsc_supervisions, desc="Generating transcript_words_en.txt"):
            # print(s.text.lower(), file=f)
            # continue
            # Remove the non-English content wrapped within <mandarin> and </mandarin>
            filtered = re.sub(r"\<mandarin\>.*?\<\/mandarin\>", "", s.text)
            # # The filler words in [] and () are removed
            # filtered = re.sub(r"\[.*?\]", "", s.text)
            # filtered = re.sub(r"\(.*?\)", "", filtered)
            # # Punctuations are also removed
            # filtered = re.sub(r"[^\w\s]", "", filtered)
            # # The extra spaces are removed
            # filtered = re.sub(r"\s+", " ", filtered).strip()
            # The _ is removed
            filtered = re.sub("_", "", filtered)
            if len(filtered) > 0:
                print(filtered, file=f)
        for s in tqdm(librispeech_supervisions, desc="Generating transcript_words_en.txt"):
            print(s.text.lower(), file=f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Directory where manifests are stored",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output file",
    )
    args = parser.parse_args()
    transcribe_tokens(args.manifests_dir, args.output)


if __name__ == "__main__":
    main()