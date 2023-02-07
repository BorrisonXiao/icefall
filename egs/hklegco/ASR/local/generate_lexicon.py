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
from char2ph import char2ph
from tqdm import tqdm
from collections import Counter

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def generate_lexicon(manifests_dir: Path, lexicon: Path):
    prefix = "hklegco"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=["train"],
        output_dir=manifests_dir,
        suffix=suffix,
        prefix=prefix,
    )
    assert manifests is not None

    assert len(manifests) == 1, (
        len(manifests),
        list(manifests.keys())
    )

    # First generate a token list from the training set
    supervisions = manifests["train"]["supervisions"]
    tokens = Counter()
    # Count each token's occurrence
    for s in tqdm(supervisions, desc="Generating lexicon"):
        tokens.update(s.text.split())

    # Then generate a lexicon from the token list
    with open(lexicon, "w") as f:
        char2ph(tokens, f, threshold=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Directory where manifests are stored",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=Path("data/lang_phone/lexicon.txt"),
        help="Path to the lexicon file",
    )
    args = parser.parse_args()
    generate_lexicon(args.manifests_dir, args.lexicon)


if __name__ == "__main__":
    main()
