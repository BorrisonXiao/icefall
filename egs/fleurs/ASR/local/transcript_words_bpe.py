#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert the transcripts to words based on its BPE tokenization.
Essentially, the word tokenization is done by removing the special BPE token "▁".
"""
import argparse
from pathlib import Path
import re


def convert_transcript(transcript: Path, output: Path):
    with transcript.open("r") as f, output.open("w") as fout:
        for line in f:
            line = line.strip()
            line = re.sub("▁", "", line)
            line = re.sub(r"\s+", " ", line)
            print(line.strip(), file=fout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transcript",
        type=Path,
        help="The input token-level transcript file."
        "We assume that the transcript file consists of "
        "lines. Each line consists of unseparated words.",
    )
    parser.add_argument("--output", type=Path,
                        help="The output word-level transcript file.")

    args = parser.parse_args()
    convert_transcript(transcript=args.transcript, output=args.output)


if __name__ == "__main__":
    main()
