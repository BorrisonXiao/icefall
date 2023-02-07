#!/usr/bin/env python3
# 2022 Cihan Xiao

import pinyin_jyutping_sentence as pjs
from typing import TextIO
import re
import nltk
from collections import Counter

CONSONANTS = ["gw", "kw", "ng", "b", "c", "d", "f", "g", "h", "j", "k",
              "l", "m", "n", "p", "s", "t", "w", "z"]


def isEng(word):
    """
    Return true if the char is an English word.
    """
    return re.search('[a-zA-Z]', word) != None


def char2ph(tokens: Counter, output: TextIO, threshold: int = 1):
    """
    Convert Cantonese chars to lexicons.
    """
    tokenset = set([token for token, count in tokens.items() if count >= threshold])
    arpabet = nltk.corpus.cmudict.dict()
    ools = set(tokens) - tokenset
    N = len(tokenset)
    for char in tokenset:
        if not isEng(char):  # Avoid phonemize some english letters
            # Seems pjs has some bug that might produce unexpected "，"
            ph = re.sub("，", "", pjs.jyutping(char, tone_numbers=True))
            if ph != char:
                for con in CONSONANTS:
                    if ph.startswith(con) and not ph[len(con)].isdigit():
                        ph = con + " " + ph[(len(con)):]
                        break  # Note that plural consonants are reached first
                print(f"{char} {ph}", file=output)
            else:  # Those will be treated as OOLs since they are mostly puncs or rare Cantonese chars
                ools.add(char)
                print(f"{char} <ool>", file=output)
        else:
            # Use cmudict to phonemize English words
            char_lower = char.lower()
            # Note that OOVs will be characterized and converted
            ph = " ".join(arpabet[char_lower][0]).lower() if char_lower in arpabet else " ".join([
                " ".join(arpabet[c][0]).lower() for c in char_lower])
            print(f"{char} {ph}", file=output)

    print(
        f"There are {len(ools)} OOVs among {N} word types, i.e. OOV rate = {len(ools) / N * 100:.2f}%.")
    print(ools)
