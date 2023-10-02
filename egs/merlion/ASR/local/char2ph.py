#!/usr/bin/env python3
# 2022 Cihan Xiao

import pinyin_jyutping_sentence as pjs
from typing import TextIO
import re
import nltk
from collections import Counter

CONSONANTS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "w", "y"]


def isNotEng(word):
    """
    Return true if the char is not pure English.
    """
    return not re.match(r"^[a-zA-Z]*$", word)


def char2lex(tokens: Counter, output: TextIO, threshold: int = 1):
    """
    Convert Cantonese chars to lexicons.
    """
    tokenset = set(
        [token for token, count in tokens.items() if count >= threshold])
    arpabet = nltk.corpus.cmudict.dict()
    ools = set(tokens) - tokenset
    N = len(tokenset)
    for char in tokenset:
        if char == "<s>" or char == "(" or char == ")" or char == "[" or char == "]" or char == "#":
            continue
        clean_char = re.sub(r"\<[^\>]*\>", "", char)
        clean_char = re.sub(r"\'", "", clean_char)
        if len(clean_char) == 0:
            continue
        if isNotEng(clean_char):
            lex = char
            # Treat the filler words as a special token
            if char.startswith("(") and char.endswith(")") or char.startswith("[") and char.endswith("]"):
                lex = char
                print(f"{char} {lex}", file=output)
                continue
            if char.startswith("<mandarin>"):
                # Remove all English chars
                lex = re.sub(r"[a-zA-Z]", "", lex)
            # Remove all <> surrounded tags
            lex = re.sub(r"\<[^\>]*\>", "", lex)
            # Phonemize Chinese chars to pinyin to reduce the vocab size
            phs = re.sub("，", "", pjs.pinyin(lex, tone_numbers=True, spaces=True)).lower()
            clean_phs = []
            for ph in phs.split():
                if ph != char:
                    for con in CONSONANTS:
                        if ph.startswith(con) and len(ph) > len(con) and not ph[len(con)].isdigit():
                            ph = con + "_cn" + " " + ph[(len(con)):] + "_cn"
                            break  # Note that plural consonants are reached first

                    # # Remove the tone number
                    # ph = re.sub("[1-5]", "", ph) + "_cn"
                    clean_phs.append(ph)
                else:  # Those will be treated as OOLs since they are mostly puncs or rare Cantonese chars
                    clean_phs.append("<ool>")
            if len(clean_phs) == 0:
                print(f"{char} {char}", file=output)
            else:
                print(f"{char} {' '.join(clean_phs)}", file=output)
        else:
            # Treat the filler words as a special token
            if char.startswith("(") and char.endswith(")") or char.startswith("[") and char.endswith("]"):
                lex = char
                print(f"{char} {lex}", file=output)
                continue
            # Remove all <> surrounded tags
            lex = re.sub(r"\<[^\>]*\>", "", char)

            # Use cmudict to phonemize English words
            lex_lower = lex.lower()
            # Punctuations are removed in the text
            lex_lower = re.sub("[^a-zA-Z0-9]", "", lex_lower)
            # Numbers are mapped to themselves
            if lex_lower.isdigit():
                print(f"{char} {lex_lower}", file=output)
                continue
            # Note that OOVs will be characterized and converted
            add_s = False
            if lex_lower.endswith("'s"):
                lex_lower = lex_lower[:-2]
                add_s = True
            ph = " ".join(arpabet[lex_lower][0]).lower() if lex_lower in arpabet else "<ool>"
            if ph != "<ool>" and add_s:
                ph += " z"
            print(f"{char} {ph}", file=output)


def char2ph(tokens: Counter, output: TextIO, threshold: int = 1):
    """
    Convert Cantonese chars to lexicons.
    """
    tokenset = set(
        [token for token, count in tokens.items() if count >= threshold])
    arpabet = nltk.corpus.cmudict.dict()
    ools = set(tokens) - tokenset
    N = len(tokenset)
    for char in tokenset:
        if char == "<s>":
            continue
        lex = " ".join(char.split("_"))
        print(f"{char} {lex}", file=output)
        continue
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
            # Punctuations are removed in the text
            char_lower = re.sub("[^a-zA-Z0-9]", "", char_lower)
            # Numbers are mapped to themselves
            if char_lower.isdigit():
                print(f"{char} {char}", file=output)
                continue
            # Note that OOVs will be characterized and converted
            ph = " ".join(arpabet[char_lower][0]).lower() if char_lower in arpabet else " ".join([
                " ".join(arpabet[c][0]).lower() for c in char_lower])
            print(f"{char} {ph}", file=output)

    print(
        f"There are {len(ools)} OOVs among {N} word types, i.e. OOV rate = {len(ools) / N * 100:.2f}%.")
    print(ools)
