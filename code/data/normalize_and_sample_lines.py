# -*- coding: utf-8 -*-
"""

Sample X lines after normalizing tokenization (PTB->GBW tokenization)

"""

import sys
import argparse

import string
import codecs

from os import path
import random
from collections import defaultdict
import operator

import numpy as np

random.seed(1776)


def get_lines(filepath_with_name):
    lines = []
    idx = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            if idx % 10000 == 0:
                print(f"Currently processing line {idx}")

            line = line.strip()
            lines.append(line)
            idx += 1
    return lines


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def normalize_tokenizations(line):
    """
    do n't -> don 't
    ``  -> "
    ` -> '
    '' -> "
    -LRB- -> (
    -RRB- -> )

    """
    tokens = line.split()

    reformatted_line = []
    for line_i, token in enumerate(tokens):
        # need to fix n 't -> n't
        if token == "n't":
            if line_i > 0:
                prev_token = reformatted_line.pop()
                prev_token = prev_token + "n"
                reformatted_line.append(prev_token)
                reformatted_line.append("'t")
            else:
                reformatted_line.append(token)
        else:
            reformatted_line.append(token)

    normalized_line = []
    for token in reformatted_line:
        if token == "``":
            normalized_line.append('"')
        elif token == "`":
            normalized_line.append("'")
        elif token == "''":
            normalized_line.append('"')
        elif token == "-LRB-":
            normalized_line.append("(")
        elif token == "-RRB-":
            normalized_line.append(")")
        elif token == "-LCB-":
            normalized_line.append("[")
        elif token == "-RCB-":
            normalized_line.append("]")
        elif ":" in token and len(token) != 1: # times, for example, should be split: 3:35 -> 3 : 35
            token_split = token.split(":")
            assert len(token_split) == 2, "ERROR, Unexpected format"
            normalized_line.extend([token_split[0], ":", token_split[1]])
        else:
            normalized_line.append(token)

    return normalized_line



def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_file', help="input_file")
    parser.add_argument('--sample_size', type=int, default=100, help="sample size")
    parser.add_argument('--output_file', help="output_file")
    parser.add_argument('--output_shuffled_file', help="output_shuffled_file")
    parser.add_argument('--output_index_file', help="Record of indexes (indexed from 0) sampled from input_file appearing in output_file")

    args = parser.parse_args(arguments)

    input_file = args.input_file
    sample_size = args.sample_size
    output_file = args.output_file
    output_shuffled_file = args.output_shuffled_file
    output_index_file = args.output_index_file


    seed_value = 1776
    np_random_state = np.random.RandomState(seed_value)

    lines = get_lines(input_file)
    sample_indecies = np_random_state.choice(len(lines), sample_size, replace=False)
    sampled_lines = []
    sampled_shuffled_lines = []
    sampled_idx_lines = []
    for idx in sample_indecies:
        line = lines[idx]
        normalized_tokens = normalize_tokenizations(line)
        shuffled_tokens = list(normalized_tokens)
        random.shuffle(shuffled_tokens)
        sampled_lines.append(" ".join(normalized_tokens) + "\n")
        sampled_shuffled_lines.append(" ".join(shuffled_tokens) + "\n")
        sampled_idx_lines.append(f"{idx}\n")
    save_lines(output_file, sampled_lines)
    save_lines(output_shuffled_file, sampled_shuffled_lines)
    save_lines(output_index_file, sampled_idx_lines)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

