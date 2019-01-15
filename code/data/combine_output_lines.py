# -*- coding: utf-8 -*-
"""

The output from the ordering script may have been split into multiple files for efficiency. This script combines the files
and adds a header. The output needs to be in a single directory, with filenames ending in .txt (however, files with
log.txt in their name will be ignored).

This also checks that each line only occurs once and that all indices are present in [0, --expected_number_of_sentences).

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

import glob

random.seed(1776)


def get_lines(filepath_with_name, lines_dict):
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line != "":
                assert len(line.split("||")) == 8
                line_tokens = line.split()
                line_num = int(line_tokens[0])
                assert line_num not in lines_dict
                lines_dict[line_num] = line
    return lines_dict


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_dir', help="input_dir")
    parser.add_argument('--expected_number_of_sentences', type=int, default=100, help="expected_number_of_sentences")
    parser.add_argument('--output_file', help="output_file")

    args = parser.parse_args(arguments)

    input_dir = args.input_dir
    expected_number_of_sentences = args.expected_number_of_sentences
    output_file = args.output_file

    lines_dict = {}
    files_in_dir = glob.glob(path.join(input_dir, "*.txt"))
    for input_file in files_in_dir:
        if "log.txt" not in input_file:
            lines_dict = get_lines(input_file, lines_dict)

    output_lines = []
    for i in range(0, expected_number_of_sentences):
        assert i in lines_dict
        output_lines.append(lines_dict[i] + "\n")

    # add header:
    header_line = "Sentence number\t||\tNumber of Future LM (unigram) unknown tokens\t||\tNumber of LM unknown tokens\t||\tNumber of tokens\t||\tOriginal sentence log probability\t||\tReordered sentence log probability\t||\tOriginal sentence\t||\tReordered sentence"
    output_lines = [header_line + "\n"] + output_lines

    save_lines(output_file, output_lines)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

