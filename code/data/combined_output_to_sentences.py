# -*- coding: utf-8 -*-
"""

The combined output from the ordering script is converted to sentences (one reordered sentence per line). The first
line is expected to be a header and is ignored.

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


def get_lines(filepath_with_name):
    lines = []
    line_num = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line != "":
                if line_num != 0:  # skip header
                    line_split = line.split("||")
                    assert len(line_split) == 8
                    for i, one_field in enumerate(line_split):
                        if i == 7:  # reordered sentence is the final field
                            one_field = one_field.strip()
                            lines.append(one_field + "\n")
                line_num += 1
    return lines


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_file', help="input_file")
    parser.add_argument('--expected_number_of_sentences', type=int, default=100, help="expected_number_of_sentences")
    parser.add_argument('--output_file', help="output_file")

    args = parser.parse_args(arguments)

    input_file = args.input_file
    expected_number_of_sentences = args.expected_number_of_sentences
    output_file = args.output_file

    output_lines = get_lines(input_file)
    assert len(output_lines) == expected_number_of_sentences

    save_lines(output_file, output_lines)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

