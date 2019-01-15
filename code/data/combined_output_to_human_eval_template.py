# -*- coding: utf-8 -*-
"""

The combined output from the ordering script is converted to a template for human evaluation.

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

import math

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

                    fields = []
                    for i, one_field in enumerate(line_split):
                        one_field = one_field.strip()
                        fields.append(one_field)
                    sentence_number = int(fields[0])
                    num_future_lm_unks = int(fields[1])
                    num_lm_unks = int(fields[2])
                    num_tokens = int(fields[3])
                    original_log_prob = float(fields[4])
                    reordered_log_prob = float(fields[5])
                    original_sentence = fields[6]
                    reordered_sentence = fields[7]

                    # build eval template for this sentence
                    eval_template = []
                    eval_template.append(f"----------------------------------\n")
                    eval_template.append(f"SENTENCE NUMBER: {sentence_number}\n")
                    eval_template.append(f"ORIGINAL: {original_sentence}\n")
                    eval_template.append(f"REORDERED: {reordered_sentence}\n")
                    eval_template.append(f"\tORIGINAL log prob: {original_log_prob}\n")
                    eval_template.append(f"\tREORDERED log prob: {reordered_log_prob}\n")
                    eval_template.append(f"\tExact match: {original_sentence==reordered_sentence}\n")

                    has_search_error = True
                    if reordered_log_prob >= original_log_prob or math.isclose(reordered_log_prob, original_log_prob,
                                                                               rel_tol=1e-5):
                        has_search_error = False
                    eval_template.append(f"\tSearch error: {has_search_error}\n")
                    eval_template.append(f"\tNumber of future (unigram) LM unks: {num_future_lm_unks}; Number of LM unks: {num_lm_unks}\n")
                    eval_template.append(f"\tLength: {num_tokens}\n")
                    eval_template.append(f"++Original sentence: Regular sentence: [ ]; Headline: [ ]; Source: [ ]; Other (see Notes): [ ]\n")
                    eval_template.append(f"++Original sentence: Grammatical: [ ]; Ungrammatical: [ ]\n")
                    eval_template.append(f"++Original sentence: Semantically acceptable: [ ]; Semantically unacceptable: [ ]\n")

                    eval_template.append(f"--Reordered sentence: Grammatical: [ ]; Ungrammatical: [ ]\n")
                    eval_template.append(f"--Reordered sentence: Semantically acceptable: [ ]; Semantically unacceptable: [ ]\n")

                    if original_sentence == reordered_sentence:
                        eval_template.append(f"**Original vs. Reordered sentence: Semantically identical: [ x ]; Semantically similar: [ ]; Semantically different: [ ]\n")
                    else:
                        eval_template.append(f"**Original vs. Reordered sentence: Semantically identical: [ ]; Semantically similar: [ ]; Semantically different: [ ]\n")
                    eval_template.append(f"Notes: [[[ ]]]\n")
                    eval_template.append(f"\n")
                    lines.extend(eval_template)
                line_num += 1
    return lines


def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_file', help="input_file")
    parser.add_argument('--output_file', help="output_file")

    args = parser.parse_args(arguments)

    input_file = args.input_file
    output_file = args.output_file

    output_lines = get_lines(input_file)

    save_lines(output_file, output_lines)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

