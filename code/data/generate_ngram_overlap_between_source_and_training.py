# -*- coding: utf-8 -*-
"""

For each source sentence, this script saves summary statistics regarding the n-gram overlap between the evaluation
data and the training data.

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

UNK_SYM = "$$UNK$$"
PAD_SYM = "$$PAD$$"

def get_lines(filepath_with_name):
    lines = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            lines.append(tokens)
    return lines


def get_vocab(filepath_with_name):
    vocab = {}
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            token_and_freq = line.strip().split()
            assert len(token_and_freq) == 2
            vocab[token_and_freq[0]] = int(token_and_freq[1])
    print(f"Total vocabulary size (excluding unknowns and special symbols): {len(vocab)}")
    return vocab

def collect_ngrams_for_sentence(ngram_dict, ngram_size, line_tokens):
    for i in range(0, len(line_tokens)-(ngram_size-1)):
        ngram = line_tokens[i:i+ngram_size]
        assert len(ngram) == ngram_size
        ngram = " ".join(ngram)
        if ngram in ngram_dict:
            ngram_dict[ngram] += 1
        else:
            ngram_dict[ngram] = 1
    return ngram_dict

def get_covered_ngrams_in_source(source_lines, vocab):
    bigrams = {}
    trigrams = {}
    fourgrams = {}
    fivegrams = {}

    for line in source_lines:
        line_tokens = []
        for token in line:
            if token in vocab:
                line_tokens.append(token)
            else:
                line_tokens.append(UNK_SYM)
        # collect ngrams:
        bigrams = collect_ngrams_for_sentence(bigrams, 2, [PAD_SYM] * 1 + line_tokens + [PAD_SYM] * 1)
        trigrams = collect_ngrams_for_sentence(trigrams, 3, [PAD_SYM] * 2 + line_tokens + [PAD_SYM] * 2)
        fourgrams = collect_ngrams_for_sentence(fourgrams, 4, [PAD_SYM] * 3 + line_tokens + [PAD_SYM] * 3)
        fivegrams = collect_ngrams_for_sentence(fivegrams, 5, [PAD_SYM] * 4 + line_tokens + [PAD_SYM] * 4)

    return bigrams, trigrams, fourgrams, fivegrams

def update_train_ngram_dict(line_tokens, train_ngram_dict, source_ngram_dict, ngram_size):
    _train_ngrams = collect_ngrams_for_sentence({}, ngram_size, [PAD_SYM] * (ngram_size-1) + line_tokens + [PAD_SYM] * (ngram_size-1))
    # only add the ngram if it appears in the source sentences:
    for ngram in _train_ngrams:
        if ngram in source_ngram_dict:
            if ngram in train_ngram_dict:
                train_ngram_dict[ngram] += 1
            else:
                train_ngram_dict[ngram] = 1
    return train_ngram_dict

def get_attested_training_ngrams(filepath_with_name, vocab, bigrams, trigrams, fourgrams, fivegrams):
    train_bigrams = {}
    train_trigrams = {}
    train_fourgrams = {}
    train_fivegrams = {}

    count = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            if count % 1000000 == 0:
                print(f"Currently processing line {count}")
            line = line.strip().split()
            line_tokens = []
            for token in line:
                if token in vocab:
                    line_tokens.append(token)
                else:
                    line_tokens.append(UNK_SYM)

            if len(line_tokens) > 0:
                train_bigrams = update_train_ngram_dict(line_tokens, train_bigrams, bigrams, 2)
                train_trigrams = update_train_ngram_dict(line_tokens, train_trigrams, trigrams, 3)
                train_fourgrams = update_train_ngram_dict(line_tokens, train_fourgrams, fourgrams, 4)
                train_fivegrams = update_train_ngram_dict(line_tokens, train_fivegrams, fivegrams, 5)
            count += 1
    return train_bigrams, train_trigrams, train_fourgrams, train_fivegrams


def get_ngram_count_string(line_tokens, train_ngram_dict, ngram_size):
    _source_ngrams = collect_ngrams_for_sentence({}, ngram_size,
                                                 [PAD_SYM] * (ngram_size - 1) + line_tokens + [PAD_SYM] * (
                                                 ngram_size - 1))
    ngram_total_count = 0
    ngram_covered_count = 0  # count of ngrams appearing in the training sentences
    for ngram in _source_ngrams:
        ngram_count = _source_ngrams[ngram]
        ngram_total_count += ngram_count
        if ngram in train_ngram_dict:
            ngram_covered_count += ngram_count
    return f"{ngram_total_count}\t{ngram_covered_count}"

def get_source_ngram_summary_stats(source_lines, vocab, train_bigrams, train_trigrams, train_fourgrams, train_fivegrams):
    line_stats = []

    for line in source_lines:
        line_tokens = []
        unk_count = 0
        for token in line:
            if token in vocab:
                line_tokens.append(token)
            else:
                unk_count += 1
                line_tokens.append(UNK_SYM)
        # also add unigram unknowns (note that in the larger ngrams, unknowns are folded in, as would be seen in training)

        line_ngram_count_strings = f"{len(line_tokens)}\t{len(line_tokens)-unk_count}\t"

        line_ngram_count_strings += get_ngram_count_string(line_tokens, train_bigrams, 2) + f"\t"
        line_ngram_count_strings += get_ngram_count_string(line_tokens, train_trigrams, 3) + f"\t"
        line_ngram_count_strings += get_ngram_count_string(line_tokens, train_fourgrams, 4) + f"\t"
        line_ngram_count_strings += get_ngram_count_string(line_tokens, train_fivegrams, 5) + f"\n"

        line_stats.append(line_ngram_count_strings)
    return line_stats

def save_lines(filename_with_path, list_of_strings_with_newlines):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        f.writelines(list_of_strings_with_newlines)


def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_training_sentences_file', help="training sentences")
    parser.add_argument('--input_eval_sentences_file', help="source sentences used for evaluation")
    parser.add_argument('--vocab_file', help="vocab file used by the main language model")
    parser.add_argument('--expected_number_of_sentences', type=int, default=100, help="expected_number_of_sentences")
    parser.add_argument('--output_file', help="output_file")

    args = parser.parse_args(arguments)

    input_training_sentences_file = args.input_training_sentences_file
    input_eval_sentences_file = args.input_eval_sentences_file
    vocab_file = args.vocab_file
    expected_number_of_sentences = args.expected_number_of_sentences
    output_file = args.output_file

    vocab = get_vocab(vocab_file)
    source_lines = get_lines(input_eval_sentences_file)
    assert len(source_lines) == expected_number_of_sentences
    assert UNK_SYM not in vocab
    assert PAD_SYM not in vocab


    bigrams, trigrams, fourgrams, fivegrams = get_covered_ngrams_in_source(source_lines, vocab)

    # Note: This pass over the training only collects ngrams appearing in source, as the training file may be very large
    train_bigrams, train_trigrams, train_fourgrams, train_fivegrams = get_attested_training_ngrams(input_training_sentences_file, vocab, bigrams, trigrams, fourgrams, fivegrams)

    # summary stats per line:
    # total ngrams\tcovered ngrams, where the first two (i.e., the 'unigram' entries) are the number of covered non-unknown unigrams
    line_stats = get_source_ngram_summary_stats(source_lines, vocab, train_bigrams, train_trigrams, train_fourgrams, train_fivegrams)

    save_lines(output_file, line_stats)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

