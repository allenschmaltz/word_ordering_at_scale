# -*- coding: utf-8 -*-
"""

This prints out summary stats for the filled-out human evaluation template.

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

import matplotlib.pyplot as plt
import matplotlib

random.seed(1776)

ROUND_CONSTANT = 3


def parse_bracket_line_and_return_contents(line, num_brackets):

    assert line.count("[") == num_brackets and line.count("]") == num_brackets, "ERROR: the line does not contain num_brackets brackets"
    contents = []
    for _ in range(0, num_brackets):
        left_index = line.find("[")
        right_index = line.find("]")
        contents.append(line[left_index+1:right_index].strip())
        line = line[right_index+1:]
    return contents

def validate_contents(contents):
    # check that one and only one field has been selected
    selected = 0
    for index in range(0, len(contents)):
        if is_content_true_at_index(contents, index, False):
            selected += 1
    assert selected == 1, f"{selected}"

def is_content_true_at_index(contents, index, run_validation=True):
    if run_validation:
        validate_contents(contents)
    if contents[index] != "":
        assert contents[index] == "x"
        return True
    return False


def round_value(value):
    return round(value, ROUND_CONSTANT)

def get_lines(filepath_with_name, ngram_summary_stats, restrict_to_non_search_error_sentences, length_restriction, length_restriction_lte, generate_graphs, graph_output_file):

    line_num = 0
    reading_sentence = False

    all_num_tokens = []  # length for all sentences (without restrictions)

    valid_sentences_wihout_search_error = []  # this always excludes search error, whereas total_valid_sentences may not depending on commandline option
    total_valid_sentences = 0
    target_sentences = 0
    target_sentences_exact_match = []

    unigram_counts_valid = []
    unigram_counts_problem = []

    bigram_counts_valid = []
    bigram_counts_problem = []

    trigram_counts_valid = []
    trigram_counts_problem = []

    fourgram_counts_valid = []
    fourgram_counts_problem = []

    fivegram_counts_valid = []
    fivegram_counts_problem = []

    ngram_counts_valid_num_tokens = []
    ngram_counts_problem_num_tokens = []

    num_semantics_identical = 0
    num_semantics_similar = 0
    num_semantics_different = 0
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line != "":
                if line.startswith("----------------------------------"):
                    # save previous sentence data
                    reading_sentence = True

                    sentence_number = None
                    original_sentence = None
                    reordered_sentence = None
                    original_log_prob = None
                    reordered_log_prob = None
                    is_exact_match = None
                    has_search_error = None
                    num_future_lm_unks = None
                    num_lm_unks = None
                    num_tokens = None
                    original_sentence_type_contents = None
                    original_sentence_grammar_contents = None
                    original_sentence_semantics_contents = None
                    reordered_sentence_grammar_contents = None
                    reordered_sentence_semantics_contents = None
                    comparative_semantics_contents = None

                elif line.startswith("Notes: [[["):
                    reading_sentence = False

                    # do analysis here:
                    ngrams = ngram_summary_stats[line_num]

                    all_num_tokens.append(num_tokens)

                    original_is_regular_sentence = is_content_true_at_index(original_sentence_type_contents, 0)
                    original_grammar_correct = is_content_true_at_index(original_sentence_grammar_contents, 0)
                    original_semantics_correct = is_content_true_at_index(original_sentence_semantics_contents, 0)

                    reordered_grammar_correct = is_content_true_at_index(reordered_sentence_grammar_contents, 0)
                    reordered_semantics_correct = is_content_true_at_index(reordered_sentence_semantics_contents, 0)

                    semantics_different = is_content_true_at_index(comparative_semantics_contents, 2)

                    if original_is_regular_sentence and original_grammar_correct and original_semantics_correct:
                        if not has_search_error:
                            valid_sentences_wihout_search_error.append(1)
                        else:
                            valid_sentences_wihout_search_error.append(0)

                    # check if sentence meets optional restrictions (provided by command line arguments):
                    additional_restriction_met = True
                    if restrict_to_non_search_error_sentences:
                        if has_search_error:
                            additional_restriction_met = False

                    if length_restriction != 0 and length_restriction_lte:
                        if num_tokens > length_restriction:
                            additional_restriction_met = False
                    else:
                        if num_tokens <= length_restriction:
                            additional_restriction_met = False

                    if original_is_regular_sentence and original_grammar_correct and original_semantics_correct and additional_restriction_met:
                        total_valid_sentences += 1

                        if is_exact_match:
                            assert reordered_grammar_correct and reordered_semantics_correct
                            target_sentences_exact_match.append(1)
                        else:
                            target_sentences_exact_match.append(0)

                        if reordered_grammar_correct and reordered_semantics_correct:
                            target_sentences += 1
                            ngram_counts_valid_num_tokens.append(num_tokens)

                            assert num_lm_unks == ngrams[0] - ngrams[1]
                            unigram_counts_valid.append(ngrams[1] / ngrams[0])
                            bigram_counts_valid.append(ngrams[3] / ngrams[2])
                            trigram_counts_valid.append(ngrams[5] / ngrams[4])
                            fourgram_counts_valid.append(ngrams[7] / ngrams[6])
                            fivegram_counts_valid.append(ngrams[9] / ngrams[8])


                            if is_content_true_at_index(comparative_semantics_contents, 0):
                                num_semantics_identical += 1
                            #     if original_sentence != reordered_sentence:
                            #        print(
                            #         f"Sentence is grammatically and semantically acceptable AND semantically identical (but not verbatim): Sent: {sentence_number}; 5-gram coverage: {ngrams[9] / ngrams[8]}\n\tOriginal: {original_sentence};\n\tReordered: {reordered_sentence}")

                            if is_content_true_at_index(comparative_semantics_contents, 1):
                                num_semantics_similar += 1
                                assert original_sentence != reordered_sentence
                                #print(
                                #    f"Sentence is grammatically and semantically acceptable AND semantically similar: Sent: {sentence_number}; 5-gram coverage: {ngrams[9] / ngrams[8]}\n\tOriginal: {original_sentence};\n\tReordered: {reordered_sentence}")

                            if semantics_different:
                                num_semantics_different += 1
                                assert original_sentence != reordered_sentence
                                # print(
                                #     f"Sentence is grammatically and semantically acceptable AND semantically different: Sent: {sentence_number}; 5-gram coverage: {ngrams[9] / ngrams[8]}\n\tOriginal: {original_sentence};\n\tReordered: {reordered_sentence}")

                        else:
                            ngram_counts_problem_num_tokens.append(num_tokens)
                            assert num_lm_unks == ngrams[0] - ngrams[1]
                            unigram_counts_problem.append(ngrams[1] / ngrams[0])
                            bigram_counts_problem.append(ngrams[3] / ngrams[2])
                            trigram_counts_problem.append(ngrams[5] / ngrams[4])
                            fourgram_counts_problem.append(ngrams[7] / ngrams[6])
                            fivegram_counts_problem.append(ngrams[9] / ngrams[8])


                        #TEMP -- can uncomment these and other print() statements elsewhere to see applicable examples
                        #if (not reordered_grammar_correct) and reordered_semantics_correct:
                        #if (not reordered_grammar_correct) and (not reordered_semantics_correct):
                        #     print(
                        #         f"Problematic Sent: {sentence_number}; 5-gram coverage: {ngrams[9] / ngrams[8]}\n\tOriginal: {original_sentence};\n\tReordered: {reordered_sentence}")


                    # if ngrams[9] / ngrams[8] == 1:
                        #     print(original_sentence == reordered_sentence)
                        #     print(f"Sentence with fully-covered 5-grams: Original: {original_sentence};\nReordered: {reordered_sentence}")


                    line_num += 1
                elif reading_sentence:
                    if line.startswith("SENTENCE NUMBER:"):
                        sentence_number = int(line[len("SENTENCE NUMBER:"):].strip())
                        assert sentence_number == line_num, f"{sentence_number}, {line_num}"
                    elif line.startswith("ORIGINAL:"):
                        original_sentence = line[len("ORIGINAL:"):].strip()
                    elif line.startswith("REORDERED:"):
                        reordered_sentence = line[len("REORDERED:"):].strip()
                    elif line.startswith("ORIGINAL log prob:"):
                        original_log_prob = float(line[len("ORIGINAL log prob:"):].strip())
                    elif line.startswith("REORDERED log prob:"):
                        reordered_log_prob = float(line[len("REORDERED log prob:"):].strip())
                    elif line.startswith("Exact match:"):
                        is_exact_match = line[len("Exact match:"):].strip() == "True"
                        if is_exact_match:
                            assert original_sentence == reordered_sentence
                        else:
                            assert original_sentence != reordered_sentence
                    elif line.startswith("Search error:"):
                        has_search_error = line[len("Search error:"):].strip() == "True"
                        if has_search_error:
                            assert original_sentence != reordered_sentence
                            assert not (reordered_log_prob >= original_log_prob or math.isclose(reordered_log_prob, original_log_prob, rel_tol=1e-5))
                        else:
                            assert reordered_log_prob >= original_log_prob or math.isclose(reordered_log_prob, original_log_prob, rel_tol=1e-5)
                    elif line.startswith("Number of future (unigram) LM unks:"):
                        unk_line = line[len("Number of future (unigram) LM unks:"):].strip()
                        semicolon_index = unk_line.find(";")
                        num_future_lm_unks = int(unk_line[0:semicolon_index])
                        num_lm_unks = int(unk_line[unk_line.rfind(":")+1:])
                    elif line.startswith("Length:"):
                        num_tokens = int(line[len("Length:"):].strip())
                        assert num_tokens == len(original_sentence.split()) and num_tokens == len(reordered_sentence.split())
                    elif line.startswith("++Original sentence: Regular sentence:"):
                        original_sentence_type_contents = parse_bracket_line_and_return_contents(line, 4)
                    elif line.startswith("++Original sentence: Grammatical:"):
                        original_sentence_grammar_contents = parse_bracket_line_and_return_contents(line, 2)
                    elif line.startswith("++Original sentence: Semantically acceptable:"):
                        original_sentence_semantics_contents = parse_bracket_line_and_return_contents(line, 2)
                    elif line.startswith("--Reordered sentence: Grammatical:"):
                        reordered_sentence_grammar_contents = parse_bracket_line_and_return_contents(line, 2)
                    elif line.startswith("--Reordered sentence: Semantically acceptable:"):
                        reordered_sentence_semantics_contents = parse_bracket_line_and_return_contents(line, 2)
                    elif line.startswith("**Original vs. Reordered sentence: Semantically identical:"):
                        comparative_semantics_contents = parse_bracket_line_and_return_contents(line, 3)


    if generate_graphs:
        FONTNAME = "Times New Roman"
        FONTSIZE = 11

        fig, ax = plt.subplots(2,2, sharex=True, sharey=True)

        matplotlib.rcParams['font.family'] = "serif"
        ax[0, 0].set_title("Bigrams", fontname=FONTNAME, fontsize=FONTSIZE)
        ax[0, 0].plot(ngram_counts_valid_num_tokens, bigram_counts_valid, 'o')
        ax[0, 0].plot(ngram_counts_problem_num_tokens, bigram_counts_problem, 'o', color="darkred")

        ax[0, 0].axhline(y=np.mean(bigram_counts_valid), linestyle='-')
        ax[0, 0].axhline(y=np.mean(bigram_counts_problem), color="darkred", linestyle='-')

        ax[0, 1].set_title("Trigrams", fontname=FONTNAME, fontsize=FONTSIZE)
        ax[0, 1].plot(ngram_counts_valid_num_tokens, trigram_counts_valid, 'o')
        ax[0, 1].plot(ngram_counts_problem_num_tokens, trigram_counts_problem, 'o', color="darkred")

        ax[0, 1].axhline(y=np.mean(trigram_counts_valid), linestyle='-')
        ax[0, 1].axhline(y=np.mean(trigram_counts_problem), color="darkred", linestyle='-')

        ax[1, 0].set_title("4-grams", fontname=FONTNAME, fontsize=FONTSIZE)
        ax[1, 0].plot(ngram_counts_valid_num_tokens, fourgram_counts_valid, 'o')
        ax[1, 0].plot(ngram_counts_problem_num_tokens, fourgram_counts_problem, 'o', color="darkred")

        ax[1, 0].axhline(y=np.mean(fourgram_counts_valid), linestyle='-')
        ax[1, 0].axhline(y=np.mean(fourgram_counts_problem), color="darkred", linestyle='-')

        ax[1, 1].set_title("5-grams", fontname=FONTNAME, fontsize=FONTSIZE)
        ax[1, 1].plot(ngram_counts_valid_num_tokens, fivegram_counts_valid, 'o')
        ax[1, 1].plot(ngram_counts_problem_num_tokens, fivegram_counts_problem, 'o', color="darkred")

        ax[1, 1].axhline(y=np.mean(fivegram_counts_valid), linestyle='-')
        ax[1, 1].axhline(y=np.mean(fivegram_counts_problem), color="darkred", linestyle='-')

        for row in range(0, 2):
            for col in range(0, 2):
                ax[row, col].spines['top'].set_visible(False)
                ax[row, col].spines['right'].set_visible(False)
                ax[row, col].set_xlim(0, 50)
                ax[row, col].set_ylim(0, 1.1)

        fig.text(0.02, 0.5, 'Proportion of Covered N-grams', va='center', rotation='vertical')
        fig.text(0.5, 0.02, 'Sentence Length', ha='center')

        plt.savefig(graph_output_file)



    print(f"Average number of tokens (across all sentences): {np.mean(all_num_tokens)}")
    print(f"Proportion of valid sentences without search error: {round_value(np.mean(valid_sentences_wihout_search_error))} ({np.sum(valid_sentences_wihout_search_error)}/{len(valid_sentences_wihout_search_error)})")

    if restrict_to_non_search_error_sentences:
        print(f"Valid sentences are those in which the source is an acceptable sentence AND the reordered sentence does not have source-relative search errors:")
    else:
        print(f"Valid sentences are those in which the source is an acceptable sentence:")

    if length_restriction > 0:
        print(f"\t(Length restrictions are in effect: {length_restriction})")
    print(
    f"\t{target_sentences} grammatically and semantically acceptable re-ordered sentences out of {total_valid_sentences} valid source sentences: {round_value(target_sentences/total_valid_sentences)}")

    print(f"Proportion of valid re-ordered sentences exactly matching the source: {round_value(np.mean(target_sentences_exact_match))} ({np.sum(target_sentences_exact_match)} / {len(target_sentences_exact_match)})")

    print(f"Average proportion of tokens in the LM vocab: valid: {round_value(np.mean(unigram_counts_valid))}; problem: {round_value(np.mean(unigram_counts_problem))}")
    print(f"Average proportion of covered 2-grams: valid: {round_value(np.mean(bigram_counts_valid))}; problem: {round_value(np.mean(bigram_counts_problem))}")
    print(f"Average proportion of covered 3-grams: valid: {round_value(np.mean(trigram_counts_valid))}; problem: {round_value(np.mean(trigram_counts_problem))}")
    print(f"Average proportion of covered 4-grams: valid: {round_value(np.mean(fourgram_counts_valid))}; problem: {round_value(np.mean(fourgram_counts_problem))}")
    print(f"Average proportion of covered 5-grams: valid: {round_value(np.mean(fivegram_counts_valid))}; problem: {round_value(np.mean(fivegram_counts_problem))}")

    print(
    f"Max proportion of covered 5-grams: valid: {np.max(fivegram_counts_valid)}; problem: {np.max(fivegram_counts_problem)}")

    print(f"Number of sentences with fully covered 5-grams: {fivegram_counts_valid.count(1)}; problem: {fivegram_counts_problem.count(1)}")

    print(f"Average number of re-ordered tokens in valid + problem: {round_value(np.mean(ngram_counts_valid_num_tokens+ngram_counts_problem_num_tokens))}")
    print(f"Average number of re-ordered tokens in valid: {round_value(np.mean(ngram_counts_valid_num_tokens))}; problem: {round_value(np.mean(ngram_counts_problem_num_tokens))}")
    print(f"Min number of re-ordered tokens in valid: {np.min(ngram_counts_valid_num_tokens)}; problem: {np.min(ngram_counts_problem_num_tokens)}")
    print(f"Max number of re-ordered tokens in valid: {np.max(ngram_counts_valid_num_tokens)}; problem: {np.max(ngram_counts_problem_num_tokens)}")

    print(f"Number of valid re-ordered sentences that are (relative to the valid source) semantically: identical: {num_semantics_identical} ({round_value(num_semantics_identical/target_sentences)}), similar: {num_semantics_similar} ({round_value(num_semantics_similar/target_sentences)}), different: {num_semantics_different} ({round_value(num_semantics_different/target_sentences)}); out of {target_sentences}")


def get_ngram_summary_stats(filepath_with_name):
    ngram_summary_stats = []
    with codecs.open(filepath_with_name, encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            assert len(tokens) == 10
            ngram_summary_stats.append([int(x) for x in tokens])
    return ngram_summary_stats

def main(arguments):

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_file', help="the filled-out human eval template")
    parser.add_argument('--ngram_summary_stats_file', help="summary stats on the ngrams covered in training (relative to the original source file)")
    parser.add_argument('--restrict_to_non_search_error_sentences', action='store_true', help="If provided, the core summary statistics are calculated on the subset of sentences without source-relative search errors.")
    parser.add_argument('--length_restriction', type=float, default=0.0, help="If > 0, the core summary statistics are calculated on the subset of sentences that are only greater than (or less than or equal, if --length_restriction_lte) this length.")
    parser.add_argument('--length_restriction_lte', action='store_true', help="If provided (along with --length_restriction), results in the core summary statistics being calculated only on sentences less than or equal to the provided length.")
    parser.add_argument('--generate_graphs', action='store_true', help="If provided, generate the analysis graphs.")
    parser.add_argument('--graph_output_file', default="", help="File in which to save graphs, if --generate_graphs")


    args = parser.parse_args(arguments)
    input_file = args.input_file
    ngram_summary_stats_file = args.ngram_summary_stats_file
    restrict_to_non_search_error_sentences = args.restrict_to_non_search_error_sentences
    length_restriction = args.length_restriction
    assert length_restriction >= 0
    length_restriction_lte = args.length_restriction_lte
    generate_graphs = args.generate_graphs
    graph_output_file = args.graph_output_file

    ngram_summary_stats = get_ngram_summary_stats(ngram_summary_stats_file)
    get_lines(input_file, ngram_summary_stats, restrict_to_non_search_error_sentences, length_restriction, length_restriction_lte, generate_graphs, graph_output_file)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

