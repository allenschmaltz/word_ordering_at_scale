#!/usr/bin/env python3 -u

# Original License:

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Print the perplexity of the ordered sentences (and a as a check, the shuffled sentences, as well)
"""

import numpy as np
import torch

from fairseq import options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter

from fairseq.tokenizer import Tokenizer

import codecs
import random
import copy
from collections import namedtuple
from collections import defaultdict

import re

import time


SPACE_NORMALIZER = re.compile(r"\s+")
OUTPUT_LINE_SEPARATOR = "\t||\t"

def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def advance(model, input_token_ids, ouput_token_ids, use_cuda, return_value_type):  # non-batched
    # return_value_type:
    # if 1, return the sum of log prob.'s over all ouput_token_ids
    # if 2, return log prob. of final element of ouput_token_ids
    # if 3, return log prob. of final two elements of ouput_token_ids (for use when scoring final eos)
    assert len(input_token_ids) == len(ouput_token_ids)
    single_net_input_dict = {}
    single_target_dict = {}
    if use_cuda:
        single_net_input_dict["src_tokens"] = torch.tensor([input_token_ids]).cuda()
        single_net_input_dict["src_lengths"] = torch.tensor([[len(input_token_ids)]]).cuda()
        single_target_dict['target'] = torch.tensor([ouput_token_ids]).cuda()
    else:
        single_net_input_dict["src_tokens"] = torch.tensor([input_token_ids])
        single_net_input_dict["src_lengths"] = torch.tensor([[len(input_token_ids)]])
        single_target_dict['target'] = torch.tensor([ouput_token_ids])
    with torch.no_grad():
        model.eval()
        single_target_decoder_out = model.forward(**single_net_input_dict)
        # attn = decoder_out[1] ## attn is always expected to be None here
    single_probs = model.get_normalized_probs(single_target_decoder_out, log_probs=True,
                                                       sample=single_target_dict).data

    single_probs = single_probs.gather(
        dim=2,
        index=single_target_dict['target'].data.unsqueeze(-1),
    ).squeeze(2)

    if return_value_type == 1:
        return torch.sum(single_probs).item()
    elif return_value_type == 2:
        return single_probs[0][len(ouput_token_ids)-1].item()
    elif return_value_type == 3:
        return single_probs[0][len(ouput_token_ids) - 2].item() + single_probs[0][len(ouput_token_ids) - 1].item()
    else:
        assert False




def main(parsed_args):
    assert parsed_args.path is not None, '--path required for evaluation!'
    assert parsed_args.input_ordered_file != "", '--input_ordered_file required for evaluation!'
    assert parsed_args.input_shuffled_file != "", '--input_shuffled_file required for evaluation!'

    print(parsed_args)

    use_cuda = torch.cuda.is_available() and not parsed_args.cpu

    task = tasks.setup_task(parsed_args)

    # Load ensemble
    print('| loading model(s) from {}'.format(parsed_args.path))
    models, args = utils.load_ensemble_for_inference(parsed_args.path.split(':'), task, model_arg_overrides=eval(parsed_args.model_overrides))

    for arg in vars(parsed_args).keys():
        if arg not in {'self_target', 'future_target', 'past_target', 'tokens_per_sample', 'output_size_dictionary'}:
            setattr(args, arg, getattr(parsed_args, arg))
    task = tasks.setup_task(args)

    print(f"Reference file: {parsed_args.input_ordered_file}")
    print(f"Shuffled file: {parsed_args.input_shuffled_file}")

    # Optimize ensemble for generation and set the source and dest dicts on the model (required by scorer)
    for model in models:
        model.make_generation_fast_()
        if args.fp16:
            model.half()

    assert len(models) == 1

    print('num. model params: {}'.format(sum(p.numel() for p in models[0].parameters())))

    lm_input_dictionary = task.dictionary
    vocab = task.target_dictionary

    #print(f"len(lm_input_dictionary): {len(lm_input_dictionary)}")
    #print(f"len(lm_dictionary): {len(vocab)}")
    print(f"Size of vocabulary dictionary: {len(vocab)}")
    assert len(lm_input_dictionary) == len(vocab)
    # print(lm_input_dictionary.eos())
    # print(vocab.eos())

    model = models[0]
    if use_cuda:
        model.cuda()

    #######################################
    eos_id = lm_input_dictionary.eos()

    ordered_input_file_object = codecs.open(parsed_args.input_ordered_file, encoding="utf-8")
    sentence_num = 0

    total_shuffled_neg_log_prob = 0.0
    total_original_neg_log_prob = 0.0
    total_num_tokens = 0

    with codecs.open(parsed_args.input_shuffled_file, encoding="utf-8") as f:
        for line in f:
            if sentence_num >= parsed_args.ordering_start_sentence and sentence_num <= parsed_args.ordering_end_sentence:

                token_ids_tensor = Tokenizer.tokenize(
                    line, lm_input_dictionary, add_if_not_exist=False,
                    append_eos=False, reverse_order=False,
                ).long()

                token_ids = [t for t in token_ids_tensor]
                tokens = tokenize_line(line)
                assert len(token_ids) == len(tokens)

                # print(f"------------------Sentence: {sentence_num}")
                # print(f"Length: {len(tokens)}")

                shuffled_log_prob = advance(model, [eos_id] + token_ids, token_ids + [eos_id], use_cuda, 1)
                total_shuffled_neg_log_prob += -1*shuffled_log_prob

                original_line = ordered_input_file_object.readline()
                original_token_ids_tensor = Tokenizer.tokenize(
                    original_line, lm_input_dictionary, add_if_not_exist=False,
                    append_eos=False, reverse_order=False,
                ).long()

                original_token_ids = [t for t in original_token_ids_tensor]
                original_tokens = tokenize_line(original_line)
                assert len(original_token_ids) == len(original_tokens) and len(original_tokens) == len(tokens)

                original_log_prob = advance(model, [eos_id] + original_token_ids, original_token_ids + [eos_id], use_cuda, 1)
                total_original_neg_log_prob += -1*original_log_prob
                total_num_tokens += len(original_token_ids) + 1  # +1 for the eos_id

            else:
                ordered_input_file_object.readline()  # advance file to maintain alignment with the shuffled file

            sentence_num += 1

    shuffled_perplexity = np.exp(total_shuffled_neg_log_prob / total_num_tokens)
    original_perplexity = np.exp(total_original_neg_log_prob / total_num_tokens)

    print(f"Shuffled perplexity: {shuffled_perplexity}")
    print(f"Original perplexity: {original_perplexity}")
    print(f"Total tokens (including eos): {total_num_tokens}")

if __name__ == '__main__':
    parser = options.get_eval_lm_ordering_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
