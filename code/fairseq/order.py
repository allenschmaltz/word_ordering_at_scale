#!/usr/bin/env python3 -u

# Original License:

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

"""
Order shuffled input with a trained language model.
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


## temp global
global global_max_batch_size
global global_batch_size
global global_batch_num
global_max_batch_size = 0
global_batch_size = 0
global_batch_num = 0

SPACE_NORMALIZER = re.compile(r"\s+")
OUTPUT_LINE_SEPARATOR = "\t||\t"


Hypothesis = namedtuple("Hypothesis", ['score', 'last_action', "bow", "future_score", "current_sequence", "last_beam"])


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


def _batch_advance(model, input_token_ids, intput_token_ids_lens, ouput_token_ids, use_cuda, final_action):

    global global_max_batch_size
    global global_batch_size
    global global_batch_num

    if len(input_token_ids) > global_max_batch_size:
        global_max_batch_size = len(input_token_ids)
    global_batch_size += len(input_token_ids)
    global_batch_num += 1

    # if not final_action, return log prob. of final element of ouput_token_ids
    # if final_action, return log prob. of final two elements of ouput_token_ids (for use when scoring final eos)
    assert len(input_token_ids) == len(ouput_token_ids)
    single_net_input_dict = {}
    single_target_dict = {}
    if use_cuda:
        single_net_input_dict["src_tokens"] = torch.tensor(input_token_ids).cuda()
        single_net_input_dict["src_lengths"] = torch.tensor(intput_token_ids_lens).cuda()
        single_target_dict['target'] = torch.tensor(ouput_token_ids).cuda()
    else:
        single_net_input_dict["src_tokens"] = torch.tensor(input_token_ids)
        single_net_input_dict["src_lengths"] = torch.tensor(intput_token_ids_lens)
        single_target_dict['target'] = torch.tensor(ouput_token_ids)
    with torch.no_grad():
        model.eval()
        single_target_decoder_out = model.forward(**single_net_input_dict)
        # attn = decoder_out[1] ## attn is always expected to be None here
    batch_probs = model.get_normalized_probs(single_target_decoder_out, log_probs=True,
                                                       sample=single_target_dict).data

    batch_probs = batch_probs.gather(
        dim=2,
        index=single_target_dict['target'].data.unsqueeze(-1),
    ).squeeze(2)

    if not final_action:
        return batch_probs[:,len(ouput_token_ids[0])-1]  # returns a tensor of shape Size[batch_size] -- this works because all batches are the same length
    elif final_action:  # all beams end at same beam step
        return batch_probs[:,len(ouput_token_ids[0]) - 2] + batch_probs[:,len(ouput_token_ids[0]) - 1]
    else:
        assert False

def _packed_batch_advance(model, input_token_ids, intput_token_ids_lens, ouput_token_ids_grouped, use_cuda, final_action):

    global global_max_batch_size
    global global_batch_size
    global global_batch_num

    if len(input_token_ids) > global_max_batch_size:
        global_max_batch_size = len(input_token_ids)
    global_batch_size += len(input_token_ids)
    global_batch_num += 1

    # if not final_action, return log prob. of final element of ouput_token_ids
    # if final_action, return log prob. of final two elements of ouput_token_ids (for use when scoring final eos)
    assert len(input_token_ids) == len(ouput_token_ids_grouped)
    single_net_input_dict = {}
    assert use_cuda
    assert not final_action
    single_net_input_dict["src_tokens"] = torch.tensor(input_token_ids).cuda()
    single_net_input_dict["src_lengths"] = torch.tensor(intput_token_ids_lens).cuda()


    with torch.no_grad():
        model.eval()
        single_target_decoder_out = model.forward(**single_net_input_dict)
        # attn = decoder_out[1] ## attn is always expected to be None here

    ouput_token_ids_grouped = np.array(ouput_token_ids_grouped)
    output_probs = np.zeros((ouput_token_ids_grouped.shape[0],ouput_token_ids_grouped.shape[1]))  # hyps X actions

    for action_i in range(0, ouput_token_ids_grouped.shape[1]):
        ouput_token_ids = ouput_token_ids_grouped[:,action_i]
        single_target_dict = {}
        single_target_dict['target'] = torch.tensor(ouput_token_ids).cuda()
        batch_probs = model.get_normalized_probs(single_target_decoder_out, log_probs=True,
                                                       sample=single_target_dict).data
        batch_probs = batch_probs.gather(
            dim=2,
            index=single_target_dict['target'].data.unsqueeze(-1),
        ).squeeze(2)

        output_probs[:,action_i] = batch_probs[:, len(ouput_token_ids[0]) - 1].cpu().numpy()
    return output_probs.flatten(order='C')

def future(bow, futurelm):
    """
    Pre-condition: action has been removed from bow
    """
    score = 0.0
    if futurelm != {}:
        for token_id_and_token, token_count in bow.items():
            token = token_id_and_token[1]
            if token_count == 0:
                continue
            else:
                score += futurelm[token] * token_count
    return score


def reorder(model, eos_id, bow, beam_size, futurelm, ordering_max_batch_size_control, use_cuda):

    n = sum([v for action, v in bow.items()])

    current_sequence = [eos_id]


    beams = {}
    #Hypothesis = namedtuple("Hypothesis", ['score', 'last_action', "bow", "future_score", "current_sequence", "last_beam"])
    beams[0] = [Hypothesis(0, None, bow, future(bow, futurelm), current_sequence, None)]

    for i in range(1, n + 1):
        beams[i] = []

    for i in range(n):
        print(f"---------------------Batch stats:")
        global global_max_batch_size
        global global_batch_size
        global global_batch_num
        print(f"\tglobal_max_batch_size: {global_max_batch_size}")
        print(f"\t\tglobal_batch_size: {global_batch_size}")
        print(f"\t\tglobal_batch_num: {global_batch_num}")
        print(f"Average batch size: {global_batch_size/global_batch_num if global_batch_num > 0 else 0}")
        global_max_batch_size = 0
        global_batch_size = 0
        global_batch_num = 0

        if ordering_max_batch_size_control > 0:
            ordering_max_batch_size = ordering_max_batch_size_control
        elif ordering_max_batch_size_control == -1:  # batch sequence for 12gb card
            if i <= 10:
                ordering_max_batch_size = 96
            elif i <= 20:
                ordering_max_batch_size = 64
            elif i <= 25:
                ordering_max_batch_size = 50
            elif i <= 30:
                ordering_max_batch_size = 40
            elif i <= 40:
                ordering_max_batch_size = 28  # was 36, 48
            elif i <= 45:
                ordering_max_batch_size = 24
            elif i <= 50:
                ordering_max_batch_size = 20
            else:
                ordering_max_batch_size = 1

        print(f"Beam index {i}; beam size: {len(beams[i])}; batch size: {ordering_max_batch_size}")

        if i == n - 1:
            final_action = True  # on last step, also consider eos
            seq_i = 0
            packed_beam_indexes = []
            packed_input_sequences = []
            packed_input_token_ids_lens = []
            packed_output_sequences = []
            packed_token_id_and_token_actions = []
            packed_seq_i = []
            for j, hyp in enumerate(beams[i]):
                for token_id_and_token in bow:
                    action_token_id = token_id_and_token[0]
                    if hyp.bow[token_id_and_token] > 0:  # here, if 0, ignore (i.e., don't add to packed batches)
                        packed_input_sequences.append(hyp.current_sequence + [action_token_id])
                        packed_output_sequences.append(hyp.current_sequence[1:] + [action_token_id] + [eos_id])
                        packed_input_token_ids_lens.append([len(packed_input_sequences[-1])])
                        packed_token_id_and_token_actions.append(token_id_and_token)
                        packed_seq_i.append(seq_i)
                        packed_beam_indexes.append(j)
                        seq_i += 1

            packed_scores = np.zeros(len(packed_input_sequences))

            for b_i in range(0, len(packed_input_sequences), ordering_max_batch_size):
                batch_range = min(ordering_max_batch_size, len(packed_input_sequences) - b_i)

                batch_log_probs = _batch_advance(model, packed_input_sequences[b_i:b_i+batch_range], packed_input_token_ids_lens[b_i:b_i+batch_range], packed_output_sequences[b_i:b_i+batch_range], use_cuda, final_action)
                packed_scores[b_i:b_i+batch_range] = batch_log_probs.cpu().numpy()

        else:
            final_action = False
            seq_i = 0
            packed_beam_indexes = []
            packed_input_sequences = []
            packed_input_token_ids_lens = []
            packed_output_sequences_grouped = []
            packed_token_id_and_token_actions = []
            packed_seq_i = []

            for j, hyp in enumerate(beams[i]):
                packed_input_sequences.append(hyp.current_sequence)
                packed_input_token_ids_lens.append([len(packed_input_sequences[-1])])

                packed_output_sequences = []

                for token_id_and_token in bow:
                    action_token_id = token_id_and_token[0]
                    if hyp.bow[token_id_and_token] > 0:  # here, if 0, ignore (i.e., don't add to packed batches)
                        packed_output_sequences.append(hyp.current_sequence[1:] + [action_token_id])
                        packed_token_id_and_token_actions.append(token_id_and_token)

                        packed_seq_i.append(seq_i)
                        packed_beam_indexes.append(j)
                        seq_i += 1

                packed_output_sequences_grouped.append(packed_output_sequences)

            packed_scores = np.zeros(len(packed_seq_i))

            packed_scores_i = 0
            for b_i in range(0, len(packed_input_sequences), ordering_max_batch_size):
                batch_range = min(ordering_max_batch_size, len(packed_input_sequences) - b_i)

                packed_log_probs = _packed_batch_advance(model, packed_input_sequences[b_i:b_i+batch_range], packed_input_token_ids_lens[b_i:b_i+batch_range], packed_output_sequences_grouped[b_i:b_i+batch_range], use_cuda, final_action)
                packed_scores[packed_scores_i:packed_scores_i+len(packed_log_probs)] = packed_log_probs
                packed_scores_i += len(packed_log_probs)


        # Add to beam
        ni = i + 1 # always a single step
        for seq_i, j in zip(packed_seq_i, packed_beam_indexes):
            hyp = beams[i][j]
            score = packed_scores[seq_i]
            token_id_and_token = packed_token_id_and_token_actions[seq_i]
            updated_sequence = hyp.current_sequence + [token_id_and_token[0]]

            new_bow = copy.copy(hyp.bow)
            new_bow[token_id_and_token] -= 1
            fscore = future(new_bow, futurelm)
            if len(beams[ni]) < beam_size or (hyp.score + score + fscore
                                                  > beams[ni][-1].score + beams[ni][-1].future_score):
                new_hyp = Hypothesis(hyp.score + score, token_id_and_token, new_bow,
                                     fscore, updated_sequence, j)
                beams[ni].append(new_hyp)
                beams[ni].sort(key=lambda a: a.score + a.future_score)
                beams[ni].reverse()
                beams[ni] = beams[ni][:beam_size]

    order = []
    cur = n
    pos = 0
    while cur > 0:
        order.append(beams[cur][pos].last_action)  # token_id_and_token
        old_cur = cur
        cur -= 1  # single token
        pos = beams[old_cur][pos].last_beam

    order.reverse()
    return [x[1] for x in order], beams[n][0].score



def load_unigram_lm(input_unigram_file):
    futurelm = {}
    if input_unigram_file != "":
        print(f"Loading future lm from {input_unigram_file}")
        with codecs.open(input_unigram_file, encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) == 2 and tokens[0] != "ngram":
                    log_base_10 = float(tokens[0])
                    futurelm[tokens[1]] = np.log(10**log_base_10)
    if futurelm != {}:
        print(f"Using unigram future costs. Total unigram lm vocab: {len(futurelm)}")
    else:
        print(f"NOT using unigram future costs.")
    return futurelm

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

    # print(f"len(lm_input_dictionary): {len(lm_input_dictionary)}")
    # print(f"len(lm_dictionary): {len(vocab)}")
    # print(lm_input_dictionary.eos())
    # print(vocab.eos())

    print(f"Size of vocabulary dictionary: {len(vocab)}")
    assert len(lm_input_dictionary) == len(vocab)

    futurelm = load_unigram_lm(parsed_args.input_unigram_file)

    model = models[0]
    if use_cuda:
        model.cuda()

    #######################################
    eos_id = lm_input_dictionary.eos()

    assert parsed_args.output_reordered_filepath_with_name != ""
    out_file_object = codecs.open(parsed_args.output_reordered_filepath_with_name, 'w', 'utf-8')
    ordered_input_file_object = codecs.open(parsed_args.input_ordered_file, encoding="utf-8")
    sentence_num = 0

    running_times = []

    with codecs.open(parsed_args.input_shuffled_file, encoding="utf-8") as f:
        for line in f:
            if sentence_num >= parsed_args.ordering_start_sentence and sentence_num <= parsed_args.ordering_end_sentence:
                # Sentence number\t||\tNumber of Future LM (unigram) unknown tokens\t||\tNumber of LM unknown tokens\t||\tNumber of tokens\t||\tOriginal sentence log probability\t||\Reordered sentence log probability\t||\tOriginal sentence\t||\tReordered sentence
                output_line = []
                output_line.append(f"{sentence_num}")
                output_line.append(OUTPUT_LINE_SEPARATOR)

                token_ids_tensor = Tokenizer.tokenize(
                    line, lm_input_dictionary, add_if_not_exist=False,
                    append_eos=False, reverse_order=False,
                ).long()

                token_ids = [t for t in token_ids_tensor]
                tokens = tokenize_line(line)
                assert len(token_ids) == len(tokens)

                print(f"------------------Sentence: {sentence_num}")
                print(f"Length: {len(tokens)}")
                futurelm_filtered = {}  # can filter future lm to only contain words in the bag
                bow = defaultdict(int)
                in_futurelm = 0
                for token_id, token in zip(token_ids, tokens):
                    bow[(token_id, token)] += 1
                    if futurelm != {}:
                        if token in futurelm:
                            in_futurelm += 1
                            futurelm_filtered[token] = futurelm[token]
                        else:
                            futurelm_filtered[token] = futurelm["<unk>"]

                if futurelm_filtered != {}:
                    print(f"\tProportion of tokens in future lm (non-unks): {in_futurelm / len(tokens)}")
                    output_line.append(f"{len(tokens)-in_futurelm}")
                    output_line.append(OUTPUT_LINE_SEPARATOR)
                else:
                    output_line.append(f"{-1}")
                    output_line.append(OUTPUT_LINE_SEPARATOR)
                start_time = time.time()
                reordered, reordered_log_prob = reorder(model, eos_id, bow, parsed_args.ordering_beam_size, futurelm_filtered, parsed_args.ordering_max_batch_size, use_cuda)
                run_time = time.time() - start_time
                running_times.append(run_time)


                original_line = ordered_input_file_object.readline()
                original_token_ids_tensor = Tokenizer.tokenize(
                    original_line, lm_input_dictionary, add_if_not_exist=False,
                    append_eos=False, reverse_order=False,
                ).long()

                original_token_ids = [t for t in original_token_ids_tensor]
                original_tokens = tokenize_line(original_line)
                assert len(original_token_ids) == len(original_tokens)

                original_log_prob = advance(model, [eos_id] + original_token_ids, original_token_ids + [eos_id], use_cuda, 1)

                annotated_original_line = []
                for token in original_line.split():
                    if token not in lm_input_dictionary.indices:
                        annotated_original_line.append(f"{token}(UNKNOWN)")
                    else:
                        annotated_original_line.append(token)
                annotated_original_line = " ".join(annotated_original_line)
                num_unks_in_reordered = 0
                reordered_line = " ".join(reordered)
                annotated_reordered_line = []
                for token in reordered:
                    if token not in lm_input_dictionary.indices:
                        annotated_reordered_line.append(f"{token}(UNKNOWN)")
                        num_unks_in_reordered += 1
                    else:
                        annotated_reordered_line.append(token)
                annotated_reordered_line = " ".join(annotated_reordered_line)

                original_line = original_line.strip()
                if original_line != annotated_original_line:
                    print(f"Original (with unknowns): {annotated_original_line}")
                else:
                    print(f"Original: {original_line}")
                print("")
                if annotated_reordered_line != reordered_line:
                    print(f"Reordered (with unknowns): {annotated_reordered_line}")
                else:
                    print(f"Reordered: {reordered_line}")
                print(f"Original log prob: {original_log_prob}; Reordered log prob: {reordered_log_prob}")
                print(f"\t Ordering time: {run_time} seconds; Running total: {np.sum(running_times)} seconds; Running average: {np.sum(running_times) / len(running_times)} seconds / sentence")


                # Sentence number\t||\tNumber of Future LM (unigram) unknown tokens\t||\tNumber of LM unknown tokens\t||\tNumber of tokens\t||\tOriginal sentence log probability\t||\Reordered sentence log probability\t||\tOriginal sentence\t||\tReordered sentence


                output_line.append(f"{num_unks_in_reordered}")
                output_line.append(OUTPUT_LINE_SEPARATOR)
                output_line.append(f"{len(reordered)}")
                output_line.append(OUTPUT_LINE_SEPARATOR)

                output_line.append(f"{original_log_prob}")
                output_line.append(OUTPUT_LINE_SEPARATOR)
                output_line.append(f"{reordered_log_prob}")
                output_line.append(OUTPUT_LINE_SEPARATOR)

                output_line.append(f"{original_line}")
                output_line.append(OUTPUT_LINE_SEPARATOR)
                output_line.append(f"{' '.join(reordered)}")

                out_file_object.write(" ".join(output_line))
                out_file_object.write('\n')
                out_file_object.flush()

            else:
                ordered_input_file_object.readline()  # advance file to maintain alignment with the shuffled file

            sentence_num += 1


if __name__ == '__main__':
    parser = options.get_eval_lm_ordering_parser()
    args = options.parse_args_and_arch(parser)
    main(args)
