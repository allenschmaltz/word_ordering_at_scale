# In the following, change ${REPO_DIR} to point to this repo, ${MODEL_PATH} to
# point to the pre-trained model, and ${GPU_ID} to point to the applicable GPU.

# ${OUTPUT_DIR} should point to a directory to save the re-ordered output,
# and ${UNIGRAM_LM_MODEL_PATH}, if provided, should point to a unigram LM (which
# can be created using kenlm, for example).

GPU_ID=FILL_IN
REPO_DIR=FILL_IN
MODEL_PATH=FILL_IN
OUTPUT_DIR=FILL_IN
UNIGRAM_LM_MODEL_PATH=FILL_IN/LM1_noeos_google_1b_data.arpa

##
# Note that since there is no shared state across sentences, decoding is
# embarrassingly parallel. Use the --ordering_start_sentence and
# --ordering_end_sentence options to re-order particular ranges of sentences.
#
# --ordering_max_batch_size -1 sets the maximum batch-size to preset values
# that should be suitable (i.e., not lead to memory issues) for ordering all of
# the sentences examined here on a 12gb card. Alternatively, a value greater
# than 0 can override the preset values (for a single, constant size across
# sentence/prefix lengths).
##


################################################################################
#######  re-order -- PTB
################################################################################

cd ${REPO_DIR}/code/fairseq

DATADIR=${REPO_DIR}/support_files/model_vocab

ORDERED_INPUT_FILE=${REPO_DIR}/data/ptb/valid_words_ref.txt.sample100.google_lm_1b_normalized.ordered.txt
SHUFFLED_INPUT_FILE=${REPO_DIR}/data/ptb/valid_words_ref.txt.sample100.google_lm_1b_normalized.shuffled.txt

BEAM_SIZE=20000

START_SENT_INDEX=0
END_SENT_INDEX=99

OUTPUT_FILE=${OUTPUT_DIR}/valid_words_ref.txt.sample100.google_lm_1b_normalized.shuffled.txt.reordered_beam${BEAM_SIZE}.packed_adapt_schedule.future.sent${START_SENT_INDEX}_to_sent${END_SENT_INDEX}.txt

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u order.py ${DATADIR}/ --path ${MODEL_PATH}/model.pt \
--input_ordered_file ${ORDERED_INPUT_FILE} \
--input_shuffled_file ${SHUFFLED_INPUT_FILE} \
--output_reordered_filepath_with_name ${OUTPUT_FILE} \
--ordering_beam_size ${BEAM_SIZE} \
--ordering_max_batch_size -1 \
--ordering_start_sentence ${START_SENT_INDEX} \
--ordering_end_sentence ${END_SENT_INDEX} \
--input_unigram_file ${UNIGRAM_LM_MODEL_PATH}


################################################################################
#######  re-order -- GBW
################################################################################

cd ${REPO_DIR}/code/fairseq

DATADIR=${REPO_DIR}/support_files/model_vocab

ORDERED_INPUT_FILE=${REPO_DIR}/data/google_1b/news.en.heldout-00000-of-00050.sample100.ordered.txt
SHUFFLED_INPUT_FILE=${REPO_DIR}/data/google_1b/news.en.heldout-00000-of-00050.sample100.shuffled.txt

BEAM_SIZE=20000

START_SENT_INDEX=0
END_SENT_INDEX=99

OUTPUT_FILE=${OUTPUT_DIR}/news.en.heldout-00000-of-00050.sample100.shuffled.txt.reordered_beam${BEAM_SIZE}.packed_adapt_schedule.future.sent${START_SENT_INDEX}_to_sent${END_SENT_INDEX}.txt

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u order.py ${DATADIR}/ --path ${MODEL_PATH}/model.pt \
--input_ordered_file ${ORDERED_INPUT_FILE} \
--input_shuffled_file ${SHUFFLED_INPUT_FILE} \
--output_reordered_filepath_with_name ${OUTPUT_FILE} \
--ordering_beam_size ${BEAM_SIZE} \
--ordering_max_batch_size -1 \
--ordering_start_sentence ${START_SENT_INDEX} \
--ordering_end_sentence ${END_SENT_INDEX} \
--input_unigram_file ${UNIGRAM_LM_MODEL_PATH}
