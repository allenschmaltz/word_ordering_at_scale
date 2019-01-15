# Calculate BLEU on all sentences for reference.

# Replace FILL_IN with applicable paths. (NOTE that this
# will overwrite the existing data in the repo.)

# First, separate the output sentences from other summary information, and
# then calculate BLEU. (Here we use the BLEU script used in previous work
# on word ordering.)

##############################################################################
### save reordered sentences alone into a single file -- PTB
##############################################################################

REPO_DIR=FILL_IN

python -u ${REPO_DIR}/code/data/combined_output_to_sentences.py \
--input_file ${REPO_DIR}/output/ptb/valid_words_ref.txt.sample100.google_lm_1b_normalized.shuffled.txt.reordered_beam20000.packed_adapt_schedule.future.sent0_to_sent99.with_header.txt \
--expected_number_of_sentences 100 \
--output_file ${REPO_DIR}/output/ptb/valid_words_ref.txt.sample100.google_lm_1b_normalized.shuffled.txt.reordered_beam20000.packed_adapt_schedule.future.sent0_to_sent99.with_header.txt.only_reordered_sentences.txt

##############################################################################
### save reordered sentences alone into a single file -- GBW
##############################################################################

REPO_DIR=FILL_IN

python -u ${REPO_DIR}/code/data/combined_output_to_sentences.py \
--input_file ${REPO_DIR}/output/google_1b/news.en.heldout-00000-of-00050.sample100.shuffled.txt.reordered_beam20000.packed_adapt_schedule.future.sent0_to_sent99.with_header.txt \
--expected_number_of_sentences 100 \
--output_file ${REPO_DIR}/output/google_1b/news.en.heldout-00000-of-00050.sample100.shuffled.txt.reordered_beam20000.packed_adapt_schedule.future.sent0_to_sent99.with_header.txt.only_reordered_sentences.txt


##############################################################################
### Evalutate BLEU -- PTB
##############################################################################

WORD_ORDER_2016_REPO=FILL_IN # path to https://github.com/allenschmaltz/word_ordering
REPO_DIR=FILL_IN # path to this repo
REF_DATA_FILE=${REPO_DIR}/data/ptb/valid_words_ref.txt.sample100.google_lm_1b_normalized.ordered.txt
TEMP_DIR=FILL_IN/temp_delete_any_time

OUTPUT_FILE=${REPO_DIR}/output/ptb/valid_words_ref.txt.sample100.google_lm_1b_normalized.shuffled.txt.reordered_beam20000.packed_adapt_schedule.future.sent0_to_sent99.with_header.txt.only_reordered_sentences.txt

${WORD_ORDER_2016_REPO}/analysis/eval/zgen_bleu/ScoreBLEU.sh -t ${OUTPUT_FILE} -r ${REF_DATA_FILE} -odir ${TEMP_DIR}

# BLEU score = 0.5511 (0.5511 * 1.0000) for system "1"

##############################################################################
### Evalutate BLEU -- GBW
##############################################################################

WORD_ORDER_2016_REPO=FILL_IN # path to https://github.com/allenschmaltz/word_ordering
REPO_DIR=FILL_IN # path to this repo
REF_DATA_FILE=${REPO_DIR}/data/google_1b/news.en.heldout-00000-of-00050.sample100.ordered.txt
TEMP_DIR=FILL_IN/temp_delete_any_time

OUTPUT_FILE=${REPO_DIR}/output/google_1b/news.en.heldout-00000-of-00050.sample100.shuffled.txt.reordered_beam20000.packed_adapt_schedule.future.sent0_to_sent99.with_header.txt.only_reordered_sentences.txt

${WORD_ORDER_2016_REPO}/analysis/eval/zgen_bleu/ScoreBLEU.sh -t ${OUTPUT_FILE} -r ${REF_DATA_FILE} -odir ${TEMP_DIR}

#BLEU score = 0.6180 (0.6180 * 1.0000) for system "1"
