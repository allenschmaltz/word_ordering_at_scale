
# See notes/setup/unigram_lm.sh for instructions on downloading the training data
# (replace TRAINING_DATA as applicable).

# The generated files are used in the analysis scripts.

# Replace FILL_IN with applicable paths. (NOTE that this
# will overwrite the existing data in the repo.)

##############################################################################
### generate ngram summary stats from the GBW training data -- PTB
##############################################################################

REPO_DIR=FILL_IN

TRAINING_DATA=FILL_IN/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/combined/google_1b_combined.txt
DICT_FILE=${REPO_DIR}/support_files/model_vocab/dict.txt
ORDERED_INPUT_FILE=${REPO_DIR}/data/ptb/valid_words_ref.txt.sample100.google_lm_1b_normalized.ordered.txt
OUTPUT_FILE=${REPO_DIR}/data/ptb/summary_stats/valid_words_ref.txt.sample100.google_lm_1b_normalized.ordered.txt.covered_train_ngrams.txt

python -u ${REPO_DIR}/code/data/generate_ngram_overlap_between_source_and_training.py \
--input_training_sentences_file ${TRAINING_DATA} \
--input_eval_sentences_file ${ORDERED_INPUT_FILE} \
--vocab_file ${DICT_FILE} \
--expected_number_of_sentences 100 \
--output_file ${OUTPUT_FILE}


##############################################################################
### generate ngram summary stats from the GBW training data -- GBW
##############################################################################

REPO_DIR=FILL_IN

TRAINING_DATA=FILL_IN/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/combined/google_1b_combined.txt
DICT_FILE=${REPO_DIR}/support_files/model_vocab/dict.txt
ORDERED_INPUT_FILE=${REPO_DIR}/data/google_1b/news.en.heldout-00000-of-00050.sample100.ordered.txt
OUTPUT_FILE=${REPO_DIR}/data/google_1b/summary_stats/news.en.heldout-00000-of-00050.sample100.ordered.txt.covered_train_ngrams.txt

python -u ${REPO_DIR}/code/data/generate_ngram_overlap_between_source_and_training.py \
--input_training_sentences_file ${TRAINING_DATA} \
--input_eval_sentences_file ${ORDERED_INPUT_FILE} \
--vocab_file ${DICT_FILE} \
--expected_number_of_sentences 100 \
--output_file ${OUTPUT_FILE}
