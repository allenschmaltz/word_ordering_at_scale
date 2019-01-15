# For reference, these are the scripts used for randomly sampling the test datasets.
# In the case of PTB, the tokenization is converted to match that of the GBW
# dataset. The pre-processed data appears in the repo in the data/ folder,
# but these may be of interest in the future for increasing the sample size.

################################################################################
#######  Sample and normalize tokenization to GBW -- PTB
################################################################################

# First, create the standard PTB word ordering dataset with the instructions at
# https://github.com/allenschmaltz/word_ordering

# Run the following, replacing FILL_IN with applicable paths. (NOTE that this
# will overwrite the existing data in the repo.)

# The file ending '.shuffled.txt' contains the sentences with fully shuffled
# tokens (i.e., the input for ordering models).

REPO_DIR=FILL_IN
INPUT_FILE=FILL_IN/datasets/zgen_data_gold/valid_words_ref.txt

SAMPLE_SIZE=100

OUTPUT_ORDERED_FILE=${REPO_DIR}/data/ptb/valid_words_ref.txt.sample${SAMPLE_SIZE}.google_lm_1b_normalized.ordered.txt
OUTPUT_SHUFFLED_FILE=${REPO_DIR}/data/ptb/valid_words_ref.txt.sample${SAMPLE_SIZE}.google_lm_1b_normalized.shuffled.txt
OUTPUT_INDEX_FILE=${REPO_DIR}/data/ptb/valid_words_ref.txt.sample${SAMPLE_SIZE}.google_lm_1b_normalized.index_file.txt

python ${REPO_DIR}/code/data/normalize_and_sample_lines.py \
--input_file ${INPUT_FILE} \
--sample_size ${SAMPLE_SIZE} \
--output_file ${OUTPUT_ORDERED_FILE} \
--output_shuffled_file ${OUTPUT_SHUFFLED_FILE} \
--output_index_file ${OUTPUT_INDEX_FILE}



################################################################################
#######  Sample -- GBW
################################################################################

# The standard test split of the dataset is available at the following URL:
# http://download.tensorflow.org/models/LM_LSTM_CNN/test/news.en.heldout-00000-of-00050
# which is linked from the following repo:
# https://github.com/tensorflow/models/tree/master/research/lm_1b

# Run the following, replacing FILL_IN with applicable paths. (NOTE that this
# will overwrite the existing data in the repo.)

# The file ending '.shuffled.txt' contains the sentences with fully shuffled
# tokens (i.e., the input for ordering models).


REPO_DIR=FILL_IN
INPUT_FILE=FILL_IN/news.en.heldout-00000-of-00050

SAMPLE_SIZE=100

OUTPUT_ORDERED_FILE=${REPO_DIR}/data/google_1b/news.en.heldout-00000-of-00050.sample${SAMPLE_SIZE}.ordered.txt
OUTPUT_SHUFFLED_FILE=${REPO_DIR}/data/google_1b/news.en.heldout-00000-of-00050.sample${SAMPLE_SIZE}.shuffled.txt
OUTPUT_INDEX_FILE=${REPO_DIR}/data/google_1b/news.en.heldout-00000-of-00050.sample${SAMPLE_SIZE}.index_file.txt

python ${REPO_DIR}/code/data/sample_lines.py \
--input_file ${INPUT_FILE} \
--sample_size ${SAMPLE_SIZE} \
--output_file ${OUTPUT_ORDERED_FILE} \
--output_shuffled_file ${OUTPUT_SHUFFLED_FILE} \
--output_index_file ${OUTPUT_INDEX_FILE}
