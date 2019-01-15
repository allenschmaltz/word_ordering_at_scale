
# In the following, change ${REPO_DIR} to point to this repo, ${MODEL_PATH} to
# point to the pre-trained model, and ${GPU_ID} to point to the applicable GPU

GPU_ID=FILL_IN
REPO_DIR=FILL_IN
MODEL_PATH=FILL_IN


################################################################################
#######  print perplexity -- PTB
################################################################################

cd ${REPO_DIR}/code/fairseq

DATADIR=${REPO_DIR}/support_files/model_vocab

ORDERED_INPUT_FILE=${REPO_DIR}/data/ptb/valid_words_ref.txt.sample100.google_lm_1b_normalized.ordered.txt
SHUFFLED_INPUT_FILE=${REPO_DIR}/data/ptb/valid_words_ref.txt.sample100.google_lm_1b_normalized.shuffled.txt

START_SENT_INDEX=0
END_SENT_INDEX=99

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u print_perplexity.py ${DATADIR}/ --path ${MODEL_PATH}/model.pt \
--input_ordered_file ${ORDERED_INPUT_FILE} \
--input_shuffled_file ${SHUFFLED_INPUT_FILE} \
--ordering_start_sentence ${START_SENT_INDEX} \
--ordering_end_sentence ${END_SENT_INDEX}

    # num. model params: 796771584
    # Size of vocabulary dictionary: 793302
    # Shuffled perplexity: 18604.335381436093
    # Original perplexity: 63.45925627515932
    # Total tokens (including eos): 2416

################################################################################
#######  print perplexity -- GBW
################################################################################

cd ${REPO_DIR}/code/fairseq

DATADIR=${REPO_DIR}/support_files/model_vocab

ORDERED_INPUT_FILE=${REPO_DIR}/data/google_1b/news.en.heldout-00000-of-00050.sample100.ordered.txt
SHUFFLED_INPUT_FILE=${REPO_DIR}/data/google_1b/news.en.heldout-00000-of-00050.sample100.shuffled.txt

START_SENT_INDEX=0
END_SENT_INDEX=99

CUDA_VISIBLE_DEVICES=${GPU_ID} python -u print_perplexity.py ${DATADIR}/ --path ${MODEL_PATH}/model.pt \
--input_ordered_file ${ORDERED_INPUT_FILE} \
--input_shuffled_file ${SHUFFLED_INPUT_FILE} \
--ordering_start_sentence ${START_SENT_INDEX} \
--ordering_end_sentence ${END_SENT_INDEX}

    # num. model params: 796771584
    # Size of vocabulary dictionary: 793302
    # Shuffled perplexity: 15238.758074692141
    # Original perplexity: 31.035624245338145
    # Total tokens (including eos): 2641
