################################################################################
#######  Create the unigram LM from GBW
################################################################################

# download the pre-processed version of the Google Billion Words benchmark (GBW)
# training dataset (Chelba et al., 2013), which is currently available via:

wget http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz

# unpack, then cat the training files, as for example (replacing FILL_IN with applicable paths):

DATA_DIR=FILL_IN/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled
cat ${DATA_DIR}/news.en* > ${DATA_DIR}/combined/google_1b_combined.txt

MODEL_PATH=FILL_IN/LM1_noeos_google_1b_data.arpa
TEMP_DIR=FILL_IN/temp_delete_any_time/

# create the unigram LM with, for example, kenlm (https://github.com/kpu/kenlm):

lmplz -o 1 -S 3145728K -T ${TEMP_DIR} --discount_fallback <${DATA_DIR}/combined/google_1b_combined.txt >${MODEL_PATH}
