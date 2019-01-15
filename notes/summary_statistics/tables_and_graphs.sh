##############################################################################
### The following generates the values and graphs in the main tables
### and figures of Chapter 2. Replace FILL_IN with applicable paths.
##############################################################################

REPO_DIR=FILL_IN
PTB_OUTPUT_GRAPH_FILE=FILL_IN/ptb_graph.eps  # can also change to .pdf, etc.
GBW_OUTPUT_GRAPH_FILE=FILL_IN/gbw_graph.eps

PTB_HUMAN_EVAL_TEMPLATE=${REPO_DIR}/output_human_eval/ptb/valid_words_ref.txt.sample100.google_lm_1b_normalized.shuffled.txt.reordered_beam20000.packed_adapt_schedule.future.sent0_to_sent99.with_header.txt.human_eval_template.txt_as_of_2018_12_31.txt
PTB_NGRAM_SUMMARY_FILE=${REPO_DIR}/data/ptb/summary_stats/valid_words_ref.txt.sample100.google_lm_1b_normalized.ordered.txt.covered_train_ngrams.txt
GBW_HUMAN_EVAL_TEMPLATE=${REPO_DIR}/output_human_eval/google_1b/news.en.heldout-00000-of-00050.sample100.shuffled.txt.reordered_beam20000.packed_adapt_schedule.future.sent0_to_sent99.with_header.txt.human_eval_template.txt_as_of_2018_12_31.txt
GBW_NGRAM_SUMMARY_FILE=${REPO_DIR}/data/google_1b/summary_stats/news.en.heldout-00000-of-00050.sample100.ordered.txt.covered_train_ngrams.txt

#### Tables 2.1, 2.2:

python -u ${REPO_DIR}/code/data/human_eval_template_analysis.py \
--input_file ${PTB_HUMAN_EVAL_TEMPLATE} \
--ngram_summary_stats_file ${PTB_NGRAM_SUMMARY_FILE}

python -u ${REPO_DIR}/code/data/human_eval_template_analysis.py \
--input_file ${GBW_HUMAN_EVAL_TEMPLATE} \
--ngram_summary_stats_file ${GBW_NGRAM_SUMMARY_FILE}

#### Tables 2.3, 2.6, 2.7:
# restricted to sentences without source-relative search errors:

python -u ${REPO_DIR}/code/data/human_eval_template_analysis.py \
--input_file ${PTB_HUMAN_EVAL_TEMPLATE} \
--ngram_summary_stats_file ${PTB_NGRAM_SUMMARY_FILE} \
--restrict_to_non_search_error_sentences

python -u ${REPO_DIR}/code/data/human_eval_template_analysis.py \
--input_file ${GBW_HUMAN_EVAL_TEMPLATE} \
--ngram_summary_stats_file ${GBW_NGRAM_SUMMARY_FILE} \
--restrict_to_non_search_error_sentences

#### Table 2.9

python -u ${REPO_DIR}/code/data/human_eval_template_analysis.py \
--input_file ${PTB_HUMAN_EVAL_TEMPLATE} \
--ngram_summary_stats_file ${PTB_NGRAM_SUMMARY_FILE} \
--restrict_to_non_search_error_sentences \
--length_restriction 21.143

python -u ${REPO_DIR}/code/data/human_eval_template_analysis.py \
--input_file ${GBW_HUMAN_EVAL_TEMPLATE} \
--ngram_summary_stats_file ${GBW_NGRAM_SUMMARY_FILE} \
--restrict_to_non_search_error_sentences \
--length_restriction 20.806

#### Table 2.8

python -u ${REPO_DIR}/code/data/human_eval_template_analysis.py \
--input_file ${PTB_HUMAN_EVAL_TEMPLATE} \
--ngram_summary_stats_file ${PTB_NGRAM_SUMMARY_FILE} \
--restrict_to_non_search_error_sentences \
--length_restriction 21.143 \
--length_restriction_lte

python -u ${REPO_DIR}/code/data/human_eval_template_analysis.py \
--input_file ${GBW_HUMAN_EVAL_TEMPLATE} \
--ngram_summary_stats_file ${GBW_NGRAM_SUMMARY_FILE} \
--restrict_to_non_search_error_sentences \
--length_restriction 20.806 \
--length_restriction_lte

#### Figure 2.1

python -u ${REPO_DIR}/code/data/human_eval_template_analysis.py \
--input_file ${PTB_HUMAN_EVAL_TEMPLATE} \
--ngram_summary_stats_file ${PTB_NGRAM_SUMMARY_FILE} \
--restrict_to_non_search_error_sentences \
--generate_graphs \
--graph_output_file ${PTB_OUTPUT_GRAPH_FILE}

#### Figure 2.2

python -u ${REPO_DIR}/code/data/human_eval_template_analysis.py \
--input_file ${GBW_HUMAN_EVAL_TEMPLATE} \
--ngram_summary_stats_file ${GBW_NGRAM_SUMMARY_FILE} \
--restrict_to_non_search_error_sentences \
--generate_graphs \
--graph_output_file ${GBW_OUTPUT_GRAPH_FILE}
