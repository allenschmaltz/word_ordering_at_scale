# The following simply creates a template for an end-user to assess the
# ordering output. Note that the --input_file expects that the FIRST LINE IS
# A HEADER, which is ignored.

##############################################################################
### Generate human evaluation template -- PTB
##############################################################################

REPO_DIR=FILL_IN

python -u ${REPO_DIR}/code/data/combined_output_to_human_eval_template.py \
--input_file ${REPO_DIR}/output/ptb/valid_words_ref.txt.sample100.google_lm_1b_normalized.shuffled.txt.reordered_beam20000.packed_adapt_schedule.future.sent0_to_sent99.with_header.txt \
--output_file ${REPO_DIR}/output_human_eval/ptb/valid_words_ref.txt.sample100.google_lm_1b_normalized.shuffled.txt.reordered_beam20000.packed_adapt_schedule.future.sent0_to_sent99.with_header.txt.human_eval_template.txt


##############################################################################
### Generate human evaluation template -- GBW
##############################################################################

REPO_DIR=FILL_IN

python -u ${REPO_DIR}/code/data/combined_output_to_human_eval_template.py \
--input_file ${REPO_DIR}/output/google_1b/news.en.heldout-00000-of-00050.sample100.shuffled.txt.reordered_beam20000.packed_adapt_schedule.future.sent0_to_sent99.with_header.txt \
--output_file ${REPO_DIR}/output_human_eval/google_1b/news.en.heldout-00000-of-00050.sample100.shuffled.txt.reordered_beam20000.packed_adapt_schedule.future.sent0_to_sent99.with_header.txt.human_eval_template.txt
