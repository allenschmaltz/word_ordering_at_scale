# word_ordering_at_scale

This is the associated repo for the second chapter, "Word Ordering at Scale", of my dissertation, "Learning to Order and Learning to Correct".

# Summary

In summary, in this chapter, we perform a small human evaluation of the output of the surface-level ordering approach of Schmaltz et al. 2016 at a larger scale than in the original work (in terms of training data, model size, and beam size, as well as with an alternative model architecture) in order to analyze the remaining ordering errors made by a particular surface-level model, with implications for future work. Previous works (e.g., Schmaltz et al. 2016 and Hasler et al. 2017) have suggested that strong surface-level language models are at least as effective as models trained with explicit syntactic structures for modeling ordering constraints. However, such surface-level models still make ordering errors under the experimental conditions of standard evaluation setups. At the larger scale of our experiments here, with the particular surface-level language model of Dauphin et al. 2017, we find that while the model is reasonably effective at ordering when accounting for search errors, grammatical (and semantic) errors are still present in some of the re-ordered sentences. We find that remaining errors are associated with greater proportions of n-grams unseen in training, highlighting both a path for future improvements in effectiveness and the clear brittleness of such models, with implications for generation models, more generally.

# Limitations

Such models are clearly brittle in the face of data that diverges from that seen in training, and of course, they have no general world knowledge beyond that encoded in the sequential training data. Thus, it is perhaps not surprising that there are observed instances of re-ordered sentences that are roughly grammatically acceptable but clearly not semantically acceptable, at least without qualifying context that would be unlikely in the given news domains. Adversarial examples could likely be readily constructed that would be challenging for such models on a semantic level.

Additionally, we assume that the remaining search errors could be resolved in practice with additional computing resources (as, for example, re-ordering with a beam size greater than 20,000), but we leave to future work to verify this. The sample size here is on the small side, restricted to the news domain (generally construed), so it would also be of interest to analyze additional sentences, including those from alternative domains. Finally, while the hand-annotated syntactic rules do not appear to be particularly necessary as ordering constraints when large-scale surface-level information is available, other types of hand-written rules and feature engineering may well be useful for NLP applications in practice, or when the amount of surface-level information is very limited.

# Contents

The pre-trained model used in the experiments here is available at [https://github.com/pytorch/fairseq](https://github.com/pytorch/fairseq).

The pre-processed data is available [here](data/).

The re-ordered output is available [here](output/).

The human evaluations are available [here](output_human_eval/).

Examples of running the model for ordering are available [here](notes/order/order.sh).
