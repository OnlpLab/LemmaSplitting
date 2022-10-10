# Lemma Split Data for Morphological Inflection

Repo for the paper [(Un)solving Morphological Inflection: Lemma Overlap Artificially Inflates Models' Performance](https://aclanthology.org/2022.acl-short.96.pdf).

## Contents

+ `LemmaSplitData` contains a lemma split version of the data from [SIGMORPHON's 2020 shared task 0](https://github.com/sigmorphon2020/task0-data).
+ `lstm` contains the baseline LSTM model.
+ `generate_lemma_splits.py` is the script used to produce the data in practice.

## Generating Lemma-Splits Data

1. Clone the SIGMORPHON 2020 task 0 data - the 3 folders `DEVELOPMENT_LANGUAGES`, `SURPRISE_LANGUAGES` and `GOLD-TEST` - to the folder `DataExperiments/FormSplit`.
2. Run the script `generate_lemma_splits.py`. It will generate a folder called `DataExperiments/LemmaSplit` at the same level and the same families sub-division (without the covered test files), where the samples are split across lemmas instead of randomly.
