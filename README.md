# Conditional-N-gram-Regularization

This repository contains the code used for one Research paper:

[Using Large Corpus N-gram Statistics to Improve Recurrent Neural Language Models](https://www.aclweb.org/anthology/papers/N/N19/N19-1330/)

This code was originally forked from the [PyTorch word level language modeling example](https://github.com/pytorch/examples/tree/master/word_language_model) and the variational dropout implementation is from [LSTM and QRNN Language Model Toolkit](https://github.com/salesforce/awd-lstm-lm).

Usage:
python main_fast.py --cuda  --tied --batch_size=64     --epochs=30
--ngram_bsz=500  --data=./data/1B2.5M --data_name=1B
--ngram_dir=bigram_ppl/  --gamma=0.75 --loss_type=comb

You will have build two python list objects under ngram_dir: 1. list of
n-1_grams ( you should map token to integer) 2.  list of sparse matrix (
(row, column, data) format) of corresponding conditional distribution of
n_th token conditioned on n-1_grams. Both should be a list of batches.

