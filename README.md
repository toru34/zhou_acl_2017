## Selective Encoding for Abstractive Sentence Summarization
DyNet implementation of the paper Selective Encoding for Abstractive Sentence Summarization (ACL 2017).

### Requirement
- Python 3.6.0+
- DyNet 2.0+
- NumPy 1.12.1+
- scikit-learn 0.19.0+

### Arguments for training
- `--n_epochs`: Number of epochs for training [default: 3]
- `--batch_size`: Batch size for training [default: 16]
- `--emb_dim`: Embedding size for each word [default: 32]
- `--hid_dim`: Hidden state for both encoder and decoder [default: 32]
- `--vocab_size`: Vocabulary size [default: 10000]
- `--maxout_dim`: Maxout size [default: 5]

### Arguments for testing
Work in progress.

### How to train (example)
```
python train.py --n_epochs 10 --batch_size 32 --emb_dim 64 --hid_dim 64 --vocab_size 30000 --maxout_size 10
```

### How to test (example)
Work in progress

References
- Q. Zhou. 2017. Selective Encoding for Abstractive Sentence Summarization. In Proceedings of ACL 2017
