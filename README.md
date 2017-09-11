## Selective Encoding for Abstractive Sentence Summarization
DyNet implementation of the paper Selective Encoding for Abstractive Sentence Summarization (ACL 2017).

### Requirement
- Python 3.6.0+
- DyNet 2.0+
- NumPy 1.12.1+
- scikit-learn 0.19.0+
- tqdm 4.15.0+

### Prepare dataset
To get gigaward corpus, run
```
sh download.sh
```
, or you can use your own corpus.

### Arguments for training
- `--n_epochs`: Number of epochs for training [default: 3]
- `--batch_size`: Batch size for training [default: 16]
- `--emb_dim`: Embedding size for each word [default: 32]
- `--hid_dim`: Hidden state for both encoder and decoder [default: 32]
- `--vocab_size`: Vocabulary size [default: 10000]
- `--maxout_dim`: Maxout size [default: 5]
- `--alloc_mem`: Amount of memory to allocate[mb] [default: 1024]

### Arguments for test
- `--beam_size`: Beam size for decoding [default: 5]
- `--max_len`: Maximum length of decoding [default: 50]
- `--model_file`: Model to use for generation [default: ./model]
- `--input_file`: Input file path [default: ./data/input.txt]
- `--output_file`: Output file path [default: ./data/pred.txt]
- `--w2i_file`: Word2Index file path [default: ./w2i.dump]
- `--i2w_file`: Index2Word file path [default: ./i2w.dump]

### How to train (example)
```
python train.py --n_epochs 10 --batch_size 32 --emb_dim 64 --hid_dim 64 --vocab_size 30000 --maxout_size 10
```

### How to test (example)
```
python test.py --beam_size 10
```

### Reference
- Q. Zhou. et al. 2017. Selective Encoding for Abstractive Sentence Summarization. In Proceedings of ACL 2017 \[[pdf\]](http://aclweb.org/anthology/P/P17/P17-1101.pdf)
