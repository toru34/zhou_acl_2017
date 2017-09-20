## Selective Encoding for Abstractive Sentence Summarization
DyNet implementation of the paper Selective Encoding for Abstractive Sentence Summarization (ACL 2017)[1].

### 1. Requirement
- Python 3.6.0+
- DyNet 2.0+
- NumPy 1.12.1+
- scikit-learn 0.19.0+
- tqdm 4.15.0+

### 2. Prepare dataset
To get gigaward corpus[2], run
```
sh download_giga.sh
```
.

### 3. Training

#### Arguments
- `--gpu`: GPU id to use. For cpu, set -1 [default: -1]
- `--n_train`: Number of training examples (up to 3803957 in gigaword) [default: 100000]
- `--n_valid`: Number of validation examples (up to 189651 in gigaword) [default: 100]
- `--n_epochs`: Number of epochs for training [default: 20]
- `--batch_size`: Batch size for training [default: 32]
- `--emb_dim`: Embedding size for each word [default: 256]
- `--hid_dim`: Hidden state for both encoder and decoder [default: 256]
- `--vocab_size`: Vocabulary size [default: 60000]
- `--maxout_dim`: Maxout size [default: 5]
- `--alloc_mem`: Amount of memory to allocate[mb] [default: 8192]

#### Command example
```
python train.py --n_epochs 20
```

### 4. Test
#### Arguments
- `--gpu`: GPU id to use. For cpu, set -1 [default: -1]
- `--n_test`: Number of test examples [default: 100]
- `--beam_size`: Beam size for decoding [default: 5]
- `--max_len`: Maximum length of decoding [default: 50]
- `--model_file`: Model to use for generation [default: ./model]
- `--input_file`: Input file path [default: ./data/valid.article.filter.txt]
- `--output_file`: Output file path [default: ./data/pred.txt]
- `--w2i_file`: Word2Index file path [default: ./w2i.dump]
- `--i2w_file`: Index2Word file path [default: ./i2w.dump]
- `--alloc_mem`: Amount of memory to allocate[mb] [default: 1024]

#### Command example
```
python test.py --beam_size 10
```

### 5. Results
Work in progress.

### Notes

### Reference
- [1] Q. Zhou. et al. 2017. Selective Encoding for Abstractive Sentence Summarization. In Proceedings of ACL 2017 \[[pdf\]](http://aclweb.org/anthology/P/P17/P17-1101.pdf)
- [2] A. M. Rush et al. 2015. A Neural Attention Model for Abstractive Sentence Summarization. In Proceedings of EMNLP 2015 \[[pdf\]](http://aclweb.org/anthology/D/D15/D15-1044.pdf)
