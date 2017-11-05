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
- `--gpu`: GPU ID to use. For cpu, set `-1` [default: `0`]
- `--n_epochs`: Number of epochs [default: `3`]
- `--n_train`: Number of training data (up to `3803957`) [default: `3803957`]
- `--n_valid`: Number of validation data (up to `189651`) [default: `189651`]
- `--vocab_size`: Vocabulary size [default: `124404`]
- `--batch_size`: Mini batch size [default: `32`]
- `--emb_dim`: Embedding size [default: `256`]
- `--hid_dim`: Hidden state size [default: `256`]
- `--maxout_dim`: Maxout size [default: `2`]
- `--alloc_mem`: Amount of memory to allocate [mb] [default: `10000`]

#### Command example
```
python train.py --n_epochs 20
```

### 4. Test
#### Arguments
- `--gpu`: GPU ID to use. For cpu, set `-1` [default: `0`]
- `--n_test`: Number of test data [default: `189651`]
- `--beam_size`: Beam size [default: `5`]
- `--max_len`: Maximum length of decoding [default: `100`]
- `--model_file`: Trained model file path [default: `./model_e1`]
- `--input_file`: Test file path [default: `./data/valid.article.filter.txt`]
- `--output_file`: Output file path [default: `./pred_y.txt`]
- `--w2i_file`: Word2Index file path [default: `./w2i.dump`]
- `--i2w_file`: Index2Word file path [default: `./i2w.dump`]
- `--alloc_mem`: Amount of memory to allocate [mb] [default: `1024`]

#### Command example
```
python test.py --beam_size 10
```

### 5. Evaluate
You can use pythonrouge[3] to compute the ROUGE scores.

### 6. Results
#### 6.1. Gigaword (validation data)
|                 |ROUGE-1 (F1)|ROUGE-2 (F1)|ROUGE-L (F1)|
|-----------------|:-----:|:-----:|:-----:|
|My implementation| 44.33| 19.57| 41.3|

#### 6.2. DUC2004
Work in progress.

#### 6.3. MSR
Work in progress.

### 7. Pretrained model
To get the pretrained model, run
```
sh download_pretrained_model.sh
```
.

### Notes

### Reference
- [1] Q. Zhou. et al. 2017. Selective Encoding for Abstractive Sentence Summarization. In Proceedings of ACL 2017 \[[pdf\]](http://aclweb.org/anthology/P/P17/P17-1101.pdf)
- [2] Gigaword/DUC2004 Corpus: https://github.com/harvardnlp/sent-summary
- [3] pythonrouge: https://github.com/tagucci/pythonrouge
