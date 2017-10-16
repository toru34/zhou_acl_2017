import math
from collections import defaultdict

import numpy as np
import _dynet as dy
from sklearn.utils import shuffle

def np_log(x):
    return np.log(np.clip(x, 1e-6, x))

def dy_log(x):
    return dy.log(x+1e-6)

def associate_parameters(layers):
    for layer in layers:
        layer.associate_parameters()

def encode(sentence, w2i, unksym='<unk>'):
    encoded_sentence = []
    for word in sentence:
        if word in w2i:
            encoded_sentence.append(w2i[word])
        else:
            encoded_sentence.append(w2i[unksym])
    return encoded_sentence

def build_dataset(file_path, w2i, unksym='<unk>', target=False, n_data=100000000):
    data = []
    for i, line in enumerate(open(file_path, encoding='utf-8', errors='ignore')):
        sentence = line.strip().split()
        if target:
            sentence = ['<s>'] + sentence + ['</s>']
        encoded_sentence = encode(sentence, w2i, unksym)
        data.append(encoded_sentence)
        if i >= n_data:
            break
    i2w = {i: w for w, i in w2i.items()}
    return data, w2i, i2w


class Dataset:
    def __init__(
        self,
        train_x_path,
        train_y_path,
        valid_x_path,
        valid_y_path,
        vocab=None,
        vocab_size=60000,
        unksym='<unk>',
        len_lim=100,
        n_train=10000000,
        n_valid=10000000,
        batch_size=32,
    ):
        self.train_x_path = train_x_path
        self.train_y_path = train_y_path
        self.valid_x_path = valid_x_path
        self.valid_y_path = valid_y_path

        self.vocab      = vocab
        self.vocab_size = vocab_size
        self.unksym     = unksym
        self.len_lim    = len_lim
        self.n_train    = n_train
        self.n_valid    = n_valid
        self.batch_size = batch_size
        self.w2c        = None
        self.w2i        = None
        self.i2w        = None
        self.max_len    = 0
        self.max_len    = 0

        self.build_word2count(train_x_path, n_train)
        self.build_word2count(train_y_path, n_train)

        self.train_x = self.build_dataset(train_x_path, n_train, target=False)
        self.train_y = self.build_dataset(train_y_path, n_train, target=True)
        self.valid_x = self.build_dataset(valid_x_path, n_valid, target=False)
        self.valid_y = self.build_dataset(valid_y_path, n_valid, target=True)
        self.vocab_size = len(self.w2i)

        # Make mini batches
        self.n_batches_train = math.ceil(len(self.train_x)/batch_size)
        self.n_batches_valid = math.ceil(len(self.valid_x)/batch_size)
        self.reset_train_iter()
        self.reset_valid_iter()

    def sort_by_length(self, data_x, data_y):
        data_x_lens = [len(com) for com in data_x]
        sorted_data_indexes = sorted(range(len(data_x_lens)), key=lambda x: -data_x_lens[x])

        data_x = [data_x[ind] for ind in sorted_data_indexes]
        data_y = [data_y[ind] for ind in sorted_data_indexes]

        return data_x, data_y

    def encode(self, sentence):
        encoded_sentence = []
        for word in sentence:
            if word in self.w2i:
                encoded_sentence.append(self.w2i[word])
            else:
                encoded_sentence.append(self.w2i[self.unksym])
        return encoded_sentence

    def build_word2count(self, data_path, n_data):
        if self.w2c is None:
            self.w2c = defaultdict(lambda: 0)
        for i, line in enumerate(open(data_path, encoding='utf-8', errors='ignore')):
            sentence = line.strip().split()
            if len(sentence) > self.len_lim:
                continue
            for word in sentence:
                if self.vocab:
                    if word in vocab:
                        self.w2c[word] += 1
                else:
                    self.w2c[word] += 1
            if i >= n_data:
                break

    def build_dataset(self, data_path, n_data, target):
        if self.w2i is None:
            sorted_w2c = sorted(self.w2c.items(), key=lambda x: -x[1])
            sorted_w = [w for w, c in sorted_w2c]

            self.w2i = {}
            word_id = 0
            self.w2i['<s>'], self.w2i['</s>'] = np.int32(word_id), np.int32(word_id+1)
            word_id += 2

            if self.unksym not in sorted_w:
                self.w2i[self.unksym] = np.int32(word_id)
                word_id += 1
            else:
                if sorted_w.index(self.unksym) >= self.vocab_size-word_id:
                    self.w2i[self.unksym] = np.int32(word_id)
                    word_id += 1
            w2i_update = {w: np.int32(i+word_id) for i, w in enumerate(sorted_w[:self.vocab_size-word_id])}
            self.w2i.update(w2i_update)

        data = []
        for i, line in enumerate(open(data_path, encoding='utf-8', errors='ignore')):
            sentence = line.strip().split()
            if len(sentence) > self.len_lim:
                continue
            if target:
                sentence = ['<s>'] + sentence + ['</s>']
            encoded_sentence = self.encode(sentence)
            self.max_len = max(self.max_len, len(encoded_sentence))
            data.append(encoded_sentence)
            if i+1 >= n_data:
                break
        self.i2w = {i: w for w, i in self.w2i.items()}
        return data

    def reset_train_iter(self):
        self.train_x, self.train_y = shuffle(self.train_x, self.train_y)
        self.train_iter = iter(
            [(self.train_x[i*self.batch_size:(i+1)*self.batch_size], self.train_y[i*self.batch_size:(i+1)*self.batch_size]) for i in range(self.n_batches_train)]
        )

    def reset_valid_iter(self):
        self.valid_iter = iter(
            [(self.valid_x[i*self.batch_size:(i+1)*self.batch_size], self.valid_y[i*self.batch_size:(i+1)*self.batch_size]) for i in range(self.n_batches_valid)]
        )
