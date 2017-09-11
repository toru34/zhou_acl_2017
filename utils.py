from collections import defaultdict

import numpy as np

def build_word2count(file_path, w2c=None):
    if w2c is None:
        w2c = defaultdict(lambda: 0)
    for line in open(file_path, encoding='utf-8'):
        sentence = line.strip().split()
        for word in sentence:
            w2c[word] += 1
    return w2c

def encode(sentence, w2i):
    encoded_sentence = []
    for word in sentence:
        if word in w2i:
            encoded_sentence.append(w2i[word])
        else:
            encoded_sentence.append(w2i['<unk>'])
    return encoded_sentence

def build_dataset(file_path, vocab_size=10000, w2c=None, w2i=None, target=False):
    if w2i is None:
        sorted_w2c = sorted(w2c.items(), key=lambda x: -x[1])
        w2i = {w: np.int32(i+3) for i, (w, c) in enumerate(sorted_w2c[:vocab_size-3])}
        w2i['<s>'], w2i['</s>'] = np.int32(0), np.int32(1)
        w2i['<unk>'] = np.int32(2)

    data = []
    for line in open(file_path, encoding='utf-8'):
        sentence = line.strip().split()
        if target:
            sentence = ['<s>'] + sentence + ['</s>']
        encoded_sentence = encode(sentence, w2i)
        data.append(encoded_sentence)
    i2w = {i: w for w, i in w2i.items()}
    return data, w2i, i2w
