import math
import time
import argparse

import _dynet as dy
import numpy as np
from sklearn.utils import shuffle

from utils import build_word2count, build_dataset
from layers import SelectiveBiGRU, AttentionalGRU

random_state = 42

# Activate autobatching
dyparams = dy.DynetParams()
dyparams.set_autobatch(True)
dyparams.set_autobatch(random_state)
dyparams.init()

def main():
    parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in DyNet')

    parser.add_argument('--n_epochs', type=int, default=3, help='number of epochs for training [default: 3]')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training [default: 16]')
    parser.add_argument('--emb_dim', type=int, default=32, help='embedding size for each word [default: 32]')
    parser.add_argument('--hid_dim', type=int, default=32, help='hidden state size for both encoder and decoder [default: 32]')
    parser.add_argument('--vocab_size', type=int, default=10000, help='vocabulary size [default: 10000]')
    parser.add_argument('--maxout_dim [default: 5]', type=int, default=5, help='maxout size')
    args = parser.parse_args()

    vocab_size = args.vocab_size
    emb_dim = args.emb_dim
    hid_dim = args.hid_dim
    batch_size = args.batch_size
    maxout_dim = args.maxout_dim
    n_epochs = args.n_epochs

    # Build dataset ================================================================================
    w2c = build_word2count('./data/train_x.txt')
    w2c = build_word2count('./data/train_y.txt', w2c)

    train_X, w2i, i2w = build_dataset('./data/train_x.txt', vocab_size=vocab_size, w2c=w2c, target=False)
    train_y, _, _ = build_dataset('./data/train_y.txt', w2i=w2i, target=True)

    valid_X, _, _ = build_dataset('./data/valid_x.txt', w2i=w2i, target=False)
    valid_y, _, _ = build_dataset('./data/valid_y.txt', w2i=w2i, target=True)

    vocab_size = len(w2i)

    # Build model ================================================================================
    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    V = model.add_lookup_parameters((vocab_size, emb_dim))
    encoder = SelectiveBiGRU(emb_dim, hid_dim, model)
    decoder = AttentionalGRU(emb_dim, hid_dim, maxout_dim, vocab_size, model)

    n_batches = math.ceil(len(train_X)/batch_size)
    start_time = time.time()

    for epoch in range(n_epochs):
        train_X, train_y = shuffle(train_X, train_y, random_state=random_state)
        loss_all = []
        for i in range(n_batches):
            # Create a new computation graph
            dy.renew_cg()
            encoder.associate_parameters()
            decoder.associate_parameters()

            # Create a mini batch
            start = i*batch_size
            end = start + batch_size
            train_X_mb = train_X[start:end]
            train_y_mb = train_y[start:end]

            losses = []
            for instance_x, instance_y in zip(train_X_mb, train_y_mb):
                x_embs = [dy.lookup(V, x_t) for x_t in instance_x]

                # Encoder
                h = encoder.f_prop(x_embs)
                h_b_0 = encoder.h_b[0]

                # Decoder
                t = instance_y[1:]
                y_embs = [dy.lookup(V, y_t) for y_t in instance_y[:-1]]
                decoder.set_init_states(h, h_b_0)
                y = decoder.f_prop(y_embs)
                loss = dy.esum([dy.pickneglogsoftmax(y_t, t_t) for y_t, t_t in zip(y, t)])
                losses.append(loss)

            mb_loss = dy.average(losses)

            # Forward propagation
            loss_all.append(mb_loss.value())

            # Backward propagation
            mb_loss.backward()
            trainer.update()

        end_time = time.time()
        print('EPOCH: %d, Train Loss: %.3f, Time: %.3f[s]' % (
            epoch+1,
            np.mean(loss_all),
            end_time-start_time,
        ))

if __name__ == '__main__':
    main()
