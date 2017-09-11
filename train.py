import math
import time
import argparse
import pickle

import _dynet as dy
import numpy as np
from sklearn.utils import shuffle

from utils import build_word2count, build_dataset
from layers import SelectiveBiGRU, AttentionalGRU

RANDOM_STATE = 42

def main():
    parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in DyNet')

    parser.add_argument('--n_epochs', type=int, default=3, help='number of epochs for training [default: 3]')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training [default: 16]')
    parser.add_argument('--emb_dim', type=int, default=32, help='embedding size for each word [default: 32]')
    parser.add_argument('--hid_dim', type=int, default=32, help='hidden state size for both encoder and decoder [default: 32]')
    parser.add_argument('--vocab_size', type=int, default=10000, help='vocabulary size [default: 10000]')
    parser.add_argument('--maxout_dim', type=int, default=5, help='maxout size [default: 5]')
    parser.add_argument('--alloc_mem', type=int, default=1024, help='amount of memory to allocate[mb] [default: 1024]')
    args = parser.parse_args()

    vocab_size = args.vocab_size
    EMB_DIM = args.emb_dim
    HID_DIM = args.hid_dim
    BATCH_SIZE = args.batch_size
    MAXOUT_DIM = args.maxout_dim
    N_EPOCHS = args.n_epochs
    MEMORY_SIZE = args.alloc_mem

    # DyNet setting
    dyparams = dy.DynetParams()
    dyparams.set_autobatch(True)
    dyparams.set_mem(MEMORY_SIZE)
    dyparams.set_random_seed(RANDOM_STATE)
    dyparams.init()

    # Build dataset ================================================================================
    w2c = build_word2count('./data/train_x.txt')
    w2c = build_word2count('./data/train_y.txt', w2c)

    train_X, w2i, i2w = build_dataset('./data/train_x.txt', vocab_size=vocab_size, w2c=w2c, target=False)
    train_y, _, _ = build_dataset('./data/train_y.txt', w2i=w2i, target=True)

    valid_X, _, _ = build_dataset('./data/valid_x.txt', w2i=w2i, target=False)
    valid_y, _, _ = build_dataset('./data/valid_y.txt', w2i=w2i, target=True)

    # # Use small dataset
    # train_X, train_y = train_X[:1000], train_y[:1000]
    # valid_X, valid_y = valid_X[:10], valid_y[:10]

    vocab_size = len(w2i)

    # Build model ================================================================================
    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    V = model.add_lookup_parameters((vocab_size, EMB_DIM))
    encoder = SelectiveBiGRU(EMB_DIM, HID_DIM, model)
    decoder = AttentionalGRU(EMB_DIM, HID_DIM, MAXOUT_DIM, vocab_size, model)


    # Train model ====================================================================================
    n_batches_train = math.ceil(len(train_X)/BATCH_SIZE)
    n_batches_valid = math.ceil(len(valid_X)/BATCH_SIZE)
    start_time = time.time()

    for epoch in range(N_EPOCHS):
        # Train
        train_X, train_y = shuffle(train_X, train_y, random_state=RANDOM_STATE)
        loss_all_train = []
        for i in range(n_batches_train):
            # Create a new computation graph
            dy.renew_cg()
            encoder.associate_parameters()
            decoder.associate_parameters()

            # Create a mini batch
            start = i*BATCH_SIZE
            end = start + BATCH_SIZE
            train_X_mb = train_X[start:end]
            train_y_mb = train_y[start:end]

            losses = []
            for instance_x, instance_y in zip(train_X_mb, train_y_mb):
                x_embs = [dy.lookup(V, x_t) for x_t in instance_x]

                # Encoder
                h = encoder(x_embs)
                h_b_0 = encoder.h_b[0]

                # Decoder
                t = instance_y[1:]
                y_embs = [dy.lookup(V, y_t) for y_t in instance_y[:-1]]
                decoder.set_init_states(h, h_b_0)
                y = decoder(y_embs)
                loss = dy.esum([dy.pickneglogsoftmax(y_t, t_t) for y_t, t_t in zip(y, t)])
                losses.append(loss)

            mb_loss = dy.average(losses)

            # Forward propagation
            loss_all_train.append(mb_loss.value())

            # Backward propagation
            mb_loss.backward()
            trainer.update()

        # Vaild
        loss_all_valid = []
        for i in range(n_batches_valid):
            # Create a new computation graph
            dy.renew_cg()
            encoder.associate_parameters()
            decoder.associate_parameters()

            # Create a mini batch
            start = i*BATCH_SIZE
            end = start + BATCH_SIZE
            valid_X_mb = valid_X[start:end]
            valid_y_mb = valid_y[start:end]

            losses = []
            for instance_x, instance_y in zip(valid_X_mb, valid_y_mb):
                x_embs = [dy.lookup(V, x_t) for x_t in instance_x]

                # Encoder
                h = encoder(x_embs)
                h_b_0 = encoder.h_b[0]

                # Decoder
                t = instance_y[1:]
                y_embs = [dy.lookup(V, y_t) for y_t in instance_y[:-1]]
                decoder.set_init_states(h, h_b_0)
                y = decoder(y_embs)
                loss = dy.esum([dy.pickneglogsoftmax(y_t, t_t) for y_t, t_t in zip(y, t)])
                losses.append(loss)

            mb_loss = dy.average(losses)

            # Forward propagation
            loss_all_valid.append(mb_loss.value())

        end_time = time.time()
        print('EPOCH: %d, Train Loss: %.3f, Valid Loss: %.3f, Time: %.3f[s]' % (
            epoch+1,
            np.mean(loss_all_train),
            np.mean(loss_all_valid),
            end_time-start_time,
        ))

    # Save model =================================================================================
    dy.save('./model', [encoder, decoder, V])
    with open('./w2i.dump', 'wb') as f_w2i, open('./i2w.dump', 'wb') as f_i2w:
        pickle.dump(w2i, f_w2i)
        pickle.dump(i2w, f_i2w)

if __name__ == '__main__':
    main()
