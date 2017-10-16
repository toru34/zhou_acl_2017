import os
import time
import pickle
import argparse

import numpy as np
import _dynet as dy
from tqdm import tqdm

from utils import Dataset, associate_parameters
from layers import SelectiveBiGRU, AttentionalGRU

RANDOM_SEED = 34
np.random.seed(RANDOM_SEED)

def main():
    parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in DyNet')

    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. For cpu, set -1 [default: -1]')
    parser.add_argument('--n_epochs', type=int, default=3, help='Number of epochs [default: 3]')
    parser.add_argument('--n_train', type=int, default=3803957, help='Number of training data (up to 3803957 in gigaword) [default: 3803957]')
    parser.add_argument('--n_valid', type=int, default=189651, help='Number of validation data (up to 189651 in gigaword) [default: 189651])')
    parser.add_argument('--batch_size', type=int, default=32, help='Mini batch size [default: 32]')
    parser.add_argument('--vocab_size', type=int, default=124404, help='Vocabulary size [default: 124404]')
    parser.add_argument('--emb_dim', type=int, default=256, help='Embedding size [default: 256]')
    parser.add_argument('--hid_dim', type=int, default=256, help='Hidden state size [default: 256]')
    parser.add_argument('--maxout_dim', type=int, default=5, help='Maxout size [default: 2]')
    parser.add_argument('--alloc_mem', type=int, default=8192, help='Amount of memory to allocate [mb] [default: 8192]')
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    N_EPOCHS   = args.n_epochs
    N_TRAIN    = args.n_train
    N_VALID    = args.n_valid
    BATCH_SIZE = args.batch_size
    VOCAB_SIZE = args.vocab_size
    EMB_DIM    = args.emb_dim
    HID_DIM    = args.hid_dim
    MAXOUT_DIM = args.maxout_dim
    ALLOC_MEM  = args.alloc_mem

    # File paths
    TRAIN_X_FILE = './data/train.article.txt'
    TRAIN_Y_FILE = './data/train.title.txt'
    VALID_X_FILE = './data/valid.article.filter.txt'
    VALID_Y_FILE = './data/valid.title.filter.txt'

    # DyNet setting
    dyparams = dy.DynetParams()
    dyparams.set_autobatch(True)
    dyparams.set_random_seed(RANDOM_SEED)
    dyparams.set_mem(ALLOC_MEM)
    dyparams.init()

    # Build dataset
    dataset = Dataset(
        TRAIN_X_FILE,
        TRAIN_Y_FILE,
        VALID_X_FILE,
        VALID_Y_FILE,
        vocab_size=VOCAB_SIZE,
        batch_size=BATCH_SIZE,
        n_train=N_TRAIN,
        n_valid=N_VALID
    )
    VOCAB_SIZE = len(dataset.w2i)
    print('VOCAB_SIZE', VOCAB_SIZE)

    # Build model
    model = dy.Model()
    trainer = dy.AdamTrainer(model)

    V = model.add_lookup_parameters((VOCAB_SIZE, EMB_DIM))
    encoder = SelectiveBiGRU(model, EMB_DIM, HID_DIM)
    decoder = AttentionalGRU(model, EMB_DIM, HID_DIM, MAXOUT_DIM, VOCAB_SIZE)

    # Train model
    start_time = time.time()
    for epoch in range(N_EPOCHS):
        # Train
        loss_all_train = []
        dataset.reset_train_iter()
        for train_x_mb, train_y_mb in tqdm(dataset.train_iter):
            # Create a new computation graph
            dy.renew_cg()
            associate_parameters([encoder, decoder])
            losses = []
            for x, t in zip(train_x_mb, train_y_mb):
                t_in, t_out = t[:-1], t[1:]

                # Encoder
                x_embs = [dy.lookup(V, x_t) for x_t in x]
                hp, hb_1 = encoder(x_embs)

                # Decoder
                decoder.set_initial_states(hp, hb_1)
                t_embs = [dy.lookup(V, t_t) for t_t in t_in]
                y = decoder(t_embs)

                # Loss
                loss = dy.esum(
                    [dy.pickneglogsoftmax(y_t, t_t) for y_t, t_t in zip(y, t_out)]
                )
                losses.append(loss)

            mb_loss = dy.average(losses)

            # Forward prop
            loss_all_train.append(mb_loss.value())

            # Backward prop
            mb_loss.backward()
            trainer.update()

        # Valid
        loss_all_valid = []
        dataset.reset_valid_iter()
        for valid_x_mb, valid_y_mb in dataset.valid_iter:
            # Create a new computation graph
            dy.renew_cg()
            associate_parameters([encoder, decoder])
            losses = []
            for x, t in zip(valid_x_mb, valid_y_mb):
                t_in, t_out = t[:-1], t[1:]

                # Encoder
                x_embs = [dy.lookup(V, x_t) for x_t in x]
                hp, hb_1 = encoder(x_embs)

                # Decoder
                decoder.set_initial_states(hp, hb_1)
                t_embs = [dy.lookup(V, t_t) for t_t in t_in]
                y = decoder(t_embs)

                # Loss
                loss = dy.esum(
                    [dy.pickneglogsoftmax(y_t, t_t) for y_t, t_t in zip(y, t_out)]
                )
                losses.append(loss)

            mb_loss = dy.average(losses)

            # Forward prop
            loss_all_valid.append(mb_loss.value())

        print('EPOCH: %d, Train Loss: %.3f, Valid Loss: %.3f, Time: %.3f[s]' % (
            epoch+1,
            np.mean(loss_all_train),
            np.mean(loss_all_valid),
            time.time()-start_time
        ))

        # Save model
        dy.save('./model_e'+str(epoch+1), [V, encoder, decoder])
        with open('./w2i.dump', 'wb') as f_w2i, open('./i2w.dump', 'wb') as f_i2w:
            pickle.dump(dataset.w2i, f_w2i)
            pickle.dump(dataset.i2w, f_i2w)

if __name__ == '__main__':
    main()
