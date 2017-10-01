import os
import pickle
import argparse

import numpy as np
import _dynet as dy
from tqdm import tqdm

from utils import build_dataset

RANDOM_STATE = 34

def main():
    parser = argparse.ArgumentParser(description='Deep Recurrent Generative Decoder for Abstractive Text Summarization in DyNet')

    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. For cpu, set -1 [default: `-`]')
    parser.add_argument('--n_test', type=int, default=189651, help='Number of test examples [default: `189651`]')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size [default: `5`]')
    parser.add_argument('--max_len', type=int, default=100, help='Maximum length of decoding [default: `100`]')
    parser.add_argument('--model_file', type=str, default='./model_e1', help='Trained model file path [default: `./model_e1`]')
    parser.add_argument('--input_file', type=str, default='./data/valid.article.filter.txt', help='Test file path [default: `./data/valid.article.filter.txt`]')
    parser.add_argument('--output_file', type=str, default='./pred_y.txt', help='Output file path [default: `./pred_y.txt`]')
    parser.add_argument('--w2i_file', type=str, default='./w2i.dump', help='Word2Index file path [default: `./w2i.dump`]')
    parser.add_argument('--i2w_file', type=str, default='./i2w.dump', help='Index2Word file path [default: `./i2w.dump`]')
    parser.add_argument('--alloc_mem', type=int, default=1024, help='Amount of memory to allocate [mb] [default: `1024`]')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    N_TEST    = args.n_test
    K         = args.beam_size
    MAX_LEN   = args.max_len
    ALLOC_MEM = args.alloc_mem

    # File paths
    MODEL_FILE  = args.model_file
    INPUT_FILE  = args.input_file
    OUTPUT_FILE = args.output_file
    W2I_FILE    = args.w2i_file
    I2W_FILE    = args.i2w_file

    # DyNet setting
    dyparams = dy.DynetParams()
    dyparams.set_autobatch(True)
    dyparams.set_random_seed(RANDOM_STATE)
    dyparams.set_mem(ALLOC_MEM)
    dyparams.init()

    # Load trained model ==============================================================================================
    with open(W2I_FILE, 'rb') as f_w2i, open(I2W_FILE, 'rb') as f_i2w:
        w2i = pickle.load(f_w2i)
        i2w = pickle.load(f_i2w)

    test_X, _, _ = build_dataset(INPUT_FILE, w2i=w2i, n_data=N_TEST)

    model = dy.Model()
    V, encoder, decoder = dy.load(MODEL_FILE, model)

    pred_y = []
    for instance_x in tqdm(test_X):
        dy.renew_cg()
        encoder.associate_parameters()
        decoder.associate_parameters()

        x_embs = [dy.lookup(V, x_t) for x_t in instance_x]

        # Initial states
        h = encoder(x_embs)
        h_b_0 = encoder.h_b[0]
        decoder.set_init_states(h, h_b_0)
        s_0 = decoder.s_0
        c_0 = decoder.c_0

        # [accum log prob, BOS, initial hidden state, initial contect vector, decoded sequence]
        candidates = [[0, w2i['<s>'], s_0, c_0, []]]

        t = 0
        while t < MAX_LEN:
            t += 1
            tmp_candidates = []
            end_flag = True
            for score_tm1, y_tm1, s_tm1, c_tm1, y_02tm1 in candidates:
                if y_tm1 == w2i['</s>']:
                    tmp_candidates.append([score_tm1, y_tm1, s_tm1, c_tm1, y_02tm1])
                else:
                    end_flag = False
                    y_tm1_emb = dy.lookup(V, y_tm1)
                    s_t, c_t, _q_t = decoder(y_tm1_emb, tm1s=[s_tm1, c_tm1], test=True)
                    _q_t = np.log(_q_t.npvalue()) # Calculate log probs
                    q_t, y_t = np.sort(_q_t)[::-1][:K], np.argsort(_q_t)[::-1][:K] # Pick K highest log probs and their ids
                    score_t = score_tm1 + q_t # Accumulate log probs
                    tmp_candidates.extend(
                        [[score_tk, y_tk, s_t, c_t, y_02tm1+[y_tk]] for score_tk, y_tk in zip(score_t, y_t)]
                    )
            if end_flag:
                break
            candidates = sorted(tmp_candidates, key=lambda x: -x[0]/len(x[4]))[:K] # Sort in normalized log probs and pick K highest candidates


        # Pick the candidate with the highest log prob
        pred_y.append(candidates[0][-1])

    pred_y_txt = ''
    for pred in pred_y:
        pred_y_txt += ' '.join([i2w[com] for com in pred]) + '\n'

    with open(OUTPUT_FILE, 'w') as f:
        f.write(pred_y_txt)

if __name__ == '__main__':
    main()
