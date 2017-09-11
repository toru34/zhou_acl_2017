import pickle
import argparse

import numpy as np
import dynet as dy
from tqdm import tqdm

from utils import build_dataset
from layers import SelectiveBiGRU, AttentionalGRU

def main():
    parser = argparse.ArgumentParser(description='Selective Encoding for Abstractive Sentence Summarization in DyNet')

    parser.add_argument('--beam_size', type=int, default=5, help='beam size for decoding [default: 5]')
    parser.add_argument('--max_len', type=int, default=50, help='maximum length of decoding')
    parser.add_argument('--model_file', type=str, default='./model', help='model to use for generation [default: ./model]')
    parser.add_argument('--input_file', type=str, default='./data/valid_x.txt', help='input file path [default: ./data/valid_x.txt]')
    parser.add_argument('--output_file', type=str, default='./pred_y.txt', help='output file path [default: ./pred_y.txt]')
    parser.add_argument('--w2i_file', type=str, default='./w2i.dump', help='w2i file path [default: ./w2i.dump]')
    parser.add_argument('--i2w_file', type=str, default='./i2w.dump', help='i2w file path [default: ./i2w.dump]')
    args = parser.parse_args()

    K = args.beam_size
    MAX_LEN = args.max_len
    W2I_FILE = args.w2i_file
    I2W_FILE = args.i2w_file
    INPUT_FILE = args.input_file
    OUTPUT_FILE = args.output_file
    MODEL_FILE = args.model_file

    # Load model
    with open(W2I_FILE, 'rb') as f_w2i, open(I2W_FILE, 'rb') as f_i2w:
        w2i = pickle.load(f_w2i)
        i2w = pickle.load(f_i2w)

    test_X, _, _ = build_dataset(INPUT_FILE, w2i=w2i)

    # # Use small dataset
    # test_X = test_X[:50]

    model = dy.Model()
    encoder, decoder, V = dy.load(MODEL_FILE, model)

    # Generate
    pred_y_txt = ''
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
        y = candidates[0][4]
        pred_y_txt += ' '.join([i2w[y_t] for y_t in y]) + '\n'

    with open(OUTPUT_FILE, 'w') as f:
        f.write(pred_y_txt)

if __name__ == '__main__':
    main()
