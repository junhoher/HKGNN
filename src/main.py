import sys
import argparse
import numpy as np
from time import time
from data_loader import load_data
from train import train

np.random.seed(555)


def add_arg(subject, parser):
    n_eps = 10 ; s_smp = 16 ; n_dim = 32
    n_itr = 1  ; s_bch = 65536
    w_l2r = 1e-7 ; w_lsr = 1.0 ; r_lrn = 2e-2

    if subject == 'movie':
        pass

    elif subject == 'book':
        s_smp = 8 ; n_dim = 64
        n_itr = 2 ; s_bch = 256
        w_l2r = 2e-5 ; w_lsr = 0.5 ; r_lrn = 2e-4

    elif subject == 'music':
        s_smp = 8 ; n_dim = 16
        n_itr = 1 ; s_bch = 128
        w_l2r = 1e-4 ; w_lsr = 0.1 ; r_lrn = 5e-4

    elif subject == 'restaurant':
        s_smp = 4 ; n_dim = 8
        n_itr = 2 ; s_bch =65536
        w_l2r = 1e-7 ; w_lsr = 0.5 ; r_lrn = 2e-2

    else:
        print('invalid value for the positional argument!')
        sys.exit()

    parser.add_argument('--dataset', type=str, default=subject, help='which dataset to use')
    parser.add_argument('--n_epochs', type=int, default=n_eps, help='the number of epochs')
    parser.add_argument('--neighbor_sample_size', type=int, default=s_smp, help='the number of neighbors to be sampled')
    parser.add_argument('--dim', type=int, default=n_dim, help='dimension of user and entity embeddings')
    parser.add_argument('--n_iter', type=int, default=n_itr, help='number of iterations when computing entity representation')
    parser.add_argument('--batch_size', type=int, default=s_bch, help='batch size')
    parser.add_argument('--l2_weight', type=float, default=w_l2r, help='weight of L2 regularization')
    parser.add_argument('--ls_weight', type=float, default=w_lsr, help='weight of LS regularization')
    parser.add_argument('--lr', type=float, default=r_lrn, help='learning rate')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str, metavar='Area for the Recommendation',
                        help='which subject for the recommendation ? [movie, book, music, restaurant]')
    p_arg = parser.parse_args()
    subject = p_arg.subject
    add_arg(subject, parser)

    show_loss = False
    show_time = False
    show_topk = False

    t = time()
    args = parser.parse_args()
    data = load_data(args)
    train(args, data, show_loss, show_topk)

    if show_time:
        print('time used: %d s' % (time() - t))


if __name__== '__main__':
    main()
