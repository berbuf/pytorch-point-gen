import argparse
import time
import random
import torch
import numpy as np
import sys

from seq2seq_pytorch.utils import Storage
from main import main


def run(*argv):
    parser = argparse.ArgumentParser(
        description="A pytorch implementation of the paper Get to the point (https://arxiv.org/abs/1704.04368)")

    # Model parameters
    parser.add_argument('--restore', type=str, default=None,
                        help='Checkpoints name to load. \
                        "NAME_last" for the last checkpoint of model named NAME. "NAME_best" means the best checkpoint. \
                        You can also use "last" and "best", by default use last model you run. \
                        It can also be an url started with "http". \
                        Attention: "NAME_last" and "NAME_best" are not guaranteed to work when 2 models with same name run in the same time. \
                        "last" and "best" are not guaranteed to work when 2 models run in the same time.\
                        Default: None (don\'t load anything)')
    parser.add_argument(
        '--name', type=str, help="Name for experiment. Logs will be saved in a directory with this name, under log_root.", default=time.ctime(time.time()).replace(" ", "_"))
    parser.add_argument('--datapath', type=str,
                        default="../data/#CNN", help='path to the dataset')
    parser.add_argument('--dataset', type=str, default='CNN',
                        help='Dataloader class')
    parser.add_argument('--mode', type=str, default="train",
                        help='must be one of train/eval')
    parser.add_argument('--model_dir', type=str, default="./model",
                        help='Checkpoints directory for model. Default: ./model')
    parser.add_argument('--checkpoint_steps', type=int, help="", default=20)
    parser.add_argument('--checkpoint_max_to_keep',
                        type=int, help="", default=5)
    parser.add_argument('--log_dir', type=str, default="./tensorboard",
                        help='Log directory for tensorboard. Default: ./tensorboard')
    parser.add_argument('--cuda', type=bool, help="", default=False)
    parser.add_argument('--cache', type=str, help="", default="")
    parser.add_argument('--seed', type=int, help='', default=42)
    parser.add_argument('--restore_optimizer',
                        type=bool, default=True, help='')
    parser.add_argument('--debug', action='store_true',
                        help='Enter debug mode (using ptvsd).')

    # Network parameters
    parser.add_argument('--pointer_gen', type=bool,
                        help='If True, use pointer-generator model. If False, use baseline model.', default=True)
    parser.add_argument('--coverage', type=bool, help='Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.', default=False)
    parser.add_argument('--embedding_size', type=int,
                        help='dimension of word embeddings', default=128)
    parser.add_argument('--eh_size', type=int,
                        help='dimension of RNN encoder hidden states', default=256)
    parser.add_argument('--dh_size', type=int,
                        help='dimension of RNN decoder hidden states', default=256)
    parser.add_argument('--epochs', type=int, default=30,
                        help="Epoch for training. Default: 100")
    parser.add_argument('--batch_per_epoch', type=int, default=2,
                        help="Batches per epoch. Default: 1500")
    parser.add_argument('--batch_num_per_gradient', type=int, default=2,
                        help="")
    parser.add_argument('--batch_size', type=int,
                        help='minibatch size', default=16)
    parser.add_argument('--grad_clip', type=int,
                        help='', default=5)
    parser.add_argument('--lr', type=float, help='learning rate', default=.015)
    parser.add_argument('--droprate', type=float, help="", default=.0)
    parser.add_argument('--batchnorm', type=bool, help="", default=False)
    parser.add_argument('--max_sent_length', type=int, help="", default=100)
    parser.add_argument('--max_doc_length', type=int, help="", default=400)
    parser.add_argument('--min_vocab_times', type=int, help="", default=50)

    # Decode parameters
    parser.add_argument('--cov_loss_wt', type=float,
                        help='Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.', default=1.)
    parser.add_argument('--show_sample', type=list, help="", default=[0])
    parser.add_argument('--decode_mode', type=str, choices=['max', 'sample', 'gumbel', 'samplek', 'beam'], default='beam',
                        help='The decode strategy when freerun. Choices: max, sample, gumbel(=sample), \
                        samplek(sample from topk), beam(beamsearch). Default: beam')
    parser.add_argument('--top_k', type=int, default=10,
                        help='The top_k when decode_mode == "beam" or "samplek"')
    parser.add_argument('--out_dir', type=str, default="./output",
                        help='Output directory for test output. Default: ./output')
    parser.add_argument('--length_penalty', type=float, default=0.7,
                        help='The beamsearch penalty for short sentences. The penalty will get\
                        larger when this becomes smaller.')

    args = Storage()
    for key, val in vars(parser.parse_args(argv)).items():
        args[key] = val

    random.seed(args.seed), torch.manual_seed(
        args.seed), np.random.seed(args.seed)

    main(args)


if __name__ == '__main__':
    run(*sys.argv[1:])
