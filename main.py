# coding:utf-8
import os
import logging
import json

from cotk.dataloader import TextSummarization
# from cotk.wordvector import WordVector, Glove

from utils import debug, try_cache, cuda_init, Storage
from pointer_gen import PointerGen

def main(args, load_exclude_set=[], restoreCallback=None):

    logging.basicConfig(filename=0,
		        level=logging.DEBUG,
		        format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s',
		        datefmt='%H:%M:%S')

    if args.debug:
        debug()

    logging.info(json.dumps(args, indent=2))

    cuda_init(0, args.cuda)

    volatile = Storage()
    volatile.load_exclude_set = load_exclude_set
    volatile.restoreCallback = restoreCallback

    data_arg = Storage()
    data_arg.file_id = args.datapath

    data_class = TextSummarization.load_class(args.dataset)
    volatile.dm = try_cache(data_class, (*data_arg), args.cache_dir, data_class.__name__) if args.cache else data_class(**data_arg)

    param = Storage()
    param.args = args
    param.volatile = volatile

    model = PointerGen(param)
    if args.mode == "train":
        model.train_process()
    elif args.mode == "eval":
        test_res = model.test_process()
        json.dump(test_res, open("./result.json", "w"))
    else:
        raise ValueError("Unknown mode")
