import os
import random
import logging

import numpy as np
import torch
import src.utils.utils as utils

def set_random_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f'[Note]: seed is set to:{seed}')


def set_benchmark(flag:bool):
    torch.backends.cudnn.enabled = flag
    torch.backends.cudnn.benchmark = flag
    print(f'[Note]: cudnn benchmark is set to:{flag}')


def init_train_url(opt):
    mkdir_list = [opt.train_url]
    dir_names = ['train_log','val_log','test_log','val_imgs','test_imgs','ckpts']
    for dir_name in dir_names:
        mkdir_list.append(os.path.join(opt.train_url,dir_name))
    utils.mkdir(mkdir_list)

def init_logger(opt):
    logger = [None,None,None]
    # train_logger
    if opt.is_train == True:
        utils.setup_logger('train', os.path.join(opt.train_url,'train_log'), 'train', level=logging.INFO,
                screen=True, tofile=True)
        logger[0]=logging.getLogger('train')
        # val_logger
        if opt.val.is_val == True:
            utils.setup_logger('val', os.path.join(opt.train_url,'val_log'), 'val', level=logging.INFO,
                    screen=True, tofile=True)
            logger[1]=logging.getLogger('val')
    # test_logger
    elif opt.is_train == False:
        utils.setup_logger('test', os.path.join(opt.train_url,'test_log'), 'test', level=logging.INFO,
                screen=True, tofile=True)
        logger[2]=logging.getLogger('test')
    return logger