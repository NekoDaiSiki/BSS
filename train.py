import os
import argparse
import datetime
import logging

import pprint
from torch.utils.data import DataLoader

from src.utils.env_set import set_benchmark,set_random_seed,init_train_url,init_logger
import src.utils.options as option
import src.utils.utils as utils
from src.datasets import create_dataset
from src.model import create_model

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--seed', default=2333 ,type=int)
parser.add_argument('--is_benchmark', default=True, type=str)
parser.add_argument('--gpu', default='0', type=str)

opt = parser.parse_args()

# init config
opt =  option.parse(opt, is_train=True)

# init environ setting
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
set_random_seed(opt.seed) 
set_benchmark(bool(opt.is_benchmark))
init_train_url(opt)
# init logger 
train_logger,val_logger,_ = init_logger(opt)
train_logger.info(pprint.pformat(opt,sort_dicts=False))

# Training Set
train_dataset_opt = opt.datasets.train
training_dataset = create_dataset(train_dataset_opt)
train_loader = DataLoader(dataset=training_dataset,
                            num_workers=8,
                            batch_size=train_dataset_opt.batch_size,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)

# validation set
validate_dataset_opt = opt.datasets.validate
validate_dataset = create_dataset(validate_dataset_opt)
validate_loader = DataLoader(dataset=validate_dataset,
                            num_workers=1,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False)


model = create_model(opt)
model.print_submodels(train_logger) # print submodels and tranable models

model.resume()
start_epoch = model.epoch + 1 if opt.resume.path and opt.resume.resume_optim else 0

model.move_to_device()

# pretrained loop
# if opt.pretrain.is_pretrain:
#     for epoch in range(opt.pretrain.pretrain_epoch):
#         for index, pretrain_data in enumerate(train_loader):
#             model.feed_pretrain_data(pretrain_data)
#             model.pretrain_forward()
#             model.pretrain_cal_loss()
#             model.optimize_params()

#     model.set_iter()
#     model.set_epoch()

# training loop
for epoch in range(start_epoch, opt.train.total_epoch):
#     # set logger, epoch, curr_iter
    model.train_epoch_init(train_logger,epoch)

    for index, train_data in enumerate(train_loader):
        # train forward and backward
        model.feed_train_data(train_data)
        model.train_forward()
        model.train_cal_loss()
        model.optimize_params()      

        # logging and update 
        model.train_logging()  
        model.update_iter()
    model.scheduler_step()
    
    if (epoch+1)%opt.train.save_freq==0:
        model.save()

    # validation loop
    if opt.val.is_val:
        if (epoch+1)%opt.val.val_freq==0:
            model.set_logger(val_logger)
            model.set_submodels_eval()
            for i in range(opt.val.repeat_times):
                for index, val_data in enumerate(validate_loader):
                    model.feed_validate_data(val_data)
                    model.validate_forward()
                    model.validate_cal_metrics(i,index)
                    model.save_results(repeat=i,idx=index)
            model.validate_summary()
             


