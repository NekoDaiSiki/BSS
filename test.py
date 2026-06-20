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
parser.add_argument('--resume',default=None, type=str)
parser.add_argument('--seed', default=2333 ,type=int)
parser.add_argument('--is_benchmark', default=True, type=bool)
parser.add_argument('--gpu', default='0', type=str)

opt = parser.parse_args()

# init config
opt =  option.parse(opt, is_train=False)
# init environ setting
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
set_random_seed(opt.seed) 
set_benchmark(opt.is_benchmark)
init_train_url(opt)
# init logger 
_,_,test_logger = init_logger(opt)

# validation set
test_dataset_opt = opt.datasets.test
test_dataset = create_dataset(test_dataset_opt)
test_loader = DataLoader(dataset=test_dataset,
                            num_workers=1,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False)

model = create_model(opt)
model.set_logger(test_logger)
model.resume()
model.move_to_device()


# test loop
model.set_submodels_eval()
for i in range(opt.test.repeat_times):
    for index, test_data in enumerate(test_loader):
        model.feed_validate_data(test_data)
        model.validate_forward()
        model.validate_cal_metrics(i,index)
        model.save_results(repeat=i,idx=index)
model.validate_summary()
            

