import os
import argparse
import pprint
from torch.utils.data import DataLoader
from src.utils.env_set import set_benchmark, set_random_seed, init_train_url, init_logger
from src.utils.train_stages import apply_mean_loss_stage, mean_loss_stage_for_epoch, validate_mean_loss_stages
import src.utils.options as option
from src.datasets import create_dataset
from src.model import create_model
parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, type=str)
parser.add_argument('--seed', default=2333, type=int)
parser.add_argument('--is_benchmark', default=True, type=option.parse_bool)
parser.add_argument('--gpu', default='0', type=str)
opt = parser.parse_args()
opt = option.parse(opt, is_train=True)
mean_loss_stages = validate_mean_loss_stages(opt.train)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
set_random_seed(opt.seed)
set_benchmark(opt.is_benchmark)
init_train_url(opt)
train_logger, val_logger, _ = init_logger(opt)
train_logger.info(pprint.pformat(opt, sort_dicts=False))
train_dataset_opt = opt.datasets.train
training_dataset = create_dataset(train_dataset_opt)
train_loader = DataLoader(dataset=training_dataset, num_workers=8, batch_size=train_dataset_opt.batch_size, shuffle=True, pin_memory=True, drop_last=True)
validate_dataset_opt = opt.datasets.validate
validate_dataset = create_dataset(validate_dataset_opt)
validate_loader = DataLoader(dataset=validate_dataset, num_workers=1, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
model = create_model(opt)
model.print_submodels(train_logger)
model.resume()
start_epoch = model.epoch + 1 if opt.resume.path and opt.resume.resume_optim else 0
active_mean_loss_stage = None
model.move_to_device()
for epoch in range(start_epoch, opt.train.total_epoch):
    if mean_loss_stages:
        stage = mean_loss_stage_for_epoch(mean_loss_stages, epoch)
        weight = apply_mean_loss_stage(model.loss_params, stage, epoch)
        if stage is not active_mean_loss_stage:
            train_logger.info('Loss stage for epochs %s-%s: ws=%s, lower=%s, thres=%s', stage['start_epoch'], stage['end_epoch'] - 1, stage['ws'], model.loss_params['lower'], model.loss_params['mean_loss_params']['thres'])
            active_mean_loss_stage = stage
        if 'weight_end' in stage:
            train_logger.info('Mean loss weight for epoch %s: %.6f', epoch, weight)
    model.train_epoch_init(train_logger, epoch)
    for index, train_data in enumerate(train_loader):
        model.feed_train_data(train_data)
        model.train_forward()
        model.train_cal_loss()
        model.optimize_params()
        model.train_logging()
        model.update_iter()
    model.scheduler_step()
    if (epoch + 1) % opt.train.save_freq == 0:
        model.save()
    if opt.val.is_val:
        if (epoch + 1) % opt.val.val_freq == 0:
            model.set_logger(val_logger)
            model.set_submodels_eval()
            for i in range(opt.val.repeat_times):
                for index, val_data in enumerate(validate_loader):
                    model.feed_validate_data(val_data)
                    model.validate_forward()
                    model.validate_cal_metrics(i, index)
                    model.save_results(repeat=i, idx=index)
            model.validate_summary()
