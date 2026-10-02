import argparse
import os
import yaml
from addict import Dict


class NoneDict(Dict):
    def __missing__(self, key):
        return None


def parse_bool(value):
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected true or false')


def parse(opt, is_train=True):
    opt = NoneDict(vars(opt))
    opt.is_train = is_train
    opt.is_test = not is_train
    with open(opt.config, encoding='utf-8') as handle:
        config = NoneDict(yaml.safe_load(handle))
    if opt.is_test:
        config.resume.path = opt.resume
    opt.update(config)
    opt.train_url = os.path.join(opt.train_url, opt.task_name)
    return opt
