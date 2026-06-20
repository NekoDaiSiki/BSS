import os
import os.path as osp

import yaml
from addict import Dict
from collections import OrderedDict
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

class NoneDict(Dict): # 这个是继承的 addict.Dict
     def __missing__(self, key):
        return None

class nonedict(dict): # 这个继承的是 默认的 dict 类
    def __missing__(self, key):
        return None

def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def parse(opt,is_train=True):
    Loader, Dumper = OrderedYaml()
    opt = NoneDict(vars(opt))
    opt.is_train = is_train
    opt.is_test = not is_train
    with open(opt.config, mode='r') as f:
        config = yaml.load(f, Loader=Loader)
    config = NoneDict(config)  
    if opt.is_test:
        # back process config
        config.resume.path = opt.resume
    opt.update(config)

    # update opt
    opt.train_url = os.path.join(opt.train_url,opt.task_name)

    return opt