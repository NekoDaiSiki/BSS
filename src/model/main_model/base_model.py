import os
import abc
import pprint

import torch
import torch.nn as nn
import torch.optim as optim

class BaseModel(abc.ABC):
    def __init__(self,opt) -> None:
        super().__init__()
        self.gpu = opt.gpu
        self.train_url = opt.train_url
        self.loss_params = opt.train.loss_params
        self.is_train = opt.is_train
        self.is_test = opt.is_test

        # opt
        self.resume_opt = opt.resume
        self.datasets_opt = opt.datasets
        self.main_model_opt = opt.models.main_model
        self.sub_models_opt = opt.models.sub_models
        self.train_opt = opt.train
        self.validate_opt = opt.val
        self.test_opt = opt.test
        self.optim_opt = opt.train.optimizer
        self.sched_opt = opt.train.scheduler
        

        self.device = torch.device(opt.device)
        self.sub_models = {}
        self.train_sub_models = {}
        # massage for logger
        self.epoch = 0
        self.iter = 0
        self.curr_iter = 0
        self.msg = None
        self.loss = None

        self.train_params = []
        self.validate_results = []
        self.optimizer = None
        self.scheduler = None

        self.create_models()
        self.init_models()
        self.create_train_models()

        self.create_criterion()
        self.create_optimizer_scheduler()
        # self.create_logger()

    def save(self):
        save_path = os.path.join(self.train_url,'ckpts',f'{self.epoch}.pth')
        state_dict = {}
        # save submodels
        for name in self.sub_models.keys():
            if isinstance(self.sub_models[name], nn.DataParallel) or isinstance(
                        self.sub_models[name], nn.parallel.DistributedDataParallel):
                state_dict[name] = self.sub_models[name].module.state_dict()
            else:
                state_dict[name] = self.sub_models[name].state_dict()

        # save state
        state_dict['epoch'] = self.epoch
        state_dict['iter'] = self.iter
        state_dict['optim'] = self.optimizer.state_dict()
        state_dict['scheduler'] = self.scheduler.state_dict()
        torch.save(state_dict, save_path)
        self.logger.info('Checkpoint saved to [{}]'.format(save_path))

    def resume(self):
        if self.resume_opt.path:
            if  isinstance(self.resume_opt.path,dict):
                path_dict = self.resume_opt.path
                for key in path_dict.keys():
                    sub_model_name = key
                    resume_model_name = path_dict[key][0]
                    resume_model_path = path_dict[key][1]
                    self.logger.info("Loading Checkpoint from [{}] ...".format(resume_model_path))
                    ckpt = torch.load(resume_model_path,map_location='cpu')
                    self.logger.info("Resume Submodel [{}] from Checkpoint".format(sub_model_name))
                    if resume_model_name == 'None':
                        self.sub_models[sub_model_name].load_state_dict(ckpt,
                                                             strict = self.resume_opt.strict)
                    else:
                        self.sub_models[sub_model_name].load_state_dict(ckpt[resume_model_name],
                                                             strict = self.resume_opt.strict)
            else:
                self.logger.info("Loading Checkpoint from [{}] ...".format(self.resume_opt.path))
                ckpt = torch.load(self.resume_opt.path,map_location='cpu')
                # resume submodels
                for name in self.sub_models.keys():
                    if name in ckpt:
                        self.logger.info("Resume Submodel from Checkpoint: [{}]".format(name))
                        # if DP or DDP
                        if isinstance(self.sub_models[name], nn.DataParallel) or isinstance(
                                        self.sub_models[name], nn.parallel.DistributedDataParallel):
                            self.sub_models[name].module.load_state_dict(ckpt[name],
                                                    strict = self.resume_opt.strict)
                        else:
                            self.sub_models[name].load_state_dict(ckpt[name],
                                                    strict = self.resume_opt.strict)
                    else:
                        self.logger.info("Not find Submodel: [{}]".format(name))
            # resume state'
            if self.resume_opt.resume_optim:
                self.logger.info("Resume State from Checkpoint ...")
                self.epoch = ckpt['epoch']
                self.iter = ckpt['iter']
                
                self.optimizer.load_state_dict(ckpt['optim'])
                self.scheduler.load_state_dict(ckpt['scheduler'])

                # optim parms to gpu
                for state in self.optimizer.state.values():
                    for k,v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(self.device)


    def train_update(self,epoch,curr_iter):
        self.curr_iter = curr_iter
        self.epoch = epoch

        if self.iter%self.train_opt.print_freq ==0:
            self.logging()
       
        self.iter+=1
    

    def create_models(self):
        pass
    def init_models(self):
        pass
    
    def create_train_models(self):
        if isinstance(self.train_opt.optim_models,list):
            for name in self.sub_models.keys():
                if name in self.train_opt.optim_models:
                    self.train_sub_models.update({
                        name: self.sub_models[name]
                    })
        else:
            for name in self.sub_models.keys():
                self.train_sub_models.update({
                    name: self.sub_models[name]
                })
        print('[train model]:',self.train_sub_models.keys())

    def create_optimizer_scheduler(self):
        for name in self.train_sub_models.keys():
            if isinstance(self.train_sub_models[name],dict):
                model = self.train_sub_models[name]['model']
                lr = self.train_sub_models[name]['lr'] * self.optim_opt.lr
                self.train_params.append({'params':model.parameters, 'lr':lr})
            else:
                self.train_params.append({'params':self.train_sub_models[name].parameters()})

        self.optimizer = optim.Adam(self.train_params, 
                                    lr =self.optim_opt.lr,
                                    weight_decay=self.optim_opt.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones = self.sched_opt.milestones,
                                                        gamma= self.sched_opt.gamma)
    
    def move_to_device(self):
        if len(self.gpu.split(','))==1: # single gpu
            for k in self.sub_models.keys():
                self.sub_models[k] = self.sub_models[k].to(self.device)
        elif len(self.gpu.split(','))>1:
            device_ids = [i for i in range(len(self.gpu.split(',')))]
            for k in self.sub_models.keys():
                self.sub_models[k] = nn.DataParallel(self.sub_models[k],device_ids=device_ids)

    def create_criterion(self):
        pass
    def feed_pretrain_data(self):
        pass

    def pretrain_forward(self):
        pass
    
    def pretrain_cal_loss(self):
        pass
    
    def feed_train_data(self,train_data):
        if isinstance(train_data,dict):
            self.data_name = train_data['name']
            self.data = train_data['data'].to(self.device,non_blocking=True)
        if isinstance(train_data,tuple) or isinstance(train_data,list):
            self.data_name,self.data = train_data
            if isinstance(self.data, dict):
                for k,v in self.data.items():
                    self.data[k] = v.to(self.device,non_blocking=True)
        else:
            self.data = train_data.to(self.device,non_blocking=True)
            
    def train_forward(self):
        self.train_init()
        self.train_main_process()
        
    def train_init(self):
        pass

    def train_main_process(self):
        pass

    def get_msg():
        pass
    def train_logging(self):
        self.get_msg()
        self.logger.info(self.msg)
            
    def feed_validate_data(self,val_data):
        if isinstance(val_data,dict):
            self.data_name = val_data['name']
            self.data = val_data['data'].to(self.device,non_blocking=True)
        if isinstance(val_data,tuple) or isinstance(val_data,list):
            self.data_name,self.data = val_data
            if isinstance(self.data, dict):
                for k,v in self.data.items():
                    self.data[k] = v.to(self.device,non_blocking=True)
            else:
                self.data= self.data.to(self.device,non_blocking=True)
        else:
            self.data_name = 'None'
            self.data = val_data.to(self.device,non_blocking=True)

    def validate_init(self):
        pass
    def validate_main_process(self):
        pass
    def validate_process_images(self):
        pass
    def validate_forward(self):
        self.validate_init()
        self.validate_main_process()
        self.validate_process_images()

    def validate_cal_metrics(self):
        pass

    def train_cal_loss(self):
        pass

    def optimize_params(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        # clip grad 
        ## To do later
        self.optimizer.step()
    
    # def create_logger(self):
    #     pass
    
    def train_epoch_init(self,train_logger,epoch):
        self.set_logger(train_logger)
        self.set_epoch(epoch)
        self.set_curr_iter()
        self.set_submodels_train() # submodel.train()
        current_lr = []
        for param_group in self.optimizer.param_groups:
            current_lr.append(param_group['lr'])
        self.logger.info("LearningRate of Epoch {} : {}".format(epoch, current_lr))


    def set_logger(self,curr_logger):
        self.logger = curr_logger

    def set_iter(self,iter=0):
        self.iter = iter

    def set_curr_iter(self,curr_iter=0):
        self.curr_iter = curr_iter

    def set_submodels_train(self):
        for k,v in self.sub_models.items():
            self.sub_models[k] = v.train()

    def set_submodels_eval(self):
        for k,v in self.sub_models.items():
            self.sub_models[k] = v.eval()

    def set_epoch(self,epoch=0):
        self.epoch = epoch

    def update_iter(self):
        self.curr_iter +=1
        self.iter+=1

    def scheduler_step(self):
        self.scheduler.step()

    def print_submodels(self,logger):
        self.logger = logger
        logger.info('Show Sub-models:')
        logger.info(pprint.pformat([k for k in self.sub_models.keys()], sort_dicts=False))

        logger.info('Show Trainable Sub-models:')
        logger.info(pprint.pformat([k for k in self.train_sub_models.keys()], sort_dicts=False))