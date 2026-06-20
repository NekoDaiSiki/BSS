import os
import torch

# transform ori model to fitting our framework
# model_path = '/home/hyq/workspace/AlphaSSDNModel/orib2ub'
model_path = '/home/hyq/workspace/AlphaSSDNBack/SA_pretrain'
for model_name in os.listdir(model_path):
    model_path_full = os.path.join(model_path,model_name)
    if os.path.isfile(model_path_full):
        model = torch.load(model_path_full)
        new_model = {'backbone': model}
        # new_model = torch.nn.Mod
        torch.save(new_model,os.path.join(model_path,'procd',model_name))
