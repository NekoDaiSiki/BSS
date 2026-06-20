import torch
import numpy as np
from skimage import measure

import cripser
import cripser as crip
import tcripser as trip

# 要先pip install cripser

# 这里整理一下我这边的问题
# 1. crip_wrapper, trip_wrapper, get_differentiable_barcode, 以及 measure.label 的调用函数都已经添加。
# 2. pred的shape [H,W]  target 也是 [H,W],它包含多类别的时候，用0,1,2表示。
# 3. bcode_arr 的size 自己会出来不用管。
# 4. 损失函数是哪个？没有看到对target编码的过程？ topoloss = A +Z + mse


def crip_wrapper(X, D):
    return crip.computePH(X, maxdim=D)

def trip_wrapper(X, D):
    return trip.computePH(X, maxdim=D)

def get_roi(X, thresh=0.01):
    true_points = torch.nonzero(X >= thresh)
    corner1 = true_points.min(dim=0)[0]
    corner2 = true_points.max(dim=0)[0]
    roi = [slice(None, None)] + [slice(c1, c2 + 1) for c1, c2 in zip(corner1, corner2)]
    return roi

def get_differentiable_barcode(tensor:torch.Tensor, barcode):
    '''Makes the barcode returned by CubicalRipser differentiable using PyTorch.
    Note that the critical points of the CubicalRipser filtration reveal changes in sub-level set topology.
    
    Arguments:
        REQUIRED
        tensor  - PyTorch tensor w.r.t. which the barcode must be differentiable
        barcode - Barcode returned by using CubicalRipser to compute the PH of tensor.numpy() 
    '''
    # Identify connected component of ininite persistence (the essential feature)
    inf = barcode[barcode[:, 2] == np.finfo(barcode.dtype).max] # 筛选出生命周期最长的一个
    fin = barcode[barcode[:, 2] < np.finfo(barcode.dtype).max]
    
    # Get birth of infinite feature
    inf_birth = tensor[tuple(inf[:, 3:3+tensor.ndim].astype(np.int64).T)]
    
    # Calculate lifetimes of finite features
    births = tensor[tuple(fin[:, 3:3+tensor.ndim].astype(np.int64).T)]
    deaths = tensor[tuple(fin[:, 6:6+tensor.ndim].astype(np.int64).T)]
    delta_p = (deaths - births) # 每一个bar的生命周期
    
    # Split finite features by dimension
    delta_p = [delta_p[fin[:, 0] == d] for d in range(tensor.ndim)]
    
    # Sort finite features by persistence
    delta_p = [torch.sort(d, descending=True)[0] for d in delta_p]
    
    return inf_birth, delta_p


def topo_cal_tc(target, pred, device, parallel = False, construction='0'):
    '''
    target: numpy.array, [H,W], label = 0,1,2
    pred: tensor, [3,H,W]
    '''
    inst_gt = measure.label(target == 1) # 按照连通度标记，每一个洞给一个数字
    bound_gt = measure.label(target == 2) # 每一个边给一个数字
    inst_number = np.max(inst_gt) # 获得洞的个数
    bound_number = np.max(bound_gt) # 获取边的个数



    # Set prior: class 1 is inside; 2 is boundary
    # 第一个数是连通的个数 第二个是洞的个数
    prior = {
        (0,):   (inst_number, 0),
        (1,):   (bound_number, inst_number),
        (0, 1): (bound_number, 0),
    }
    #print('prior = {}'.format(prior))

    # Inspect prior and convert to tensor
    max_dims = [len(b) for b in prior.values()] # [2,2,2]
    prior = {torch.tensor(c): torch.tensor(b) for c, b in prior.items()}
    
    # Set mode of cubical complex construction
    PH = {'0': crip_wrapper, 'N': trip_wrapper}

    outputs = pred #after softmax pred # torch.softmax(pred, 1).squeeze()
    spatial_xyz = list(pred.shape[1:])
    # Build class/combination-wise (c-wise) image tensor for prior
    combos = torch.stack([outputs[c.T].sum(0) for c in prior.keys()]) #(3,H,W)
    # Invert probababilistic fields for consistency with cripser sub-level set persistence
    combos = 1 - combos

    

    # Get barcodes using cripser in parallel without autograd            
    combos_arr = combos.detach().cpu().numpy().astype(np.float64)


    
    if parallel:
        with torch.no_grad():
            with Pool(len(prior)) as p:
                bcodes_arr = p.starmap(PH[construction], zip(combos_arr, max_dims))
    else:
        with torch.no_grad():
            bcodes_arr = [PH[construction](combo, max_dim) for combo, max_dim in zip(combos_arr, max_dims)]


    # Get differentiable barcodes using autograd
    max_features = max([bcode_arr.shape[0] for bcode_arr in bcodes_arr])
    # 预测图个数x2(联通和洞)xbar的个数， 每个值代表着对应位置的生命周期， 例如 inst中联通度中底几个生命周期，有多长
    bcodes = torch.zeros([len(prior), max(max_dims), max_features], requires_grad=False, device=device)

    for c, (combo, bcode) in enumerate(zip(combos, bcodes_arr)):
        _, fin = get_differentiable_barcode(combo, bcode)
        for dim in range(len(spatial_xyz)):
            bcodes[c, dim, :len(fin[dim])] = fin[dim]

    # Select features for the construction of the topological loss
    stacked_prior = torch.stack(list(prior.values()))
    stacked_prior.T[0] -= 1 # Since fundamental 0D component has infinite persistence
    matching = torch.zeros_like(bcodes).detach().bool()
    for c, combo in enumerate(stacked_prior):
        for dim in range(len(combo)):
            matching[c, dim, slice(0, stacked_prior[c, dim])] = True

    # Find total persistence of features which match (A) / violate (Z) the prior
    A = (1 - bcodes[matching]).sum()
    Z = bcodes[~matching].sum()
    mse = 0	# F.mse_loss(outputs, pred_unet)	# pred_unet is [old prediction]
	
    topoloss = A + Z + mse

    return topoloss

if __name__ == '__main__':
    target = np.array([[2,2,2],[2,1,2,],[2,2,2]])
    pred = torch.tensor([])
