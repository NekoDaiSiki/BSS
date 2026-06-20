import random
import torch
import torch.nn.functional as F
import numpy as np

from src.utils.register import Registry
from src.utils.builder import MASKER

def interpolate_mask(tensor, mask, mask_inv,inter_kernel=3):
    n, c, h, w = tensor.shape
    device = tensor.device
    mask = mask.to(device)
    if inter_kernel ==3:
        kernel = np.array([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)])
    elif inter_kernel==5:
        kernel = np.array([[0.1,0.2,0.3,0.2,0.1],
                           [0.2,0.5,1.0,0.5,0.2],
                           [0.5,1.0,0.0,1.0,0.5],
                           [0.2,0.5,1.0,0.5,0.2],
                           [0.1,0.2,0.3,0.2,0.1]])
    kernel = kernel[np.newaxis, np.newaxis, :, :]
    kernel = torch.Tensor(kernel).to(device)
    kernel = kernel / kernel.sum()

    filtered_tensor = torch.nn.functional.conv2d(
        tensor.view(n*c, 1, h, w), kernel, stride=1, padding=inter_kernel//2)

    return filtered_tensor.view_as(tensor) * mask + tensor * mask_inv

def interpolate_mask_pro(tensor, mask, mask_inv,preset=True,kernel_size=3,attention=False):
    n, c, h, w = tensor.shape
    device = tensor.device
    mask = mask.to(device)

    if not preset:
        kernel = torch.ones((1,1,kernel_size,kernel_size),device=device)
        kernel[:,:,kernel//2,kernel//2]=0.0
    else:
        if kernel_size==3:
            kernel = torch.tensor([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)],device=device).unsqueeze(0).unsqueeze(0)
        elif kernel_size ==5:
            kernel = torch.tensor([[0.1,0.2,0.3,0.2,0.1],
                                    [0.2,0.5,1.0,0.5,0.2],
                                    [0.5,1.0,0.0,1.0,0.5],
                                    [0.2,0.5,1.0,0.5,0.2],
                                    [0.1,0.2,0.3,0.2,0.1]],device=device).unsqueeze(0).unsqueeze(0)
    if not attention:
        kernel = kernel / kernel.sum()
        filtered_tensor = F.conv2d(
            tensor.view(n*c, 1, h, w), kernel, stride=1, padding=kernel_size//2)
        result = filtered_tensor.view_as(tensor) * mask + tensor * mask_inv
    else:   
        tensor = tensor.view(n*c, 1, h, w)
        slide_windows = F.unfold(tensor,kernel_size=kernel_size,padding=kernel_size//2,stride=1).permute(0,2,1) # -> (n*c,h*w,kernel*kernel)
        n_,s_,k_ = slide_windows.shape
        center = slide_windows[:,:,kernel_size**2//2].unsqueeze(2) #(n*c,h*w,1)
        attention = (center-slide_windows).abs()
        attention = 1-torch.softmax(attention,dim=2)
        kernel = kernel.view(1,1,kernel_size**2)
        kernel = kernel*attention
        kernel[:,:,kernel_size**2//2]=0
        kernel = kernel / kernel.sum(2).unsqueeze(2)

        result = (slide_windows * kernel).sum(2).view(n,c,h,w)
        # print(result.shape,mask.shape)
        result = result* mask + tensor.view(n,c,h,w) * mask_inv

    return result


def depth_to_space(x, block_size):
    return torch.nn.functional.pixel_shuffle(x, block_size)


@MASKER.register_module()
class b2ub_Masker(object):
    def __init__(self, width=4, mode='interpolate', mask_type='all',inter_kernel=3):
        self.width = width
        self.mode = mode
        self.mask_type = mask_type
        self.inter_kernel = inter_kernel

    def mask(self, img, mask_type=None, mode=None):
        # This function generates masked images given random masks
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        n, c, h, w = img.shape
        mask = self.generate_mask(img, width=self.width, mask_type=mask_type)
        mask_inv = torch.ones(mask.shape,device=img.device) - mask
        if mode == 'interpolate':
            masked = interpolate_mask(img, mask, mask_inv,self.inter_kernel)
        else:
            raise NotImplementedError

        net_input = masked
        return net_input, mask

    def add_mask(self, img):
        n, c, h, w = img.shape
        tensors = torch.zeros((n, self.width**2, c, h, w), device=img.device)
        masks = torch.zeros((n, self.width**2, 1, h, w), device=img.device)
        for i in range(self.width**2):
            x, mask = self.mask(img, mask_type='fix_{}'.format(i))
            tensors[:, i, ...] = x
            masks[:, i, ...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks

    def generate_mask(self,img, width=4, mask_type='random'):
        # This function generates random masks with shape (N x C x H/2 x W/2)
        n, c, h, w = img.shape
        mask = torch.zeros(size=(n * h // width * w // width * width**2, ),
                        dtype=torch.int64,
                        device=img.device)
        idx_list = torch.arange(
            0, width**2, 1, dtype=torch.int64, device=img.device)
        rd_idx = torch.zeros(size=(n * h // width * w // width, ),
                            dtype=torch.int64,
                            device=img.device)

        if mask_type == 'random':
            torch.randint(low=0,
                        high=len(idx_list),
                        size=(n * h // width * w // width, ),
                        device=img.device,
                        out=rd_idx)
        elif mask_type == 'batch':
            rd_idx = torch.randint(low=0,
                                high=len(idx_list),
                                size=(n, ),
                                device=img.device).repeat(h // width * w // width)
        elif mask_type == 'all':
            rd_idx = torch.randint(low=0,
                                high=len(idx_list),
                                size=(1, ),
                                device=img.device).repeat(n * h // width * w // width)
        elif 'fix' in mask_type:
            index = mask_type.split('_')[-1]
            index = torch.from_numpy(np.array(index).astype(
                np.int64)).type(torch.int64)
            rd_idx = index.repeat(n * h // width * w // width).to(img.device)

        rd_pair_idx = idx_list[rd_idx]
        rd_pair_idx += torch.arange(start=0,
                                    end=n * h // width * w // width * width**2,
                                    step=width**2,
                                    dtype=torch.int64,
                                    device=img.device)

        mask[rd_pair_idx] = 1

        mask = depth_to_space(mask.type_as(img).view(
            n, h // width, w // width, width**2).permute(0, 3, 1, 2), block_size=width).type(torch.int64)

        return mask

@MASKER.register_module()
class b2ub_Masker_pro(b2ub_Masker):
    def __init__(self, width=4, mode='interpolate', mask_type='all', kernel_size=3, preset=True, attention=True):
        super().__init__(width, mode, mask_type, kernel_size)
        self.preset = preset
        self.attention = attention
    def mask(self, img, mask_type=None, mode=None):
        # This function generates masked images given random masks
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        n, c, h, w = img.shape
        mask = self.generate_mask(img, width=self.width, mask_type=mask_type)
        mask_inv = torch.ones(mask.shape,device=img.device) - mask
        if mode == 'interpolate':
            masked = interpolate_mask_pro(img, mask, mask_inv,self.preset,self.inter_kernel,self.attention)
        else:
            raise NotImplementedError

        net_input = masked
        return net_input, mask


@MASKER.register_module()
class b2ub_Msker_learn_mask(b2ub_Masker):
    def __init__(self, width=4, mode='interpolate', mask_type='all', inter_kernel=3, c=0.1,trymode=0):
        super().__init__(width, mode, mask_type, inter_kernel)
        self.c = c
        self.trymode = trymode
    def add_mask(self, img, padding):
        n, c, h, w = img.shape
        self.padding = padding
        tensors = torch.zeros((n, self.width**2, c, h, w), device=img.device)
        masks = torch.zeros((n, self.width**2, 1, h, w), device=img.device)
        for i in range(self.width**2):
            x, mask = self.mask(img, mask_type='fix_{}'.format(i))
            tensors[:, i, ...] = x
            masks[:, i, ...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks
    def mask(self, img, mask_type=None, mode=None):
        # This function generates masked images given random masks
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        n, c, h, w = img.shape
        mask = self.generate_mask(img, width=self.width, mask_type=mask_type)
        mask_inv = torch.ones(mask.shape,device=img.device) - mask
        if mode == 'interpolate':
            masked = self.interpolate_mask(img, mask, mask_inv)
        else:
            raise NotImplementedError
        net_input = masked
        return net_input, mask 
    def interpolate_mask(self,tensor, mask, mask_inv):
        filtered_tensor = self.padding
        return filtered_tensor * mask + tensor * mask_inv
    
@MASKER.register_module()
class b2ub_Masker_pro2(b2ub_Masker):
    def __init__(self, width=4, mode='interpolate', mask_type='all', kernel_size=3, preset=True, attention=True):
        super().__init__(width, mode, mask_type, kernel_size)
        self.preset = preset
        self.attention = attention
    def mask(self, img, mask_type=None, mode=None):
        # This function generates masked images given random masks
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        n, c, h, w = img.shape
        mask = self.generate_mask(img, width=self.width, mask_type=mask_type)
        mask_inv = torch.ones(mask.shape,device=img.device) - mask
        if mode == 'interpolate':
            masked = self.interpolate_mask_pro(img, mask, mask_inv,self.preset,self.inter_kernel,self.attention)
        else:
            raise NotImplementedError

        net_input = masked
        return net_input, mask   
    
    def interpolate_mask_pro(self,tensor, mask, mask_inv,preset=True,kernel_size=3,attention=False):
        n, c, h, w = tensor.shape
        device = tensor.device
        mask = mask.to(device)

        if not preset:
            kernel = torch.ones((1,1,kernel_size,kernel_size),device=device)
            kernel[:,:,kernel//2,kernel//2]=0.0
        else:
            if kernel_size==3:
                kernel = torch.tensor([[0.5, 1.0, 0.5], [1.0, 0.0, 1.0], (0.5, 1.0, 0.5)],device=device).unsqueeze(0).unsqueeze(0)
            elif kernel_size ==5:
                kernel = torch.tensor([[0.1,0.2,0.3,0.2,0.1],
                                        [0.2,0.5,1.0,0.5,0.2],
                                        [0.5,1.0,0.0,1.0,0.5],
                                        [0.2,0.5,1.0,0.5,0.2],
                                        [0.1,0.2,0.3,0.2,0.1]],device=device).unsqueeze(0).unsqueeze(0)
        if not attention:
            kernel = kernel / kernel.sum()
            filtered_tensor = F.conv2d(
                tensor.view(n*c, 1, h, w), kernel, stride=1, padding=kernel_size//2)
            result = filtered_tensor.view_as(tensor) * mask + tensor * mask_inv
        else:   
            tensor = tensor.view(n*c, 1, h, w)
            tensor_slide_windows = F.unfold(tensor, kernel_size=kernel_size,padding=kernel_size//2,stride=1).permute(0,2,1) # -> (n*c,h*w,kernel*kernel)
            slide_windows = F.unfold(self.ref.view(n*c, 1, h, w), kernel_size=kernel_size,padding=kernel_size//2,stride=1).permute(0,2,1) # -> (n*c,h*w,kernel*kernel)
            center = slide_windows[:,:,kernel_size**2//2].unsqueeze(2) #(n*c,h*w,1)
            distance = (center-slide_windows)**2
            distance[:,:,kernel_size**2//2]=distance.sum(2)/(kernel_size**2-1)
                        # attention = 1-torch.softmax(attention,dim=2)
            # attention = F.gumbel_softmax(-distance,hard=True,dim=2)
            # print(distance[0,0,:])
            # inx = torch.argmax(-distance,dim=2)
            # print(inx.shape)
            # print(inx[6,8192])
            # attention = F.one_hot(inx,num_classes=kernel_size**2)
            # print(attention[0,0,:])
            # kernel = kernel.view(1,1,kernel_size**2)
            # attention_offset = torch.ones_like(attention)
            # attention = attention_offset+attention
            # kernel = kernel *attention
            # kernel[:,:,kernel_size**2//2]=0
            # kernel = kernel / kernel.sum(2).unsqueeze(2)
            # # print(kernel[0,0,:])

            # result = (tensor_slide_windows * kernel).sum(2)
            # result = 5/255.0*torch.randn_like(result) + result
            # result = result.view(n,c,h,w)
            # print(result.shape,mask.shape)
            center = center.squeeze(2).view(n,c,h,w)
            result = center+ 20/255.0*torch.randn_like(center)
            result = result* mask + tensor.view(n,c,h,w) * mask_inv

        return result
    def add_mask(self, img, ref):
        n, c, h, w = img.shape
        self.ref = ref
        tensors = torch.zeros((n, self.width**2, c, h, w), device=img.device)
        masks = torch.zeros((n, self.width**2, 1, h, w), device=img.device)
        for i in range(self.width**2):
            x, mask = self.mask(img, mask_type='fix_{}'.format(i))
            tensors[:, i, ...] = x
            masks[:, i, ...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks

@MASKER.register_module()
class rand_b2ub_Masker(b2ub_Masker):
    def __init__(self, width=4,vol=16, mode='interpolate', mask_type='random'):
        super().__init__(width, mode, mask_type)
        self.vol = vol
        self.mask_type = mask_type
    def add_mask(self, img):
        n, c, h, w = img.shape
        tensors = torch.zeros((n, self.vol, c, h, w), device=img.device)
        masks = torch.zeros((n, self.vol, 1, h, w), device=img.device)
        for i in range(self.vol):
            mask_type = self.mask_type
            x, mask = self.mask(img, mask_type=mask_type)
            tensors[:, i, ...] = x
            masks[:, i, ...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks

    def generate_mask(self,img, width=4, mask_type='random'):
        n, c, h, w = img.shape
        mask = torch.zeros(size=(n * h // width * w // width * width**2, ),
                dtype=torch.int64,
                device=img.device)
        if mask_type == 'random':
            rd_idx = torch.randint(low=0,
                        high=width**2,
                        size=(n * h // width * w // width, ),
                        device=img.device)
        elif mask_type == 'randomfix':
            rd_idx = torch.randint(low=0,
                        high=width**2,
                        size=(1,),
                        device=img.device)
            rd_idx = rd_idx.repeat(n * h // width * w // width)
        
        elif mask_type == 'Nrandomfix':
            rd_idx = torch.randint(low=0,
                        high=width**2,
                        size=(n,1),
                        device=img.device)
            rd_idx = rd_idx.repeat(1,h // width * w // width)
            rd_idx = rd_idx.view(n*h // width * w // width)
        
        elif mask_type == 'RanFixPRan':
            p_width = width//2 # 4//2
            rd_idx = torch.randint(low=0,
                high=p_width**2,
                size=(n*h // p_width * w // p_width, ),
                device=img.device)    
            rd_idx += torch.arange(start=0,
                        end=n * h // p_width * w // p_width * p_width**2,
                        step=p_width**2,
                        dtype=torch.int64,
                        device=img.device)
            
            mask[rd_idx] = 1
            mask = depth_to_space(mask.type_as(img).view(
            n, h // p_width, w // p_width, p_width**2).permute(0, 3, 1, 2), block_size=p_width).type(torch.int64)

            Ph,Pw = h//2,w//2 
            Pmask = torch.zeros(size=(n * Ph * Pw,),
                    dtype=torch.float32,
                    device=img.device)            
            
            P_rd_idx = torch.randint(low=0,
                high=p_width**2,
                size=(1,),
                device=img.device)  
            P_rd_idx = P_rd_idx.repeat(n*Ph // p_width * Pw // p_width)
            P_rd_idx += torch.arange(start=0,
                                    end=n * Ph * Pw,
                                    step=p_width**2,
                                    dtype=torch.int64,
                                    device=img.device)            
            Pmask[P_rd_idx] = 1.0
            Pmask = depth_to_space(Pmask.type_as(img).view(
            n, Ph // p_width, Pw // p_width, p_width**2).permute(0, 3, 1, 2), block_size=p_width).type(torch.float32)
            Pmask = F.upsample_nearest(Pmask,scale_factor=2)
            # Pmask = F.interpolate(Pmask,scale_factor=2,mode='nearest')
            return mask*Pmask

        elif mask_type == 'NRanFixPRan':
            p_width = width//2 # 4//2
            rd_idx = torch.randint(low=0,
                high=p_width**2,
                size=(n*h // p_width * w // p_width, ),
                device=img.device)    
            rd_idx += torch.arange(start=0,
                        end=n * h // p_width * w // p_width * p_width**2,
                        step=p_width**2,
                        dtype=torch.int64,
                        device=img.device)
            
            mask[rd_idx] = 1
            mask = depth_to_space(mask.type_as(img).view(
            n, h // p_width, w // p_width, p_width**2).permute(0, 3, 1, 2), block_size=p_width).type(torch.int64)

            Ph,Pw = h//2,w//2 
            Pmask = torch.zeros(size=(n * Ph * Pw,),
                    dtype=torch.float32,
                    device=img.device)            
            
            P_rd_idx = torch.randint(low=0,
                high=p_width**2,
                size=(n,1),
                device=img.device)  
            P_rd_idx = P_rd_idx.repeat(1,Ph // p_width * Pw // p_width)
            P_rd_idx = P_rd_idx.view(n*Ph // p_width * Pw // p_width)
            P_rd_idx += torch.arange(start=0,
                                    end=n * Ph * Pw,
                                    step=p_width**2,
                                    dtype=torch.int64,
                                    device=img.device)            
            Pmask[P_rd_idx] = 1.0
            Pmask = depth_to_space(Pmask.type_as(img).view(
            n, Ph // p_width, Pw // p_width, p_width**2).permute(0, 3, 1, 2), block_size=p_width).type(torch.float32)
            Pmask = F.upsample_nearest(Pmask,scale_factor=2)
            return mask*Pmask



        rd_idx += torch.arange(start=0,
                                end=n * h // width * w // width * width**2,
                                step=width**2,
                                dtype=torch.int64,
                                device=img.device)
        
        
        mask[rd_idx] = 1

        mask = depth_to_space(mask.type_as(img).view(
            n, h // width, w // width, width**2).permute(0, 3, 1, 2), block_size=width).type(torch.int64)

        return mask      
     
@MASKER.register_module()
class randc_b2ub_Masker(rand_b2ub_Masker):
    def add_mask(self, img):
        n, c, h, w = img.shape
        tensors = torch.zeros((n, self.vol,c, c, h, w), device=img.device)
        masks = torch.zeros((n, self.vol,c, c, h, w), device=img.device) # 一个vol 有c个通道域的mask
        for i in range(self.vol):
            x, mask = self.mask(img, mask_type=self.mask_type)
            tensors[:, i, ...] = x
            masks[:, i, ...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, c, h, w)

        return tensors, masks    
    
    def mask(self, img, mask_type=None, mode=None):
        # This function generates masked images given random masks
        if mode is None:
            mode = self.mode
        if mask_type is None:
            mask_type = self.mask_type

        n, c, h, w = img.shape
        mask_c = torch.zeros((n,c, c, h, w), device=img.device)        
        maskd_c = torch.zeros((n,c, c, h, w), device=img.device)
        mask = self.generate_mask(img, width=self.width, mask_type=mask_type)
        for c_idx in range(c):
            mask_c[:,c_idx,c_idx,:,:] = mask.squeeze(1)
            mask_inv = torch.ones(mask_c[:,c_idx,...].shape,device=img.device) - mask_c[:,c_idx,...]
            if mode == 'interpolate':
                masked = interpolate_mask(img, mask_c[:,c_idx,...], mask_inv)
                maskd_c[:,c_idx,...] = masked
            else:
                raise NotImplementedError

        net_input = maskd_c
        return net_input, mask_c

@MASKER.register_module()
class blindspot_Masker(b2ub_Masker):
    def __init__(self, width=4, mode='interpolate', mask_type='all',volume=None):
        super().__init__(width, mode, mask_type)
        self.volume=volume
    def train(self, img):
        n, c, h, w = img.shape
        
        volume = self.volume if self.volume else self.width**2

        tensors = torch.zeros((n, volume, c, h, w), device=img.device)
        masks = torch.zeros((n, volume, 1, h, w), device=img.device)
        for i in range(volume):
            idx = random.randint(0,self.width**2-1)  if self.volume else i
            x, mask = self.mask(img, mask_type='fix_{}'.format(idx))
            tensors[:, i, ...] = x
            masks[:, i, ...] = mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, 1, h, w)
        return tensors, masks


@MASKER.register_module()
class n2v_Masker(object):
    def __init__(self,volume=1,ratio=0.1,window_size=(5,5), mode='copy_remote'):
        self.volume = volume # number of images in a volume
        self.mode = mode # masked mode
        self.ratio = ratio # masked pixels rate per image
        self.window_size = window_size

    def add_mask(self,img:torch.Tensor):
        n, c, h, w = img.shape
        v = self.volume
        masks =  torch.zeros((n,v,c,h,w),device=img.device)
        tensors = torch.zeros((n,v,c,h,w),device=img.device)
        for vol in range(self.volume):
            tensor,mask = self.add_one_mask(img,mode=self.mode)
            tensors[:,vol,...], masks[:,vol,...] = tensor, mask
        tensors = tensors.view(-1, c, h, w)
        masks = masks.view(-1, c, h, w)
        return tensors,masks
    
    def add_one_mask(self,img:torch.Tensor,mode='copy_remote'):
        n, c, h, w = img.shape
        num_sample = int(h*w*self.ratio)
        idy_msk = torch.randint(0,h,(num_sample,))
        idx_msk = torch.randint(0,w,(num_sample,))

        if mode=='copy_remote':
            # for c_idx in range(c):
            idy_neigh = torch.randint(-self.window_size[0] // 2 + self.window_size[0] % 2, self.window_size[0] // 2 + self.window_size[0] % 2, (num_sample,))
            idx_neigh = torch.randint(-self.window_size[1] // 2 + self.window_size[1] % 2, self.window_size[1] // 2 + self.window_size[1] % 2, (num_sample,))

            idy_msk_neigh = idy_msk + idy_neigh
            idx_msk_neigh = idx_msk + idx_neigh
            idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * h - (idy_msk_neigh >= h) * h
            idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * w - (idx_msk_neigh >= w) * w

            id_msk = [idy_msk.tolist(), idx_msk.tolist()]
            id_msk_neigh = [idy_msk_neigh.tolist(), idx_msk_neigh.tolist()]

            
            tensor = img.clone()
            tensor[:,:,torch.tensor(id_msk[0]),torch.tensor(id_msk[1])] = img[:,:,torch.tensor(id_msk_neigh[0]),torch.tensor(id_msk_neigh[1])]
            mask = torch.zeros_like(img,device=img.device)
            mask[:,:,torch.tensor(id_msk[0]),torch.tensor(id_msk[1])]=1.0
        else:
            raise NotImplementedError
        return tensor,mask
