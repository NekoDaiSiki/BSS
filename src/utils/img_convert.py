import torch
import numpy as np

def tensor2image(tensor):
    img = tensor.permute(0, 2, 3, 1)
    img = img.cpu().detach().clamp(0, 1).numpy().squeeze(0)
    img = np.clip(img*255.0+0.5,0,255).astype(np.uint8)
    return img