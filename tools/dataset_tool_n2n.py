import os
from glob import glob
import numpy as np
from scipy.io import loadmat, savemat
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm


data_dir = "/data1/hyq/datasets/SIDD_Medium_Raw/Data"
path_all_noisy = glob(os.path.join(data_dir, '**/*NOISY*.MAT'), recursive=True) # GT or NOISY
# path_all_noisy = sorted(path_all_noisy)
print('Number of big images: {:d}'.format(len(path_all_noisy)))

# print(path_all_noisy)
index = [i//2*2+ (i%2+1)%2 for i in range(len(path_all_noisy))]
path_all_gt = []
[path_all_gt.append(path_all_noisy[i]) for i in index]
# path_all_gt = glob(os.path.join(data_dir, '**/*GT*.MAT'), recursive=True) # GT or NOISY
# path_all_gt = sorted(path_all_gt)




save_folder = "/data1/hyq/datasets/SIDD_Medium_Raw_noisy_sub512_n2n" # gt or noisy
if os.path.exists(save_folder):
    os.system("rm -r {}".format(save_folder))
os.makedirs(save_folder)   
os.makedirs(os.path.join(save_folder,'gt'))   
os.makedirs(os.path.join(save_folder,'ny'))   

crop_size = 512
step = 256

def pipline(ii):
    img_name, extension = os.path.splitext(os.path.basename(path_all_noisy[ii]))
    print(img_name)
    mat_ny = h5py.File(path_all_noisy[ii])
    mat_gt = h5py.File(path_all_gt[ii])
    # im = mat['x'].value
    im_ny = mat_ny['x']
    im_gt = mat_gt['x']
    h, w = im_ny.shape
    # prepare to crop
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > 0:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > 0:
        w_space = np.append(w_space, w - crop_size)
    # crop
    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img_ny = im_ny[x:x + crop_size, y:y + crop_size]
            cropped_img_gt = im_gt[x:x + crop_size, y:y + crop_size]
            cropped_img_ny = np.ascontiguousarray(cropped_img_ny)
            cropped_img_gt = np.ascontiguousarray(cropped_img_gt)

            save_path_ny = os.path.join(save_folder,'ny', "{}_s{:0>3d}{}".format(img_name, index, extension.lower()))
            savemat(save_path_ny, {"x": cropped_img_ny})

            save_path_gt = os.path.join(save_folder,'gt', "{}_s{:0>3d}{}".format(img_name, index, extension.lower()))
            savemat(save_path_gt, {"x": cropped_img_gt})

Parallel(n_jobs=10)(delayed(pipline)(i) for i in tqdm(range(len(path_all_noisy))))