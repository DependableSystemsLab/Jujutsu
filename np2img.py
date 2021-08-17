#"This is to visualize the image saved in npy format"

#
#
# https://www.pyimagesearch.com/2014/09/29/finding-brightest-spot-image-using-python-opencv/
#

# import the necessary packages
import numpy as np
import argparse
import cv2
import os 
from  torchvision import datasets, transforms
from torchvision.utils import save_image
import torch

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("--img_folder", type=str, required=True) 
ap.add_argument('--out_folder', type=str, required=True)
ap.add_argument('--inv_normalize', type=int, default=1)

args = vars(ap.parse_args()) 



files = []
for r, d, f in os.walk(args['img_folder']):
    for file in f:
        #if( ".JPEG" in file ):
        files.append(os.path.join(r, file))
 
SAVE = args['img_folder'] 

if(not os.path.exists(args['out_folder'])):
    os.mkdir(args['out_folder'])

inv_normalize = transforms.Normalize(
mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
std=[1/0.229, 1/0.224, 1/0.255]
)

for j in range(len(files)): 
 
    npy = np.load(files[j]) 
    print(j, npy.shape)
    # the img directly from the torch model needs inv normalize, and it has 4 dim
    # 
    if(npy.ndim == 4): 
        npy = npy[0]

    npy = torch.from_numpy(npy)
    save_img_name = files[j].replace(args['img_folder'], '').replace('.npy' , '.jpg')
    save_img_name = save_img_name.lstrip('/' )

    if( args['inv_normalize'] ):
        img = inv_normalize(npy)
    else:
        img = npy
    save_image( img, args['out_folder'] + "/" + save_img_name )







