"Save the image from the np format (normalized) into jpg format"
"NOTE that this saving might cause some loss and you can't save float array as image"

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
ap.add_argument('--inv_normalize', type=int, default=0) 

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


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)



for j in range(len(files)): 
 
    
    npy = np.load(files[j]) 
    if(args['inv_normalize']):
        npy = npy[0]

    print(j, npy.shape)
    npy = torch.from_numpy(npy)
    save_img_name = files[j].replace(args['img_folder'], '').replace('.npy' , '.jpg')
    save_img_name = save_img_name.lstrip('/' )
    img = npy
    if(args['inv_normalize']):
        for i in range(3):
            img[i, :, :] *= std[i] 
            img[i, :, :] += mean[i]

    #img = np.transpose(img, (1,2,0) )
 

    save_image( img, args['out_folder'] + "/" + save_img_name )


  






