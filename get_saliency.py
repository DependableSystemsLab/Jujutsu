# "generate saliency map for images"
 
 
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import re
torch.cuda.is_available()

import argparse
import os
import numpy as np
import random
import cv2
from tqdm import tqdm

from smooth_grad import SmoothGrad 

parser = argparse.ArgumentParser()


parser.add_argument('--model', type=str, required=True, help='model name')
parser.add_argument("--img_folder", required=True, type=str) 
parser.add_argument("--GPU", type=str, default='0') 
parser.add_argument("--output_folder", type=str, default="held_out_saliency")

"for smoothgrad"
parser.add_argument('--image', type=str, required=False)
parser.add_argument('--sigma', type=float, default=0.10)
parser.add_argument('--n_samples', type=int, default=50)
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--guided', action='store_true', default=False)

args = parser.parse_args()

CUDA = torch.cuda.is_available()

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU


seed = 1
np.random.seed(seed)
random.seed(seed) 
torch.manual_seed(seed)
 

# Load the model
if(args.model == "resnet50"):
    model = models.resnet50(pretrained=True).cuda() 
elif(args.model == "densenet"):
    model = models.densenet121(pretrained=True).cuda() 
elif(args.model == "vggnet"):
    model = models.vgg16_bn(pretrained=True).cuda()
elif(args.model == "squeezenet"):
    model = models.squeezenet1_0(pretrained=True).cuda() 
elif(args.model == "resnet152"):
    model = models.resnet152(pretrained=True).cuda()
model.eval()
 

# Load the datasets
files = []
for r, d, f in os.walk(args.img_folder):
    for file in f: 
        files.append(os.path.join(r, file))
print("total inputs: ",len(files))



# Setup the SmoothGrad
smooth_grad = SmoothGrad(model=model, cuda=args.cuda, sigma=args.sigma,
                         n_samples=args.n_samples, guided=args.guided)



save_folder = args.output_folder
if(not os.path.exists(save_folder)):
    os.mkdir(save_folder) 


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
for i in range(len(files)):
    smooth_grad.load_image(filename=files[i], transform=transform)
    prob, idx = smooth_grad.forward()
    smooth_grad.generate(filename= save_folder + "/" + files[i].replace(args.img_folder,'').replace('.jpg',''), idx=idx[0])  # "0" means only calculate the saliency for the top label

 






