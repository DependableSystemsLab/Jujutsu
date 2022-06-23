# "This file is to perform attack mitigation "


#import torchvision.models as models 
#from foolbox import PyTorchModel, accuracy, samples, read_custom_inputs, read_inputs_from_folder
#from foolbox.attacks import LinfPGD
from torchvision import models
import numpy as np
import cv2
import imageio 
from PIL import Image
import torch
import sys 
import argparse
import os 
import re
import torchvision.transforms as transforms
import datetime

ap = argparse.ArgumentParser()
ap.add_argument("--img_folder", type=str, required=True) 
ap.add_argument('--input_size', type=int, default = 1, help='size of imgs to be evaluated. Set to 0 if you want to evaluate all the imgs in the folder') 
ap.add_argument('--extract_label', type=int, default = 0, help='whether to extract the img label from the img file name') 
ap.add_argument('--normalize', type=int, default = 0, help='whether to normalize the img before inference')  
ap.add_argument('--GPU', type=str, default=0, help="index pf used GPU")
ap.add_argument('--input_tag', type=str, required=True, default='out_0', help='set ``out_0'' or ``mask'': ``out_0'' means comparing the org img with the inpainted img; ``mask'' means comparing the org img with the masked img without inpainting')

ap.add_argument('--model_path', type=str, default='./place-model/resnet50_places365.pth.tar')

args = vars(ap.parse_args()) 

os.environ["CUDA_VISIBLE_DEVICES"] = str(args['GPU'])


def main() -> None:

    normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
     
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
        )

    # instantiate a model (could also be a TensorFlow or JAX model)
 
    # load the pre-trained weights
    arch = 'resnet50'
    model_file = args['model_path']

    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()


    files = []
    for r, d, f in os.walk(args['img_folder']):
        for file in f: 
            files.append(os.path.join(r, file))

    input_size = args['input_size']
    if(args['input_size']==0):
        input_size = len(files) # eval all the images in folder 

    recover = 0.
    total = input_size
    cnt = 0
    remain_targeted_label = 0
    remain_targeted_file = []


    start = datetime.datetime.now()



    for i in range(input_size): 

        cnt += 1

        img = torch.from_numpy( np.load(files[i]) )

        #print( files[i], truth_img_file )
        if(args['normalize']):
            img = normalize_img(img)
            img = img.unsqueeze(0)
          
         
        img = img.cuda() 

        predictions = model(img)#.argmax(axis=-1) 
        _, index = torch.max(predictions.data, 1) 


    end = datetime.datetime.now()
    duration = int((end- start).total_seconds())
    print(" %d sec for %d images "%(duration, (input_size)))

if __name__ == "__main__":
    main()



