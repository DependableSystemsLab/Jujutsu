"""
perform attack detection by comparing the prediciton labels on the source image and 2 hold-out images transplanted with salient features from the source img

"""

import warnings
import os
import numpy as np
  
from torchvision import models
import numpy as np
import cv2
import torch.nn as nn

#import imageio 
#from PIL import Image
import torch
import sys 
import argparse 
import re



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--is_adv", type = int)
ap.add_argument("--num_input", type = int)
ap.add_argument("--source_folder", type=str, help="folder containing the original test images")  
ap.add_argument("--img_folder", type=str, help="first folder containing the hold-out images transplanted with salient features") 
ap.add_argument("--second_folder", type=str, help="second folder containing the hold-out images transplanted with salient features") 
ap.add_argument("--noise_percentage", type=str, help='size of the patch size. This is only for labeling the output folder')  
ap.add_argument("-r", "--radius", default=51, type = int, required=True, help = "for labeling the output folder only")
ap.add_argument('--GPU', type=str, default='0', help="index pf used GPU")
ap.add_argument('--model_path', type=str, default='./celeb-model/facial_identity_ResNet18.pth')
ap.add_argument('--target', type=int, default=80)
ap.add_argument('--patch_type', type=str, default='square', help="patch type: rectangle or square")

ap.add_argument('--save_folder', type=str, default='.')



args = vars(ap.parse_args()) 

os.environ["CUDA_VISIBLE_DEVICES"] = args['GPU']

if(args['is_adv']):
    filter_folder = "{}/celeb_{}_{}_r{}_adv_test_detected_{}".format( args['save_folder'].rstrip('/'),  str(args['target']), args['patch_type'],  args['radius'], args['noise_percentage'].replace('.', '')) 
    if(not os.path.exists(filter_folder)):
        os.mkdir(filter_folder)
elif(not args['is_adv']):
    filter_folder = "{}/celeb_{}_{}_r{}_org_test_misdetected_{}".format( args['save_folder'].rstrip('/'),  str(args['target']), args['patch_type'], args['radius'], args['noise_percentage'].replace('.', '')) 
    if(not os.path.exists(filter_folder)):
        os.mkdir(filter_folder)     


def main() -> None:
    # instantiate a model (could also be a TensorFlow or JAX model)
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 307) # multi-class classification (num_of_class == 307)
    model.load_state_dict(torch.load(args['model_path'])) 
    model.eval() 
    model.cuda()


    files = []    
    for r, d, f in os.walk(args['img_folder']):
        for file in f: 
            if(".npy" in file):
                files.append(os.path.join(r, file))

 
    input_size = 1
    if(args['num_input']==0):
        input_size = len(files) # eval all the images in folder


    total = input_size
    unchanged = 0.
    change_for_benign_input = 0.

    for i in range(input_size):


        img1 = torch.from_numpy(np.load(files[i])).type(torch.FloatTensor).cuda()
        img2 = torch.from_numpy( np.load( files[i].replace(args['img_folder'], args['second_folder']))).type(torch.FloatTensor).cuda()
        org_img = torch.from_numpy(np.load(files[i].replace(args['img_folder'], args['source_folder']))).type(torch.FloatTensor).cuda()
 
 
        out1 = model(img1)#.argmax(axis=-1)
        out2 = model(img2)
        out_org = model(org_img)
        #predictions = predictions.cpu().detach().numpy() 

        _, index1 = torch.max(out1.data, 1) 
        _, index2 = torch.max(out2.data, 1) 
        _, org_index = torch.max(out_org.data, 1) 
   
 
        # detection based on prediction consistency
        if( args['is_adv'] ):
            if(index1[0] == org_index[0] and index2[0] == org_index[0]):
                unchanged += 1
                #print( '{} out of {} adv inputs maintain the adv label: {}'.format(unchanged, (i+1), unchanged/(i+1) ) )

                #print( files[i], '===' )
                source_file = args['source_folder'] + files[i].replace(args['img_folder'], '')

                filtered_path = filter_folder + files[i].replace(args['img_folder'], '')
                os.system( "cp {} {}".format(source_file, filtered_path) ) 
                #print( "cp {} {}".format(source_file, filtered_path) )

            #else:
            #    print("\t\tMiss an adv sample", org_index[0], index1[0], index2[0] )

        else:

            if(index1[0] != org_index[0] or index2[0] != org_index[0] ):
                # this is a benign input
                change_for_benign_input += 1
                #print( '{} out of {} org inputs change the label: {}'.format(change_for_benign_input, (i+1), change_for_benign_input/(i+1) ) )
            else:
                #print("\t\t Mis-detect a benign input")
                source_file = args['source_folder'] + files[i].replace(args['img_folder'], '')
                filtered_path = filter_folder + files[i].replace(args['img_folder'], '')
                os.system( "cp {} {}".format(source_file, filtered_path) ) 
        

 
    if( args['is_adv'] ):
        print( '1) Detection success recall: {} out of {} adv inputs maintain the adv label: recall = {}'.format(unchanged, (i+1), unchanged/(i+1) ) )
    else:
        print( '2) False positive: {} out of {} org inputs change the label: FP = {}'.format(change_for_benign_input, (i+1), 1-change_for_benign_input/(i+1) ) )

    #print( args['source_folder'] )

if __name__ == "__main__":
    main()



