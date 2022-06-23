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


ap = argparse.ArgumentParser()
ap.add_argument("--img_folder", type=str, required=True) 
ap.add_argument('--input_size', type=int, default = 1, help='size of imgs to be evaluated. Set to 0 if you want to evaluate all the imgs in the folder') 
ap.add_argument('--extract_label', type=int, default = 0, help='whether to extract the img label from the img file name') 
ap.add_argument('--normalize', type=int, default = 0, help='whether to normalize the img before inference') 
ap.add_argument('--model', type=str, default='resnet50', help='model name')
ap.add_argument('--GPU', type=str, default=0, help="index pf used GPU")
ap.add_argument('--input_tag', type=str, required=True, default='out_0', help='set ``out_0'' or ``mask'': ``out_0'' means comparing the org img with the inpainted img; ``mask'' means comparing the org img with the masked img without inpainting')

 

args = vars(ap.parse_args()) 

def main() -> None:

    normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
     
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
        )

    # instantiate a model (could also be a TensorFlow or JAX model)
 
    if(args['model'] == "resnet50"):
        model = models.resnet50(pretrained=True).cuda() 
    elif(args['model'] == "densenet"):
        model = models.densenet121(pretrained=True).cuda() 
    elif(args['model'] == "vggnet"):
        model = models.vgg16_bn(pretrained=True).cuda()
    elif(args['model'] == "squeezenet"):
        model = models.squeezenet1_0(pretrained=True).cuda() 
    elif(args['model'] == "resnet152"):
        model = models.resnet152(pretrained=True).cuda()
    model.eval()


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
    for i in range(input_size):
        if(not args['input_tag'] in files[i]):
            continue

        cnt += 1
        truth_img_file = files[i].replace(args['input_tag'] , 'truth') 
        truth_img = torch.from_numpy( np.load( truth_img_file ) )

        img = torch.from_numpy( np.load(files[i]) )

        #print( files[i], truth_img_file )
        if(args['normalize']):
            img = normalize_img(img)
            img = img.unsqueeze(0)

            truth_img = normalize_img(truth_img)
            truth_img = truth_img.unsqueeze(0)            
         
        img = img.cuda()
        truth_img = truth_img.cuda()

        predictions = model(img)#.argmax(axis=-1) 
        _, index = torch.max(predictions.data, 1) 

        truth_predictions = model(truth_img)#.argmax(axis=-1) 
        _, truth_index = torch.max(truth_predictions.data, 1) 

        
        if(args['extract_label']): 
            # this is for adv patch input
            # because the ground truth img is adv input, 
            # label needs to be extracted from the file name
            regexp = re.compile("_\d+" )
            only_file_name = files[i].replace(args['img_folder'], '')
            tmp = regexp.findall(only_file_name) 
            clean_label = tmp[0][1:]

            if( index[0] == int(clean_label) ):
                recover+=1
                #print('recover {} out of {}, {}'.format(recover, cnt, recover/cnt))
            elif(index[0] == truth_index[0] ):
                remain_targeted_label += 1
                remain_targeted_file.append( files[i] )
                #print( files[i] )
            #else:
            #    print("\t\tpredictions on {}: {}, label: {}".format(files[i], index, clean_label) )

        else:
            if( index[0] == truth_index[0] ):
                recover += 1
            #    print('recover {} out of {}, {}'.format(recover, cnt, recover/cnt))
            #else:
            #    print( "fail to recover: ", files[i] )

            #print("predictions on {}: {}".format(files[i], index) )
            #print("\t\tlogit values (high to low): ", predictions[0][ index ]) 


    #print(args['img_folder'])

    if(args['extract_label']):
        print("3) Recovered {} out of {}, Robust accuracy = {:.4f}".format(recover, cnt, recover/cnt))
        print("4) (Reduced) detection success recall: {} out of {} adv images that remain the targeted adv label after mitigation (the lower the better)".format(remain_targeted_label,cnt) )
    else:
        print("5) Reduced FP on benign samples: {} out of {} benign images that remain the same label before and after masking/inpainting (the higher the better)".format(recover, cnt))
    
if __name__ == "__main__":
    main()



