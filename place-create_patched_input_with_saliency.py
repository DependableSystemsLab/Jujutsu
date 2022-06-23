#"This is to generate adv samples by applying the patch to the clean images, and also calculate its saliency map using smoothgrad"
#"The patched image is saved as npy format and saliency map as png"




# code from https://github.com/A-LinCui/Adversarial_Patch_Attack
# Adversarial Patch Attack
# Created by Junbo Zhao 2020/3/17

"""
Reference:
[1] Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""

from __future__ import print_function

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm

from torchvision import transforms as trn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
import torchvision.utils as vutils
import argparse
import csv
import os
import numpy as np
from torchvision.utils import save_image
import random
from patch_utils import*
from utils import*
from smooth_grad import SmoothGrad
from  torchvision import datasets, transforms
import torchvision.utils as vutils 

def gen_patched_img(patch_type, target, patch, test_loader, model, org_img_folder, adv_img_folder, org_saliency_folder, adv_saliency_folder,  smooth_grad, test_size):
    model.eval()

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
        )


    test_total, test_actual_total, test_success = 0, 0, 0
    cnt = 0
    for (image, label) in test_loader:

        if( cnt >= test_size ):
            break


        test_total += label.shape[0]
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'

        label = int( classes[ label.item() ] )


        if(args.GPU != "-1"):
            image = image.cuda()
            #label = label.cuda()
        
        


        output = model(image)
        _, predicted = torch.max(output.data, 1)
        if predicted[0] == label and predicted[0].data.cpu().numpy() != target:
            test_actual_total += 1
            applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 224, 224))
            applied_patch = torch.from_numpy(applied_patch)
            mask = torch.from_numpy(mask)
            perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            if(args.GPU != "-1"):
                perturbated_image = perturbated_image.cuda()
            output = model(perturbated_image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0].data.cpu().numpy() == target:
                test_success += 1

                #print(" ====== {} input ".format(test_success))
                # get saliency from org img
                smooth_grad.feed_image(image)
                prob, idx = smooth_grad.forward()
                smooth_grad.generate(filename= org_saliency_folder + "/" + str(test_success) +  "_" + str(label ), idx=idx[0])  # "0" means only calculate the saliency for the top label

                smooth_grad.feed_image(perturbated_image)
                prob, idx = smooth_grad.forward()
                smooth_grad.generate(filename= adv_saliency_folder + "/" + str(test_success) +  "_" + str(label) , idx=idx[0])  


                np.save( adv_img_folder + "/" + str(test_success) +  "_" + str(label  ) + ".npy" , perturbated_image.cpu().numpy() )
                np.save( org_img_folder + "/" + str(test_success) +  "_" + str(label ) + ".npy" , image.cpu().numpy() )

        cnt +=1 
    print("%d successful adv samples out of %d samples"%(test_success, test_actual_total))
    return test_success / test_actual_total



# python create_patched_input_with_saliency.py --noise_percentage 0.06 --GPU 0 --model vggnet --patch_file vggnet_best_org_patch_006.npy --target 



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--num_workers', type=int, default=2, help="num_workers")
parser.add_argument('--train_size', type=int, default=2000, help="number of training images")
parser.add_argument('--test_size', type=int, default=2000, help="number of test images")
parser.add_argument('--noise_percentage', type=float, required=True, help="percentage of the patch size compared with the image size")
parser.add_argument('--probability_threshold', type=float, default=0.5, help="minimum target probability")
parser.add_argument('--lr', type=float, default=1.0, help="learning rate")
parser.add_argument('--max_iteration', type=int, default=1000, help="max iteration")
parser.add_argument('--target', type=int, default=859, help="target label")
parser.add_argument('--epochs', type=int, default=20, help="total epoch")

parser.add_argument('--train_data_dir', type=str, default='place-model/place_dataset/train', help="dir of the dataset")
parser.add_argument('--test_data_dir', type=str, default='place-model/place_dataset/test', help="dir of the dataset")
parser.add_argument('--save_folder', type=str, default='./')
parser.add_argument('--model_path', type=str, default='./place-model/resnet50_places365.pth.tar')


parser.add_argument('--GPU', type=str, default='0', help="index pf used GPU")
parser.add_argument('--log_dir', type=str, default='patch_attack_log.csv', help='dir of the log')
parser.add_argument('--patch_file', type=str, required=True, help='file to the adv patch') 
parser.add_argument('--patch_type', type=str, default='square', help="patch type: rectangle or square")

"for smoothgrad"
parser.add_argument('--image', type=str, required=False)
parser.add_argument('--sigma', type=float, default=0.10)
parser.add_argument('--n_samples', type=int, default=50)
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--guided', action='store_true', default=False)

args = parser.parse_args()


'''
# Load the synset words
idx2cls = list()
with open('samples/synset_words.txt') as lines:
    for line in lines:
        line = line.strip().split(' ', 1)[1]
        line = line.split(', ', 1)[0].replace(' ', '_')
        idx2cls.append(line)
'''

os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

seed = 1
np.random.seed(seed)
random.seed(seed) 
torch.manual_seed(seed)



# Load the model
# load the pre-trained weights
arch = 'resnet50'
model_file = args.model_path

model = models.__dict__[arch](num_classes=365)
if(args.GPU != "-1"):
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
else:
    checkpoint = torch.load(model_file, map_location = torch.device('cpu'))

state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()
if(args.GPU != "-1"):
    model.cuda()

# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



# Load the datasets 
train_dataset = torchvision.datasets.ImageFolder(os.path.join(args.train_data_dir), centre_crop)
test_dataset = torchvision.datasets.ImageFolder(os.path.join(args.test_data_dir), centre_crop)
"Need to map the torch-created class index back to the original index in the dataset"
classes =  test_dataset.classes 


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=2)

print('Train dataset size:', len(train_dataset))
print('Test dataset size:', len(test_dataset))


# Test the accuracy of model on trainset and testset
#trainset_acc, test_acc = test_place(model, train_loader, classes), test_place(model, test_loader, classes)
#print('Accuracy of the model on clean trainset and testset is {:.3f}% and {:.3f}%'.format(100*trainset_acc, 100*test_acc))







 
# Initialize the patch
patch = np.load(args.patch_file)
print('The shape of the patch is', patch.shape)
 

# Setup the SmoothGrad
if(args.GPU != "-1"):
    smooth_grad = SmoothGrad(model=model, cuda=args.cuda, sigma=args.sigma,
                             n_samples=args.n_samples, guided=args.guided)
else:
    smooth_grad = SmoothGrad(model=model, cuda=False, sigma=args.sigma,
                             n_samples=args.n_samples, guided=args.guided)


test_success_rate = 0
train_success_rate = 0



save_folder2 = "{}/place_{}_{}_npy_test_org_result_{}".format(args.save_folder.rstrip("/"), args.patch_type, args.target, str(args.noise_percentage).replace('.', ''))
save_folder22 = "{}/place_{}_{}_npy_test_adv_result_{}".format(args.save_folder.rstrip("/"), args.patch_type, args.target, str(args.noise_percentage).replace('.', ''))
save_folder222 = "{}/place_{}_{}_npy_test_org_saliency_{}".format(args.save_folder.rstrip("/"), args.patch_type, args.target, str(args.noise_percentage).replace('.', ''))
save_folder2222 = "{}/place_{}_{}_npy_test_adv_saliency_{}".format(args.save_folder.rstrip("/"), args.patch_type, args.target, str(args.noise_percentage).replace('.', ''))


if(not os.path.exists(save_folder2)):
    os.mkdir(save_folder2)
if(not os.path.exists(save_folder22)):
    os.mkdir(save_folder22)
if(not os.path.exists(save_folder222)):
    os.mkdir(save_folder222)
if(not os.path.exists(save_folder2222)):
    os.mkdir(save_folder2222)

#train_success_rate = gen_patched_img(patch_type= args.patch_type, target= args.target, patch= patch, test_loader= train_loader, model= model, org_img_folder = save_folder1, adv_img_folder=save_folder11, org_saliency_folder=save_folder111, adv_saliency_folder=save_folder1111,  smooth_grad=smooth_grad)
#print("  Patch attack success rate on trainset using the patch: {:.3f}%".format( 100 * train_success_rate))


test_success_rate = gen_patched_img(patch_type= args.patch_type, target= args.target, patch= patch, test_loader= test_loader, model= model, org_img_folder = save_folder2, adv_img_folder=save_folder22, org_saliency_folder=save_folder222, adv_saliency_folder=save_folder2222, smooth_grad=smooth_grad, test_size= args.test_size)
print("  Patch attack success rate on testset: {:.3f}%".format( 100 * test_success_rate))  








