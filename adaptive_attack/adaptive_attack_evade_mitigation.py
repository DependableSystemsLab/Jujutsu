'''
For adaptive attacker to make patch survive under random masking

'''

# Adversarial Patch Attack
# Created by Junbo Zhao 2020/3/17

"""
Reference:
[1] Tom B. Brown, Dandelion Mané, Aurko Roy, Martín Abadi, Justin Gilmer
    Adversarial Patch. arXiv:1712.09665
"""


import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models

import argparse
import csv
import os
import numpy as np
import random

import sys
sys.path.append('../')

from patch_utils import*
from utils import*
import cv2 
import random
from random import randint

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--num_workers', type=int, default=2, help="num_workers")
parser.add_argument('--train_size', type=int, default=2000, help="number of training images")
parser.add_argument('--test_size', type=int, default=2000, help="number of test images")
parser.add_argument('--noise_percentage', type=float, default=0.06, help="percentage of the patch size compared with the image size")
parser.add_argument('--probability_threshold', type=float, default=0.9, help="minimum target probability")
parser.add_argument('--lr', type=float, default=1.0, help="learning rate")
parser.add_argument('--max_iteration', type=int, default=1000, help="max iteration")
parser.add_argument('--target', type=int, default=859, help="target label")
parser.add_argument('--epochs', type=int, default=30, help="total epoch")
parser.add_argument('--data_dir', type=str, default='/local/zitaoc/imagenet-val/ILSVRC2012_img_val', help="dir of the dataset")
parser.add_argument('--patch_type', type=str, default='rectangle', help="type of the patch")
parser.add_argument('--GPU', type=str, default='0', help="index pf used GPU")
parser.add_argument('--log_dir', type=str, default='patch_attack_log.csv', help='dir of the log')
parser.add_argument('--model', type=str, required=True, help='model name')

"for smoothgrad"
parser.add_argument('--image', type=str, required=False)
parser.add_argument('--sigma', type=float, default=0.10)
parser.add_argument('--n_samples', type=int, default=50) 
parser.add_argument('--guided', action='store_true', default=False)


parser.add_argument('--mask_percentage', type=float, required=True, help='maksing percentage to evaluate the adaptive attack. E.g, 0.5 means we mask 50\% of the points within the adversarial patch')
args = parser.parse_args()




def gencoordinates(size, x_min, x_max, y_min, y_max):
    # random sampling within the box, size is the number of sampling points
    seen = set()

    x_set = []
    y_set = []

    x, y = randint(x_min, x_max), randint(y_min, y_max)
    x_set.append(x)
    y_set.append(y) 

    cnt = 1
    while (not size==cnt ):
        seen.add((x, y)) 
        x, y = randint(x_min, x_max), randint(y_min, y_max)
        while (x, y) in seen:
            x, y = randint(x_min, x_max), randint(y_min, y_max)
        x_set.append(x)
        y_set.append(y) 
        cnt += 1 

    return x_set, y_set




# Patch attack via optimization
# According to reference [1], one image is attacked each time
# Assert: applied patch should be a numpy
# Return the final perturbated picture and the applied patch. Their types are both numpy
def patch_attack(image, applied_patch, mask, input_index, epoch, target, probability_threshold, model, lr=1, max_iteration=100):
    model.eval()
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    target_probability, count = 0, 0
    perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
    while target_probability < probability_threshold and count < max_iteration:
        count += 1
        # Optimize the patch
        perturbated_image = Variable(perturbated_image.data, requires_grad=True)
        per_image = perturbated_image
        per_image = per_image.cuda()
        output = model(per_image)
        target_log_softmax = torch.nn.functional.log_softmax(output, dim=1)[0][target]
        target_log_softmax.backward()
        patch_grad = perturbated_image.grad.clone().cpu()
        perturbated_image.grad.data.zero_()
        applied_patch = lr * patch_grad + applied_patch.type(torch.FloatTensor)
        applied_patch = torch.clamp(applied_patch, min=-3, max=3)
        # Test the patch
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image = torch.clamp(perturbated_image, min=-3, max=3)
        perturbated_image = perturbated_image.cuda()
        output = model(perturbated_image)
        target_probability = torch.nn.functional.softmax(output, dim=1).data[0][target]


        print("ep: {}, input: {}, step: {}, target prob: {}".format(epoch, input_index, count, target_probability))


    perturbated_image = perturbated_image.cpu().numpy()
    applied_patch = applied_patch.cpu().numpy()
    return perturbated_image, applied_patch

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
train_loader, test_loader = dataloader(args.train_size, args.test_size, args.data_dir, args.batch_size, args.num_workers, 50000)

# Test the accuracy of model on trainset and testset
trainset_acc, test_acc = test(model, train_loader), test(model, test_loader)
print('Accuracy of the model on clean trainset and testset is {:.3f}% and {:.3f}%'.format(100*trainset_acc, 100*test_acc))

# Initialize the patch
patch = patch_initialization(args.patch_type, image_size=(3, 224, 224), noise_percentage=args.noise_percentage)
print('The shape of the patch is', patch.shape)

#with open(args.log_dir, 'w') as f:
#    writer = csv.writer(f)
#    writer.writerow(["epoch", "train_success", "test_success"])

best_patch_epoch =  0

patch_name = "{}_mask_{}_adaptive_evade_detection_best_org_patch_{}.npy".format(args.model, str(args.mask_percentage).replace('0.', ''), str(args.noise_percentage).replace('.', ''))
try:
    patch = np.load(patch_name)
    best_patch_success_rate = test_patch(args.patch_type, args.target, patch, test_loader, model)
    print("restore from patch {}, attack success rate {}".format(patch_name, best_patch_success_rate))
except:
    print("Initialize new patch")
    best_patch_success_rate = 0
import datetime

# Generate the patch
for epoch in range(args.epochs):
    print("=========== attack under {} epoch =========".format(epoch+1))
    cnt = 0
    train_total, train_actual_total, train_success = 0, 0, 0

    start_ep = datetime.datetime.now().replace(microsecond=0)

    for (image, label) in train_loader:
        train_total += label.shape[0]
        #print("\t{} image so far, total: {}".format(cnt, args.train_size))
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        if predicted[0] == label and predicted[0].data.cpu().numpy() != args.target:
            train_actual_total += 1
            applied_patch, mask, x_location, y_location = mask_generation(args.patch_type, patch, image_size=(3, 224, 224))

            # "modify the mask to randomly mask the patch regions"
            #cv2.imwrite( 'org_mask.png', np.transpose(mask*255, (1,2,0))  )
            #"NOTE: the coordiante returned is for RGB, y-axis downward, x-axis right-ward" 
            len_for_random_inx = patch.shape[1] #"length of the area to derive the random index for masking within the salient region"  
            pts_for_selection = int( (len_for_random_inx*len_for_random_inx) * args.mask_percentage )
            # randomly choose certain amount of points within the mask region to be blocked out, 
            # these points will not be subject to the adversarial perturbation
            x_masked_index, y_masked_index = gencoordinates(size= pts_for_selection, x_min=x_location, x_max=x_location+ patch.shape[1]-1 , \
                                                                y_min= y_location, y_max=y_location+ patch.shape[2]-1)
            #"row first, x_masked_index is from x_location, which gives the row index" 
            mask[ :, x_masked_index, y_masked_index] = 0

            #cv2.imwrite('mask.png', np.transpose( mask * 255, (1,2,0)) )

            perturbated_image, applied_patch = patch_attack(image, applied_patch, mask, train_actual_total, epoch+1, args.target, args.probability_threshold, model, args.lr, args.max_iteration)
            perturbated_image = torch.from_numpy(perturbated_image).cuda()
            output = model(perturbated_image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0].data.cpu().numpy() == args.target:
             train_success += 1
            patch = applied_patch[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]

        cnt += 1
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


    end_ep = datetime.datetime.now().replace(microsecond=0)


    try:
        print("Epoch:{} Patch attack success rate on trainset: {:.3f}%".format(epoch+1, 100 * train_success / train_actual_total))
    except:
        print('no success on trainset')

    try:
        train_success_rate = test_patch(args.patch_type, args.target, patch, test_loader, model)
    except:
        train_success_rate = 0.
    print("Epoch:{} Patch attack success rate on trainset with universal patch: {:.3f}%".format(epoch+1, 100 * train_success_rate))
    
    try:
        test_success_rate = test_patch(args.patch_type, args.target, patch, test_loader, model)
    except:
        test_success_rate = 0.
    print("Epoch:{} Patch attack success rate on testset with universal patch: {:.3f}%".format(epoch+1, 100 * test_success_rate))

    with open(args.model + "_" + args.log_dir, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_success_rate, test_success_rate, end_ep - start_ep])

    if test_success_rate > best_patch_success_rate:
        best_patch_success_rate = test_success_rate
        best_patch_epoch = epoch
        #np.save("best_patch.npy", torch.from_numpy(cur_patch))
        
        np.save(patch_name, torch.from_numpy(patch))



    # Load the statistics and generate the line
    #log_generation(args.log_dir)

print("The best patch is found at epoch {} with success rate {}% on testset".format(best_patch_epoch, 100 * best_patch_success_rate))






