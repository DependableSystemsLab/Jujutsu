# adaptive attack to evade detection by reducing the influence to the output

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
from smooth_grad import SmoothGrad
import cv2
import math



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help="batch size")
parser.add_argument('--num_workers', type=int, default=2, help="num_workers")
parser.add_argument('--train_size', type=int, default=2000, help="number of training images")
parser.add_argument('--test_size', type=int, default=2000, help="number of test images")

parser.add_argument('--noise_percentage', type=float, default=0.06, help="percentage of the patch size compared with the image size")
parser.add_argument('--probability_threshold', type=float, default=0.9, help="minimum target probability")
parser.add_argument('--secondary_threshold', type=float, default=0.5, help="secondary target probability. We accept this lower target probability if the attack is able to evade the detection")
parser.add_argument('--lr', type=float, default=1.0, help="learning rate")
parser.add_argument('--evasion_lr', type=float, default=1e-2, help="learning rate")
parser.add_argument('--model', type=str, required=True, help='model name')


parser.add_argument('--max_iteration', type=int, default=1000, help="max iteration")
parser.add_argument('--target', type=int, default=859, help="target label")
parser.add_argument('--epochs', type=int, default=30, help="total epoch")
parser.add_argument('--data_dir', type=str, default='/imagenet-val/ILSVRC2012_img_val', help="dir of the dataset")
parser.add_argument('--patch_type', type=str, default='rectangle', help="type of the patch")
parser.add_argument('--GPU', type=str, required=True, help="index pf used GPU")
parser.add_argument('--log_dir', type=str, default='patch_attack_log.csv', help='dir of the log')
parser.add_argument('--detect_box_half_radius', type = int, default=51, help='half of the length of the detection bounding box. Eg, 51 means we draw a 102*102 box')

"for smoothgrad"
parser.add_argument('--image', type=str, required=False)
parser.add_argument('--sigma', type=float, default=0.10)
parser.add_argument('--n_samples', type=int, default=50)
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--guided', action='store_true', default=False)


args = parser.parse_args()



classification_lr = args.lr # step size for misclassification objective
evasion_lr = args.evasion_lr # step size for evasion objective, smaller in order to reduce the counter-effect for misclassfication
secondary_threshold = args.secondary_threshold # accept such threshold if the evasion criteria is met

def generate_inverse_mask(img_maxLoc, radius=51, mask_shape= (3,224,224)):
    # genereate the inverse of the mask, mask is where we put the adversarial patch
    def get_coordinate_for_recetange(x, y, radius, x_max, y_max):
        # get the start / end coordinate for the rectangle, centering at (x,y) with length of 2*radius

        start = []
        end = [] # coordinate of the rectangle

        start.append( x - radius )
        start.append( y - radius )
        end.append( x + radius )
        end.append( y + radius )

        if(start[0] < 0): 
            end[0] += abs(start[0]) 
            start[0] = 0
        if(start[1] < 0):
            end[1] += abs(start[1])
            start[1] = 0
        if(end[0] > x_max):
            start[0] -= (end[0] - x_max)
            end[0] = x_max
        if(end[1] > y_max):
            start[1] -= (end[1] - y_max)
            end[1] = y_max

        return tuple(start), tuple(end)

    # inverse mask of the adv patch
    mask = np.ones(shape= (mask_shape[1], mask_shape[2]) )
    mask *= 255 
    start, end = get_coordinate_for_recetange(img_maxLoc[0], img_maxLoc[1], radius, mask_shape[1], mask_shape[2])

    # first dim is y axis, second x 
    mask[start[0]:end[0], start[1]:end[1]] = 0 # evade the high attribution features

    return mask, start, end

def blur(img, radius=51, padding=25):
    "averaging filter over the saliency map, used for identify the high attribution features"
    # img : gray image of saliency map
    # maxLoc output from cv2.minmaxLoc follows [x, y] (column, row)
    # maxLoc output from numpy (torch conv implementation) follows [y, x] (row, column)
    # NOTE: index of the matrix is goes y (row) first, then x (column), i.e., [y, x]

    mean_conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=radius, padding=padding, bias=False)
    k_size = (radius, radius) 
    
    k_weigth = np.ones(shape=k_size) / float(radius*radius) # kernel for averaging 
    k_weigth = k_weigth[np.newaxis, ...]
    k_weigth = k_weigth[np.newaxis, ...]

    mean_conv.weight.data = torch.FloatTensor(k_weigth).cuda()

    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    output = mean_conv( img )
    output = output.cpu().detach().numpy()[0][0]
    
    return output




# Patch attack via optimization
# According to reference [1], one image is attacked each time
# Assert: applied patch should be a numpy
# Return the final perturbated picture and the applied patch. Their types are both numpy
def patch_attack(image, applied_patch, mask, target, probability_threshold, model, mask_x_location, mask_y_location, input_index, epoch, patch_shape, label, lr, max_iteration, detection_radius = 51*2):
    model.eval()
    mask_shape = mask.shape

    inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
    )


    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)
    target_probability, count = 0, 0
    perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))



    # target distance metric for evading the detection
    # distance is measured between the center of the adversarial patch, and the center of the detection bounding box
    # this is defined as the distance when the patch is completely out of the detection box 
    evasion_dist_threshold = int(math.sqrt( 2 * ((detection_radius/2)**2)) + math.sqrt( (patch_shape[1]/2)**2 + (patch_shape[2]/2)**2 )  )  

    center_dist = 0 # this is the distance of the maximal datapoint within the saliency map to the center of the actual location of adversarial patch
                    # we want to increase this distance, so that the saliency map will NOT reveal the actual location of the adv patch
    isMisclass = True # whether optimize for targeted misclassification first

    "use a secondary threshold (acceptable when fooling interpreation is succeeded)"
    while (target_probability < secondary_threshold or center_dist < evasion_dist_threshold ) and count < max_iteration:
    #while count < max_iteration:
        count += 1
        # Optimize the patch
        perturbated_image = Variable(perturbated_image.data, requires_grad=True)
        per_image = perturbated_image
        per_image = per_image.cuda()
        output = model(per_image)

        tensor_target = torch.tensor(target)
        tensor_target = tensor_target.unsqueeze(0)
        tensor_target = tensor_target.cuda() 
        "loss term for causing misclassification"
        misclass_loss =  torch.nn.functional.cross_entropy(input=output, target=tensor_target,reduction='none')

        # center of the patch location
        # "x is downward along Y-axis, y is rightward along X-axis"
        maxLoc = [ int((mask_x_location + mask_x_location + patch_shape[1])/2), int((mask_y_location + mask_y_location + patch_shape[2])/2)]
        #print('center of patch ', maxLoc)

        if(target_probability < probability_threshold and \
             not (target_probability > secondary_threshold and isMisclass == False) ):
            "optimize for targeted misclassification"
            misclass_loss = -1 * misclass_loss 
            loss_term = misclass_loss
            step_size = classification_lr
            isMisclass = True
        else:
            "optimize for evading detection"

            #print(torch.autograd.grad( misclass_loss, perturbated_image, retain_graph=False,allow_unused=True) )
            #"loss term for the patch to evade saliency localization"
            #"Get the salient feature from the perturbated image"
            smooth_grad.feed_image(per_image)
            #prob, idx = smooth_grad.forward()                
            saliency = smooth_grad.return_saliency(idx= target) # find the saliency of the prediction to the adv label (not necessarily the top-1 label right now)  
        
            inverse_mask = np.zeros(shape = (patch_shape[1], patch_shape[2]) )
            saliency /= 255
            inverse_mask = torch.from_numpy(inverse_mask).type(torch.FloatTensor)#.cuda()

            saliency = saliency.cuda()
            inverse_mask = inverse_mask.cuda()

            # get the average pixel distance 
            "The direction of this loss is to reduce the influence of the patch to the target label, thus reducing the prediction prob"
            "Not necessarily reducing the loss becuase of the -- approximation -- of gradient"
            evasion_loss = -1 * (torch.dist(saliency[ mask_x_location:mask_x_location +patch_shape[1], mask_y_location:mask_y_location+patch_shape[2]  ] , inverse_mask, p=2) **2    )  

            loss_term = -1 * evasion_loss 
            step_size = evasion_lr
            isMisclass = False
       
        grad = torch.autograd.grad(loss_term, per_image, retain_graph=True,allow_unused=True)[0]

        #print( 'misclass loss: {} \t grad: {}'.format( misclass_loss.data, torch.autograd.grad(misclass_loss, per_image, retain_graph=True,allow_unused=True)[0].sum() ))
        #print( 'evasion loss {} \t grad: {}'.format( evasion_loss, torch.autograd.grad(evasion_loss, per_image, retain_graph=False,allow_unused=True)[0].sum() ))

        delta = grad * step_size  
        
        delta = delta.cuda()
        applied_patch = applied_patch.type(torch.FloatTensor)
        applied_patch = applied_patch.cuda()
        applied_patch = delta + applied_patch 
        applied_patch = torch.clamp(applied_patch, min=-3, max=3)

    
        # Test the patch
        perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1-mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
        perturbated_image = torch.clamp(perturbated_image, min=-3, max=3)
        perturbated_image = perturbated_image.cuda()
        output = model(perturbated_image)
        target_probability = torch.nn.functional.softmax(output, dim=1).data[0][target]

        predictions = output.cpu().detach().numpy() 
        index = (np.argsort(predictions[0, :])[::-1])[0:5]
        #print(index)

        center_dist = -1
        if(not isMisclass ):
            "Now we measure how far the adversarial patch is away from the detection bounding box"
            "If the distance meets the target distance, we say this is succeeded"

            #"running SmoothGrad without replacing relu"
            org_smooth_grad.feed_image(perturbated_image) 
            s aliency = org_smooth_grad.return_saliency(idx= target) # find the saliency of the prediction to the adv label (not necessarily the top-1 label right now)  
            #"avg blurring"
            saliency_maxLoc = blur(saliency)

            #" return value: X - downward Y-axis, Y - rightward X-axis "
            saliency_maxLoc = np.where( saliency_maxLoc == np.amax(saliency_maxLoc) )
            center_dist = math.sqrt( (saliency_maxLoc[0][0] - maxLoc[0])**2 + (saliency_maxLoc[1][0] - maxLoc[1])**2 )
            #print( 'evasion loss {} '.format( evasion_loss))

        #print("Ep {}, ==> {}, lab: {}, step: {}, misclass: ==> {}, dist: ==> {:.3f} / {} , target prob: {:.3f}".format(epoch, input_index, label, count, isMisclass, center_dist, evasion_dist_threshold, target_probability))
        print("Ep {}, ==> {}, pred: {}, lab: {}, step: {}, misclass: ==> {}, dist: ==> {:.3f} / {} , target prob: {:.3f}".format(epoch, input_index, index[0], label, count, isMisclass, center_dist, evasion_dist_threshold, target_probability))

        print() 

    perturbated_image = perturbated_image.cpu().detach().numpy()
    applied_patch = applied_patch.cpu().detach().numpy()
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

# Setup the SmoothGrad
"Use the original image for gradient approximation, "
# model where relu is replaced with parametric softplus
smooth_grad = SmoothGrad(model=model, cuda=args.cuda, sigma=0.,
                         n_samples=1, guided=args.guided, replace_relu=True)

# original model for doing normal backprop
org_smooth_grad = SmoothGrad(model=model, cuda=args.cuda, sigma=args.sigma,
                         n_samples=args.n_samples, guided=args.guided, replace_relu=False)
 


# Load the datasets
train_loader, test_loader = dataloader(args.train_size, args.test_size, args.data_dir, args.batch_size, args.num_workers, 50000)

# Test the accuracy of model on trainset and testset
#trainset_acc, test_acc = test(model, train_loader), test(model, test_loader)
#print('Accuracy of the model on clean trainset and testset is {:.3f}% and {:.3f}%'.format(100*trainset_acc, 100*test_acc))

patch_name = "adaptive_{}_best_org_patch_{}.npy".format(args.model, str(args.noise_percentage).replace('.', ''))
        
try:
    patch = np.load(patch_name)
    print("load the patch to resume attack optimization {}".format(patch_name))
    best_patch_success_rate = test_patch(args.patch_type, args.target, patch, test_loader, model)
except:        
    # Initialize the patch
    patch = patch_initialization(args.patch_type, image_size=(3, 224, 224), noise_percentage=args.noise_percentage)
    print("Initialize new patch")
    best_patch_success_rate = 0
print('The shape of the patch is', patch.shape)





import datetime



best_patch_epoch = 0

# Generate the patch
for epoch in range(args.epochs):
    start_ep = datetime.datetime.now().replace(microsecond=0)

    train_total, train_actual_total, train_success = 0, 0, 0
    cnt = 0
    for (image, label) in train_loader: 
        train_total += label.shape[0]
        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        
        if predicted[0] == label and predicted[0].data.cpu().numpy() != args.target:
             cnt += 1
             train_actual_total += 1
             applied_patch, mask, x_location, y_location = mask_generation(args.patch_type, patch, image_size=(3, 224, 224))
             perturbated_image, applied_patch = patch_attack(image, applied_patch, mask, args.target, args.probability_threshold, model, x_location, y_location, cnt, epoch+1, patch.shape, label.item(), args.lr, args.max_iteration)
             perturbated_image = torch.from_numpy(perturbated_image).cuda()
             output = model(perturbated_image)
             _, predicted = torch.max(output.data, 1)
             if predicted[0].data.cpu().numpy() == args.target:
                 train_success += 1
             patch = applied_patch[0][:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]]
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

        np.save(patch_name, torch.from_numpy(patch))




print("The best patch is found at epoch {} with success rate {}% on testset".format(best_patch_epoch, 100 * best_patch_success_rate))






