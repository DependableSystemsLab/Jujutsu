"""
This is to generate adv samples by applying the patch to the clean images, and also calculate its saliency map using smoothgrad
The patched image is saved as npy format and saliency map as png
"""




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
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader 

def gen_patched_img(patch_type, target, patch, test_loader, model, org_img_folder, adv_img_folder, org_saliency_folder, adv_saliency_folder,  smooth_grad):
    model.eval()

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255]
        )


    test_total, test_actual_total, test_success = 0, 0, 0
    for (image, label) in test_loader:
        test_total += label.shape[0]

        if(test_total > args.test_size):
            break

        assert image.shape[0] == 1, 'Only one picture should be loaded each time.'
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        if predicted[0] == label and predicted[0].data.cpu().numpy() != target:
            test_actual_total += 1
            applied_patch, mask, x_location, y_location = mask_generation(patch_type, patch, image_size=(3, 224, 224))
            applied_patch = torch.from_numpy(applied_patch)
            mask = torch.from_numpy(mask)
            perturbated_image = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + torch.mul((1 - mask.type(torch.FloatTensor)), image.type(torch.FloatTensor))
            perturbated_image = perturbated_image.cuda()
            output = model(perturbated_image)
            _, predicted = torch.max(output.data, 1)
            if predicted[0].data.cpu().numpy() == target:
                test_success += 1

                #print(" ====== {} input ".format(test_success))
                # get saliency from org img
                smooth_grad.feed_image(image)
                prob, idx = smooth_grad.forward()
                smooth_grad.generate(filename= org_saliency_folder + "/" + str(test_success) +  "_" + str(label.item()), idx=idx[0])  # "0" means only calculate the saliency for the top label

                smooth_grad.feed_image(perturbated_image)
                prob, idx = smooth_grad.forward()
                smooth_grad.generate(filename= adv_saliency_folder + "/" + str(test_success) +  "_" + str(label.item()) , idx=idx[0])  


                np.save( adv_img_folder + "/" + str(test_success) +  "_" + str(label.item()) + ".npy" , perturbated_image.cpu().numpy() )
                np.save( org_img_folder + "/" + str(test_success) +  "_" + str(label.item()) + ".npy" , image.cpu().numpy() )

    print(test_success, test_actual_total)
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
parser.add_argument('--data_dir', type=str, default='imagenet-val/ILSVRC2012_img_val', help="dir of the dataset")
parser.add_argument('--GPU', type=str, default='0', help="index pf used GPU")
parser.add_argument('--log_dir', type=str, default='patch_attack_log.csv', help='dir of the log')
parser.add_argument('--patch_file', type=str, required=True, help='file to the adv patch')
parser.add_argument('--model', type=str, default='resnet18', help='model name') 
parser.add_argument('--patch_type', type=str, default='square', help="patch type: rectangle or square")
parser.add_argument('--save_folder', type=str, default='.', help="directory to save the resuluting images")
parser.add_argument('--model_path', type=str, default='./imagenette/resnet18-imagenette.pt')
"for smoothgrad"
parser.add_argument('--image', type=str, required=False)
parser.add_argument('--sigma', type=float, default=0.10)
parser.add_argument('--n_samples', type=int, default=50)
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--guided', action='store_true', default=False)

args = parser.parse_args()



seed = 1
np.random.seed(seed)
random.seed(seed) 
torch.manual_seed(seed)

device = torch.device("cuda:%s"%args.GPU if torch.cuda.is_available() else "cpu")

# Load the model
# setting up model
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10) # multi-class classification (num_of_class == 307)
model.load_state_dict(torch.load(args.model_path)) 
model = model.to(device)
model.eval()

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


image_datasets = {x: datasets.ImageFolder(os.path.join(args.data_dir, x), data_transforms[x]) for x in ['train', 'val']}


train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=1, shuffle=False, num_workers=1)
test_loader = torch.utils.data.DataLoader(image_datasets['val'], batch_size=1, shuffle=False, num_workers=1)







# Initialize the patch
patch = np.load(args.patch_file)
print('The shape of the patch is', patch.shape)
 

# Setup the SmoothGrad
smooth_grad = SmoothGrad(model=model, cuda=args.cuda, sigma=args.sigma,
                         n_samples=args.n_samples, guided=args.guided)



test_success_rate = 0
train_success_rate = 0



save_folder2 = "{}/imagenette_{}_{}_npy_test_org_result_{}".format(args.save_folder.rstrip("/"),  args.patch_type, args.target, str(args.noise_percentage).replace('.', ''))
save_folder22 = "{}/imagenette_{}_{}_npy_test_adv_result_{}".format(args.save_folder.rstrip("/"), args.patch_type, args.target, str(args.noise_percentage).replace('.', ''))
save_folder222 = "{}/imagenette_{}_{}_npy_test_org_saliency_{}".format(args.save_folder.rstrip("/"),   args.patch_type, args.target, str(args.noise_percentage).replace('.', ''))
save_folder2222 = "{}/imagenette_{}_{}_npy_test_adv_saliency_{}".format(args.save_folder.rstrip("/"),   args.patch_type, args.target, str(args.noise_percentage).replace('.', ''))


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


test_success_rate = gen_patched_img(patch_type= args.patch_type, target= args.target, patch= patch, test_loader= test_loader, model= model, org_img_folder = save_folder2, adv_img_folder=save_folder22, org_saliency_folder=save_folder222, adv_saliency_folder=save_folder2222, smooth_grad=smooth_grad)
print("  Patch attack success rate on testset: {:.3f}%".format( 100 * test_success_rate))  

print( args.patch_file  )








