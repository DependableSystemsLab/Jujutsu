import math
import random
import numpy as np
import time
import scipy
import os 
import cv2
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
from skimage.transform import resize
import scipy
import scipy.stats
import torchvision.transforms as transforms

seed = 1
np.random.seed(seed)
random.seed(seed) 

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, required=True, help='model name')
parser.add_argument('--GPU', type=str, default='0', help="index pf used GPU")
parser.add_argument('--partition', type=int, default=1, help="num of partitions for the superimposed images. E.g., if you set 1, all the superimposed imgs will be executed at once")

parser.add_argument('--train_img_folder', type=str, required=True, help='img for deriving the detection boundary')
parser.add_argument('--benign_img_folder', type=str, required=True, help='benign images for evaluation')
parser.add_argument('--adv_img_folder', type=str, required=True, help='adv images for evaluation')
parser.add_argument('--held_out_img_folder', type=str, required=True, help='img for superimposing')





args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

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


def readTrainImg( n ):
    folder = args.train_img_folder
    files = []
    for r, d, f in os.walk( folder ): 
        for file in f:
            files.append(os.path.join(r, file))

    random.shuffle(files)
    files = files[:n]

    return files

def readBenignImg( n ):


    imgFolders = args.benign_img_folder

    files = []

    for r, d, f in os.walk( imgFolders ):
        for file in f:
            files.append(os.path.join(r, file))


    #random.shuffle(files)
    #files = files[:n]

    return files

def readAdvImg( n ):

    imgFolders = args.adv_img_folder

    files = []

    for r, d, f in os.walk( imgFolders ):
        for file in f:
            files.append(os.path.join(r, file))

    #random.shuffle(files)
    #files = files[:n]

    return files

def readRandomHoldOut( n ):

    imgFolder = args.held_out_img_folder

    files = []
   
    for r, d, f in os.walk( imgFolder ):
        for file in f:
            files.append(os.path.join(r, file))

    random.shuffle(files)
    files = files[:n]

    return files



def superimpose(background, overlay):
    background = background.astype(float)
    overlay = overlay.astype(float)

    added_image = cv2.addWeighted(background,1,overlay,1,0)
    #return (added_image.reshape(32,32,3))

    "transpose to channel first"
    return np.transpose(added_image, (2,0,1))
    #return (added_image.reshape(224,224,3))

def entropyCal(background, n,set_hold_out):
    entropy_sum = [0] * n
    x1_add = [0] * n
    #index_overlay = np.random.randint(40000,49999, size=n)
    for x in range(n):

        new_hold_out = cv2.imread(set_hold_out[x])
        new_hold_out = cv2.cvtColor(new_hold_out, cv2.COLOR_BGR2RGB) 

        new_hold_out = new_hold_out.astype(float)
        new_hold_out /= 255

 
        x1_add[x] = (superimpose(background, new_hold_out ))


    #py1_add = model.predict(np.array(x1_add))
    x1_add = np.asarray(x1_add) 

    sum_preds = []
    partition = args.partition
    EntropySum = 0.

    # paritition the superimposed image for calculating the prediction entropy
    for i in range(partition):
        start = i * int(n/partition)
        end = i * int(n/partition) + int(n/partition)

        batch = x1_add[ start : end ]
        batch = np.asarray(batch)

        imgs = torch.from_numpy(batch)
        imgs = imgs.type(torch.FloatTensor)
        imgs = imgs.cuda()


        for j in range( int(n/partition) ):
            imgs[j] = normalize_img(imgs[j])
 

        preds = model(imgs)

        '''
        predictions = preds.detach().cpu().numpy() 
        index = (np.argsort(predictions[0, :])[::-1])[0:5]
        print(index)
        '''

        scores = torch.nn.functional.softmax(preds, dim=1).data 
        scores = scores.detach().cpu().numpy() 

        EntropySum += -np.nansum(scores*np.log2(scores))


    #EntropySum = -np.nansum(py1_add*np.log2(py1_add))
    return EntropySum




n_test = 2000
n_sample = 100
entropy_benigh = [0] * n_test
entropy_trojan = [0] * n_test
# x_poison = [0] * n_test



"the imgs are the pointer to the file name"
set_benign_imgs = readBenignImg( n_test )
set_adv_imgs = readAdvImg( n_test )

set_train_imgs = readTrainImg(n_test)
set_hold_out = readRandomHoldOut( n_sample )





"Images needs to be normalized"

normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
    )



"derive the entropy"
for j in range(n_test): 
    #x_background = x_train[j+26000] 
    "read a random benign img to x_background"

    print("deriving entropy {} inputs on {}".format(j+1, args.model))

    #x_background = np.load(set_benign_imgs[j]) 
    #x_background = np.transpose( x_background[0], (1,2,0) ) 
    x_background = cv2.imread( set_train_imgs[j] ) 
    x_background = cv2.cvtColor(x_background, cv2.COLOR_BGR2RGB)
    x_background = resize(x_background, (224,224,3))

    x_background = x_background.astype(float)
    x_background /= 255.

    entropy_benigh[j] = entropyCal(x_background, n_sample, set_hold_out)

entropy_benigh = [x / n_sample for x in entropy_benigh] # get entropy for 2000 clean inputs


(mu, sigma) = scipy.stats.norm.fit(entropy_benigh)
print(mu, sigma)
threshold = scipy.stats.norm.ppf(0.01, loc = mu, scale =  sigma) #use a preset FRR of 0.01. This can be 
print("threshold for {}: {}".format(args.model, threshold))
print() 










mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


"evaluation on adv inputs"
j = 1
entropy_adv = []

print("starting detection on adv inputs")
for each in set_adv_imgs:

    #print("evaluation {} for adv inputs on {}".format(j, args.model))  

    x_poison = np.load(each) 
    '''
    img = torch.from_numpy(x_poison)
    img = img.type(torch.FloatTensor)
    img = img.cuda() 
    output = model(img)
    predictions = output.detach().cpu().numpy() 
    index = (np.argsort(predictions[0, :])[::-1])[0:5]
    print(index)
    '''

    x_poison = x_poison[0]
    # de-normalize the images
    for i in range(3):
        x_poison[i, :, :] *= std[i]
        x_poison[i, :, :] += mean[i]

    "Numpy image are in RGB format"
    x_poison = np.transpose( x_poison, (1,2,0) )



    entropy_adv.append( entropyCal(x_poison, n_sample,set_hold_out) )

    j+=1


entropy_adv = [x / n_sample for x in entropy_adv] # get entropy for 2000 trojaned inputs


"evaluation on clean inputs"
j = 1
entropy_clean = []
print("starting detection on benign inputs")
for each in set_benign_imgs:

    #print("evaluation {} for benign inputs on {}".format(j, args.model)) 


    x_clean = np.load(each)

    x_clean = x_clean[0]
    # de-normalize the images
    for i in range(3):
        x_clean[i, :, :] *= std[i]
        x_clean[i, :, :] += mean[i]

    x_clean = np.transpose( x_clean, (1,2,0) )


    entropy_clean.append( entropyCal(x_clean, n_sample,set_hold_out) )

    j+=1 

entropy_clean = [x / n_sample for x in entropy_clean] # get entropy for 2000 trojaned inputs


#print()
#print()
#print()
print("low entropy indicates adv img, and vice versa")
print()
print( "correctly detect adv img: {} out of {}".format(  sum(i < threshold for i in entropy_adv), len(set_adv_imgs) ) )
print( "correctly detect benign img: {} out of {}".format(  sum(i > threshold for i in entropy_clean), len(set_benign_imgs) ) )















