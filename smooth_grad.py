from __future__ import print_function

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F
from tqdm import tqdm 
import copy 




class SmoothGrad(object):

    def __init__(self, model, cuda, sigma, n_samples, guided, replace_relu=False):


        self.model = copy.deepcopy(model)
        #self.model = model
        self.model.eval()
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()

        self.replace_relu = replace_relu
        self.sigma = sigma
        self.n_samples = n_samples


        "replace relu with softplus for calculating second-order derivative"
        def convert_relu_to_softplus(model):
            for child_name, child in model.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(model, child_name, nn.Softplus(beta=10))
                else:
                    convert_relu_to_softplus(child)

        if(replace_relu):

            convert_relu_to_softplus(self.model)
            


        # Guided Backpropagation
        if guided:
            def func(module, grad_in, grad_out):
                # Pass only positive gradients
                if isinstance(module, nn.ReLU):
                    return (torch.clamp(grad_in[0], min=0.0),)

            for module in self.model.named_modules():
                module[1].register_backward_hook(func)



    def load_image(self, filename, transform):
        raw_image = cv2.imread(filename)[:, :, ::-1]
        raw_image = cv2.resize(raw_image, (224, 224))
        image = transform(raw_image).unsqueeze(0)
        image = image.cuda() if self.cuda else image
        self.image = Variable(image, volatile=False, requires_grad=True)

    def encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.probs.size()[-1]).zero_()
        one_hot[0][idx] = 1.0
        return one_hot.cuda() if self.cuda else one_hot

    def generate(self, idx, filename):
        grads = []
        image = self.image.data.cpu()
        sigma = (image.max() - image.min()) * self.sigma

        for i in range(self.n_samples):
            # Add gaussian noises
            noised_image = image + torch.randn(image.size()) * sigma
            noised_image = noised_image.cuda() if self.cuda else noised_image
            self.image = Variable(
                noised_image, volatile=False, requires_grad=True)
            self.forward()
            self.backward(idx=idx)

            # Sample the gradients on the pixel-space
            grad = self.image.grad.data.cpu().numpy()
            grads.append(grad)

            if (i+1) % self.n_samples == 0:
                grad = np.mean(np.array(grads), axis=0)
                saliency = np.max(np.abs(grad), axis=1)[0]
                saliency -= saliency.min()
                saliency /= saliency.max()
                saliency = np.uint8(saliency * 255)
                cv2.imwrite(filename + '_{:04d}.png'.format(i), saliency)

            self.model.zero_grad()

    def forward(self):
        self.preds = self.model.forward(self.image)
        self.probs = F.softmax(self.preds)[0]
        self.prob, self.idx = self.probs.data.sort(0, True)
        return self.prob, self.idx

    def backward(self, idx):
        # Compute the gradients wrt the specific class
        self.model.zero_grad()
        one_hot = self.encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)


    def return_saliency(self, idx):
        grads = []
        image = self.image.data.cpu()
        sigma = (image.max() - image.min()) * self.sigma

        self.forward()
        # create a empty tensor to store the noisy gradients
        first_grad = torch.autograd.grad( self.preds[0][idx], self.image,  create_graph=True,allow_unused=True)[0]
        first_grad_size = list(first_grad.size())
        first_grad_size[0] = 0
        first_order_grads = torch.empty( size= first_grad_size, requires_grad=True )
        first_order_grads = first_order_grads.cuda()


        for i in range(self.n_samples):
            if(not self.replace_relu):
                # for normal smoothgrad backprop
                # Add gaussian noises
                noised_image = image + torch.randn(image.size()) * sigma
                noised_image = noised_image.cuda() if self.cuda else noised_image
                self.image = Variable(
                    noised_image, volatile=False, requires_grad=True)
            
            # else: use the original image for gradient approximation 
            self.forward()

            "need to set create_graph as true in order to calculate high-order derivative"
            # calculate the first-order grad
            first_grad = torch.autograd.grad( self.preds[0][idx], self.image,  create_graph=True,allow_unused=True)[0]

           
 
            if (i+1) % self.n_samples == 0:
                # get saliency by torch tensor
                first_order_grads = torch.cat((first_order_grads, first_grad ), 0) 

                grad = torch.mean(first_order_grads, dim=0)  
                saliency = torch.max( torch.abs(grad) , dim=0 )[0] 
                saliency = saliency - torch.min(saliency)
                saliency = saliency / torch.max( saliency )
                
                saliency *= 255

 

            self.model.zero_grad()          


        return saliency


    def feed_image(self, image): 
        if(not self.replace_relu):
            self.image = Variable(image, volatile=False, requires_grad=True)
        else:
            self.image = image




