import argparse
import numpy as np
import os
import pandas as pd
import yaml
from tqdm import tqdm
from classifiers.lisacnn.detector import LisaCNNModel
from classifiers.gtsrbcnn.detector import GtsrbCNNModel
from classifiers.resnet50.detector import ResNet50Model
#import cv2

#import torch
#from  torchvision import datasets, transforms
#from torchvision.utils import save_image

class Sentinet(object):
    def __init__(self, model):
        self.model = model
    
    # Algorithm 1
    def class_proposal(self, img, num_classes=1):
        # We perform our class proposal by using the top 5 classes for our given sign (excluding stop) + stop sign
        label_probabilities = self.model.forward(img, save_image=False, probs_only=True)
        top_classes = list(label_probabilities.argsort()[-(num_classes + 1):][::-1])
        prediction = (top_classes[0], label_probabilities[top_classes][0])
        #print(prediction)
        final_set = [(x, label_probabilities[x]) for x in top_classes[1:]]
        return prediction, final_set

    # Algorithm 2
    def mask_generation(self, img, prediction, proposed_classes, threshold=0.3, saliency_function=None):

        if saliency_function == 'xrai':
            saliency_function = self.model.xrai
        if saliency_function == 'gradcam':
            saliency_function = self.model.gradcam

        mask_y = saliency_function(img, prediction[0], binarize=True, threshold=threshold)
        # for numpy 
        #save_image(torch.from_numpy(mask_y).type(torch.FloatTensor), 'mask_y.jpeg')
        #save_image(torch.from_numpy(img[0]).type(torch.FloatTensor), 'input_img.jpeg')

        mask_set = []
        
        for (yp_class, yp_conf) in proposed_classes:
            mask_yp = saliency_function(img, yp_class, binarize=True, threshold=threshold)
            #save_image(torch.from_numpy(mask_yp).type(torch.FloatTensor), 'mask_yp.jpeg')

            delta_mask = mask_y & ~mask_yp 
            #save_image(torch.from_numpy(delta_mask).type(torch.FloatTensor), 'delta_mask.jpeg')
           
            mask_set.append({'mask': delta_mask, 'confidence': yp_conf, 'class': yp_class})
        
        return mask_set, mask_y


    # Algorithm 3
    def testing(self, img, adv_example_class, masks, test_image_paths, pattern='noise'): #test_image_paths -> ref images, adv_example_class -> prediction of img
        img = img[0] # comment for lisacnn_cvpr
        '''
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
            std=[1/0.229, 1/0.224, 1/0.255]
        )
        '''
        R = []
        IP = []
        if pattern == 'noise':
            #inert_pattern = (np.random.random(img.shape) * 255).astype(np.uint8)
            inert_pattern = np.random.random(img.shape)
        elif pattern == 'checker':
            inert_pattern = self.model.load_image_jpg('pattern.png')
        else:
            raise ValueError('Unsupported pattern, choose one of %s' % ["noise", "checker"])

        for m in masks:
            img_mask = np.copy(img)
            img_mask[~m['mask']] = 0 #same as delta_mask. 3, 224, 224
            R.append(img_mask)
            inert_mask = np.copy(inert_pattern)
            inert_mask[~m['mask']] = 0
            IP.append(inert_mask)

        Xr = []
        Xip = []
        count =1
        for test_im_path in test_image_paths:
            image = self.model.load_image(test_im_path)
            for i, (r, m) in enumerate(zip(R, masks)):
                #mask = (r.sum(axis=-1) == 0)[..., np.newaxis].astype(np.uint8) # r.shape -> 3, 224, 224. mask.shape 3, 224, 1
                #new_image = r + image*mask     
                delta_mask = m['mask'][np.newaxis, ...]          
                mask = r[np.newaxis, ...]
                new_image = mask + (1-delta_mask)* image
                #save_image(torch.from_numpy(new_image).type(torch.FloatTensor), 'new_img.jpeg') #Test_image+mask
                Xr.append(new_image)
            for ip in IP:
                #mask = (ip.sum(axis=-1) == 0)[..., np.newaxis].astype(np.uint8)
                #new_image = ip + image* mask
                delta_mask = m['mask'][np.newaxis, ...]          
                mask = r[np.newaxis, ...]
                new_image = ip[np.newaxis, ...] + (1-delta_mask)*image
                Xip.append(new_image)
            count = count + 1 

        fooled_yr = 0
        avg_conf_ip = 0
        total = 0
        per_image_results = {}
        assert len(Xr) == len(Xip)
        for i, (xr, xip) in enumerate(zip(Xr, Xip)):
            #yr -> salient features
            #yip -> inert patterns
            #inv_normal_img = inv_normalize(torch.from_numpy(xr).type(torch.FloatTensor))
            #save_image(inv_normal_img, 'xr_img.jpeg') 
            #save_image(torch.from_numpy(xip).type(torch.FloatTensor), 'xip_img.jpeg')
            yr, conf_r = self.model.forward(xr, save_image=False, prediction_only=True)
            yip, conf_ip = self.model.forward(xip, save_image=False, prediction_only=True)
            #print("Prediction:", yr, yip)  
            per_image_results[i] = {
                'inert': xip, 'adversarial': xr, 'fooled': yr == adv_example_class, 'inert_conf': conf_ip,
                'adv_conf': conf_r}
            if yr == adv_example_class:
                fooled_yr += 1
            total += 1
            avg_conf_ip += conf_ip

        avg_conf_ip /= total
        return fooled_yr, avg_conf_ip, total, per_image_results

    # Algorithm 4 Decision Boundary. Combine the Fooling ratio and confidence to derive a metric  


    def run_sentinet(self, image_file, threshold, test_imgpaths, num_candidates=1, saliency='xrai', pattern='noise'):
        if saliency == 'xrai':
            saliency_func = self.model.xrai
        elif saliency == 'gradcam':
            saliency_func = self.model.gradcam
        else:
            raise ValueError('Unsupported saliency function, choose one of %s' % ["xrai, gradcam"])

        img = self.model.load_image(image_file)

        prediction, class_proposals = self.class_proposal(img, num_candidates)
        masks, mask_y = self.mask_generation(img, prediction, class_proposals, threshold=threshold, saliency_function=saliency_func)
        #print(masks)
        fooled_yr, avg_conf_ip, total, _ = self.testing(img, prediction[0], masks, test_imgpaths, pattern=pattern)
        return fooled_yr / total, avg_conf_ip


def run_wrap(sentinet, image_paths, reference_img_paths, threshold, candidates, saliency, pattern):
    results = []
    for image_path in tqdm(image_paths):
        fooled_percentage, confidence = sentinet.run_sentinet(image_path, threshold, reference_img_paths, candidates, saliency=saliency, pattern=pattern)
        results.append((fooled_percentage, confidence, image_path, pattern, saliency))
    df = pd.DataFrame(results, columns=['FoolPercentage', 'Confidence', 'FileName', "Pattern", "Saliency"])
    return df


def images_in_folder(folder, formats=("jpg", "jpeg", "png", "npy")):
    files = os.listdir(folder)
    allowed_files = list(filter(lambda x: x.split(".")[-1] in formats, files))
    allowed_filepaths = [os.path.join(folder, x) for x in allowed_files]
    return allowed_filepaths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model', choices=["lisacnn_cvpr18", "gtsrbcnn_cvpr18"])
    parser.add_argument('-t', "--reference_imgs_folder", action="store")
    parser.add_argument('-b', "--test_benign_imgs_folder", action="store")
    parser.add_argument('-a', "--test_adversarial_imgs_folder", action="store")
    #parser.add_argument('-o', "--output_folder", action="store", required=True)
    parser.add_argument('--threshold', help='Threshold for saliency', default=0.25, type=float)
    parser.add_argument('--saliency', help='Saliency algorithm to use', default='xrai', choices=["xrai", "gradcam"])
    parser.add_argument('--candidates', help='Specify the number of candidate classes', default=1, type=int) #class proposal
    parser.add_argument("-p", '--pattern', choices=["checker", "noise"], default="noise", action="store")
    args = parser.parse_args()
    
    #pytorch resnet50 model
    model = ResNet50Model(saliency = True)
    
    reference_imgs_fps = images_in_folder(args.reference_imgs_folder)
    test_benign_imgs_fps = images_in_folder(args.test_benign_imgs_folder)
    test_adversarial_imgs_fps = images_in_folder(args.test_adversarial_imgs_folder)

    sentinet = Sentinet(model)

    results = []
    
    # benign images
    df = run_wrap(sentinet, test_benign_imgs_fps, reference_imgs_fps, args.threshold, args.candidates, args.saliency, args.pattern)
    df.to_csv('resnet50_benign_results.csv')
    
    # adv images
    df = run_wrap(sentinet, test_adversarial_imgs_fps, reference_imgs_fps, args.threshold, args.candidates, args.saliency, args.pattern)
    df.to_csv('resnet50_adversarial_results.csv')
