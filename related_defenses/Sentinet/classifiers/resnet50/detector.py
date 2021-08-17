import cv2
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from classifiers import classifiers_utils as clfutils
from classifiers.Detector import Detector
import saliency.tf1 as saliency

#import torch
#from  torchvision import datasets, transforms
#from torchvision.utils import save_image

tf.compat.v1.disable_eager_execution()

PARAMS = {
    "resnet50": {
        "meta_graph_path": "./models/resnet50.pb",
        "input_tensor_name": "input.1:0",
        "output_tensor_name": "495:0",
        "last_conv_name": "Add_52:0",
        "last_fc_name": "mul:0",
        "add_feed_tensors": {},
    }, 
    "resnet152":{
        "meta_graph_path": "./models/resnet152.pb",
        "input_tensor_name": "input.1:0",
        "output_tensor_name": "1447:0",
        "last_conv_name": "Add_154:0",
        "last_fc_name": "mul:0",
        "add_feed_tensors": {},
    }, 
    "densenet":{
        "meta_graph_path": "./models/densenet121.pb",
        "input_tensor_name": "input.1:0",
        "output_tensor_name": "1161:0",
        "last_conv_name": "concat_119/concat:0",
        "last_fc_name": "mul:0",
        "add_feed_tensors": {},
    }, 
    "squeezenet":{
        "meta_graph_path": "./models/squeezenet.pb",
        "input_tensor_name": "input.1:0",
        "output_tensor_name": "117:0",
        "last_conv_name": "Add_25:0",
        "last_fc_name": "Mean:0", 
        "add_feed_tensors": {},
    }, 
    "vggnet":{
        "meta_graph_path": "./models/vggnet.pb",
        "input_tensor_name": "input.1:0",
        "output_tensor_name": "148:0",
        "last_conv_name": "Add_12:0",
        "last_fc_name": "mul_4:0",
        "add_feed_tensors": {},
    }
}

class ResNet50Model(Detector):

    def __init__(self, saliency=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # select appropriate network  
        self.p = PARAMS["vggnet"]
        p = self.p
        # output labels
        #self.class_dict = {k: v for k, v in pd.read_csv(p["class_descr_path"], index_col=False).values}

        # setup session and graph
        with tf.gfile.FastGFile(p["meta_graph_path"], "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def, name="")
        
        self.sess = tf.Session(graph=g_in)

        # link tensors
        self.input = tf.compat.v1.get_default_graph().get_tensor_by_name(p["input_tensor_name"])
        self.output = tf.compat.v1.get_default_graph().get_tensor_by_name(p["output_tensor_name"])
        self.last_conv = tf.compat.v1.get_default_graph().get_tensor_by_name(p["last_conv_name"])
        self.last_fc = tf.compat.v1.get_default_graph().get_tensor_by_name(p["last_fc_name"])
        self.feed_tensors = p["add_feed_tensors"]
        self.prob = self.output
        #self.preprocess_input = p["preprocess_input"]
        #self.adjust_input_for_plot = p["adjust_input_for_plot"]
        
        # saliency setup
        self.saliency = False
        if saliency:
            self.saliency = True
            self.neuron_selector = tf.compat.v1.placeholder(tf.int32)
            self.saliency_target = self.last_fc[0][self.neuron_selector]
            self.prediction = tf.argmax(self.last_fc, 1)

    @staticmethod
    def load_image(path):
        #img = cv2.imread(path)[..., [2, 1, 0]]
        #img = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
        img = np.load(path)
        return img
    
    @staticmethod
    def load_image_jpg(path):
        img = cv2.imread(path)
        print(img.shape)
        img = cv2.resize(img, (224,224), cv2.INTER_LINEAR)
        return img

    def needs_roi(self):
        return True

    def xrai(self, image, prediction_class, binarize=False, threshold=0.3):
        
        #image = self.preprocess_input(image) # npy is already preprocessed
        if not hasattr(self, 'xrai_object'):
            self.xrai_object = saliency.XRAI(tf.compat.v1.get_default_graph(), self.sess, self.saliency_target, self.input)
        image = image[0]
        xrai_params = saliency.XRAIParameters()
        xrai_params.algorithm = 'fast'
        feed_dict = {**dict({self.neuron_selector: prediction_class}), **self.feed_tensors}
        xrai_attributions = self.xrai_object.GetMask(image, feed_dict=feed_dict, extra_parameters=xrai_params)
        print("XRAI:", xrai_attributions.shape)
        cv2.imwrite(str(prediction_class)+"xrai_attribute.jpg", xrai_attributions)
        # most salient 30%
        xrai_salient_mask = xrai_attributions > np.percentile(xrai_attributions, (1-threshold)*100)
        xrai_im_mask = np.ones(image.shape)
        xrai_im_mask[~xrai_salient_mask] = 0
        print("mask:", xrai_im_mask.shape)
        save_image(torch.from_numpy(xrai_im_mask).type(torch.FloatTensor), str(prediction_class) +'saliency_mask_threshold_final.jpeg')

        if binarize:
            xrai_im_mask = (xrai_im_mask > 0).astype(bool)
        return xrai_im_mask
    
    def gradcam(self, image, prediction_class, binarize=False, threshold=0.3):
        image = image[0]
        #image = np.transpose(image, (1,2,0))
        if not hasattr(self, 'gradcam_object'):
            # construct saliency object
            self.gradcam_object = saliency.GradCam(tf.compat.v1.get_default_graph(), self.sess, self.saliency_target, self.input, self.last_conv)
        
        feed_dict = {**dict({self.neuron_selector: prediction_class}), **self.feed_tensors}
        gradcam_mask = self.gradcam_object.GetMask(image, feed_dict=feed_dict)

        #cv2.imwrite(str(prediction_class)+"original_mask.jpeg", gradcam_mask)
        gradcam_mask = np.transpose(gradcam_mask, (2, 0, 1))
        
        #saving saliency mask
        #save_image(torch.from_numpy(gradcam_mask).type(torch.FloatTensor), str(prediction_class) + 'saliency_mask.jpeg')
        # most salient 30%
        gradcam_salient_mask = gradcam_mask > np.percentile(gradcam_mask, (1-threshold)*100)
        
        gradcam_im_mask = np.ones(image.shape)
        gradcam_im_mask[~gradcam_salient_mask] = 0  
        #save_image(torch.from_numpy(gradcam_im_mask).type(torch.FloatTensor), str(prediction_class) +'saliency_mask_threshold_final.jpeg')

        if binarize:
            gradcam_im_mask = (gradcam_im_mask > 0).astype(bool)
        #save_image(torch.from_numpy(gradcam_im_mask).type(torch.FloatTensor), str(prediction_class) +'saliency_mask_threshold_binarize.jpeg')
        return gradcam_im_mask # Shape 3, 224, 224
        
    def forward(self, img_ori, tracker_box=None, tracker_pad=None, tracker_min_pad=None, save_image=True, probs_only=False, prediction_only=False):
        cutout = np.copy(img_ori)
        
        #cutout_r = cv2.resize(cutout, (224, 224), cv2.INTER_LINEAR) # npy is already 224X224
        #cutout_r_preproc = self.preprocess_input(cutout_r) # npy does not need preprocess
        cutout_r_preproc = cutout
        #feed_dict = {**dict({self.input: cutout_r_preproc[np.newaxis, ...]}), **self.feed_tensors}
        feed_dict = {**dict({self.input: cutout_r_preproc}), **self.feed_tensors}
        
        labels_out_ = self.sess.run(self.output, feed_dict=feed_dict)

        if prediction_only:
            prediction = np.argmax(labels_out_[0])
            return prediction, labels_out_[0][prediction]

        if probs_only:
            return np.array(labels_out_[0])

