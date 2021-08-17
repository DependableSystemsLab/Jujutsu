# Code for paper "Turning Your Strength against You: Detecting and Mitigating Robust and Universal Adversarial Patch Attack" link to paper

This paper introduces a technique to *detect and mitigation* robust and universal adversarial patch attack. Below demonstrates the main ideas behind our technique:

## Attack Detection
![Alt](./technique_figures/detection.jpg "Attack detection")

<br/>

## Attack Mitigation
![Alt](./technique_figures/mitigation.jpg "Attack mitigation")


## Requirements
- PyTorch (1.2.0 or higher)
- Torch-vision (0.2.1 or higher)
- numpy
- scipy
- tpdm
- OpenCV 
- visdom 
- dominate
- teste on Python 3.7.3

## Directory
- Root directory contains the code for our technique
- ```patch_files```: adversarial patches from different DNNs. Sample patches are provided
- ```Pluralistic-Inpainting```: Code for performing image inpainting
- ```adaptive_attack```: code for conducting adaptive attacks against our defence
- ```related_defenses```: code for the related defences evaluated in our paper
- ```held_out_inputs```: Sample hold-out images. 

## How to run 

### 1. Generate adversarial patch

```
python attack.py --noise_percentage 0.06 --GPU 0 --model resnet50 --target 859 --patch_type square --train_size 2000 --test_size 2000 --epoch 30 --data_dir /path_to_data_dir/
```

- ```--noise_percentage```: size of adversarial patch (e.g., 0.06 means a patch occupying 6% of image pixels)
- ```--target```: target label 
- ```--patch_type```: shape of the patch (square or rectangle)
- ```--data_dir```: image folder
- ```--train_size```: num of images to train the adversarial patch
- ```--test_size```: num of images to test the adversarial patch, i.e., measure the attack success rate
- ```--epoch 30```: epoch to train the patch 


Running the above will generate a npy file ```resnet50_best_org_patch_006.npy```, which is the patch file and you can put it into random images to create adversarial samples (next step).

We provide the generated patches from 5 DNNs in both square and rectangular shape in ```./patch_files```.

### 2. Generate adversarial samples using the adversarial patch and derive their saliency maps

```
python create_patched_input_with_saliency.py --noise_percentage 0.06 --GPU 0 --model resnet50 --patch_file patch_files/square/resnet50_best_org_patch_006.npy --target 859 --data_dir /path_to_data_dir/ --patch_type square --train_size 2000 --test_size 2000
```

- ```--test_size```: number of adversarial samples to generate
- ```--patch_file```: path to the file of adversarial patch

This will generate both the adversarial samples and their saliency maps, each saved in a different folder. The image files are saved in .npy file and each image has the following naming format: ```xx_yy.npy```, where *xx* is the index of the image and *yy* is the ground truth label of the image.

Running the above will generate 4 folders:

-  ```resnet50_square_859_npy_test_adv_result_006```: adversarial samples with a  6% square patch whose target label is 859 on ResNet50
-  ```resnet50_square_859_npy_test_adv_saliency_006```: saliency maps for the adversarial samples
-  ```resnet50_square_859_npy_test_org_result_006```: benign images that are the same as the adversarial samples except not being perturbed by adversarial patch
-  ```resnet50_square_859_npy_test_org_saliency_006```: saliency maps for the benign images


### 3. Prepare held-out images and their saliency maps

Hold-out images are some random images from the dataset (e.g, you can draw 1000 random images from the dataset). Some sample images are in ```./held_out_inputs```

Next, we need to generate the saliency maps for these held-out images.

```
python get_saliency.py --model resnet50 --img_folder held_out_inputs --GPU 0 --output_folder held_out_saliency
```

- ```--output_folder```: output folder to save the saliency maps. 

Running the above will generate a folder called ```held_out_saliency```

### 4. Perform feature transplantation 

```
python feature_transfer.py --radius 51 --saliency_folder resnet50_square_859_npy_test_adv_saliency_006 \
    --img_folder resnet50_square_859_npy_test_adv_result_006 --adv_input 1 \
    --held_out_input_folder held_out_inputs --held_out_saliency held_out_saliency \
    --noise_percentage 0.06 --model resnet50 --target 859 --patch_type square \
    --num_of_feature_trans 1 --random_seed 10

python feature_transfer.py --radius 51 --saliency_folder resnet50_square_859_npy_test_adv_saliency_006 \
    --img_folder resnet50_square_859_npy_test_adv_result_006 --adv_input 1 \
    --held_out_input_folder held_out_inputs --held_out_saliency held_out_saliency \
    --noise_percentage 0.06 --model resnet50 --target 859 --patch_type square \
    --num_of_feature_trans 2 --random_seed 20
```

- ```--radius```: parameter for performing average filter over the saliency map and for extracting the salient features in the saliency map.
- ```--adv_input```: indicate whether the source images are adversarial or not. This is for labeling the output folder only.
- ```--noise_percentage```: indicate the size of the adversarial patch in the images. This is for labeling the output folder only.
- ```--num_of_feature_trans```: a tag to label the output folder only.
- ```--random_seed```: random seed for initializing the random number generator.

Running the above will generate 2 folders:

- ```resnet50_859_1feature_transfer_adv_comp_006```: A set of random held-out images on which we transplanted the salient features from the source images. Source images are adversarial samples
- ```resnet50_859_2feature_transfer_adv_comp_006```: Another set of random held-out images on which we transplanted the salient features from the source image

You should also run the following 2 commands for the benign images:

```
python feature_transfer.py --radius 51 --saliency_folder resnet50_square_859_npy_test_org_saliency_006 \
    --img_folder resnet50_square_859_npy_test_org_result_006 --adv_input 0 \
    --held_out_input_folder held_out_inputs --held_out_saliency held_out_saliency \
    --noise_percentage 0.06 --model resnet50 --target 859 --patch_type square \
    --num_of_feature_trans 1 --random_seed 10

python feature_transfer.py --radius 51 --saliency_folder resnet50_square_859_npy_test_org_saliency_006 \
    --img_folder resnet50_square_859_npy_test_org_result_006 --adv_input 0 \
    --held_out_input_folder held_out_inputs --held_out_saliency held_out_saliency \
    --noise_percentage 0.06 --model resnet50 --target 859 --patch_type square \
    --num_of_feature_trans 2 --random_seed 20
```

This will generate 2 folders:

- ```resnet50_859_1feature_transfer_org_comp_006```: A set of random held-out images on which we transplanted the salient features from the source image. Source images are benign samples.
- ```resnet50_859_2feature_transfer_org_comp_006```: Another set of random held-out images on which we transplanted the salient features from the source image.


### 5. Attack detection based on prediction consistency

```
python detection.py --is_adv 1 --num_input 0 --img_folder resnet50_859_1feature_transfer_adv_comp_006  \
        --source_folder resnet50_square_859_npy_test_adv_result_006 --noise_percentage 0.06 \
        --model resnet50 --radius 51 --second_folder resnet50_859_2feature_transfer_adv_comp_006
```

- ```--source_folder```: The test images that we want to test whether they are adversarial or not
- ```--img_folder```: first folder that contains the hold-out images transplanted with salient features from the test images
- ```--second_folder```: second folder that contains *another* set of hold-out images transplanted with salient features from the test images
- ```--radius```: a tag for labeling the output folder
- ```--noise_percentage```: a tag for labeling the output folder

Running the above will generate a folder:

- ```resnet50_r51_adv_test_detected_006```: These are the images detected as adversarial

You should also run the following command for the benign images:

```
python detection.py --is_adv 0 --num_input 0 --img_folder resnet50_859_1feature_transfer_org_comp_006  \
        --source_folder resnet50_square_859_npy_test_org_result_006 --noise_percentage 0.06 \
        --model resnet50 --radius 51 --second_folder resnet50_859_2feature_transfer_org_comp_006
```
This will generate a folder:
- ```resnet50_r51_org_test_misdetected_006```: These are the images *incorrectly* detected as adversarial, i.e., false positive



### 6. Performing image inpainting on the test images

#### 6.1 Identify the image regions for inpainting 

```
python get_coordinate.py --saliency_folder resnet50_square_859_npy_test_adv_saliency_006 --img_folder resnet50_r51_adv_test_detected_006
```
- ```--img_folder```: Test images on which we want to perform image inpainting
- ```--saliency_folder```: Saliency maps of the test images, used for locating the regions to perform inpainting

This will generate a folder:
- ```resnet50_r51_adv_test_detected_006_with_coordinate```: These are the same images as the test images but we add the coordinate (pp,qq) to the name of the image file. E.g., ```xx_yy=pp=qq.npy```, where (pp,qq) is the coordinate. The coordinate is used to locate the regions to perform inpainting

You should also run the following command for the benign images:
```
python get_coordinate.py --saliency_folder resnet50_square_859_npy_test_org_saliency_006 --img_folder resnet50_r51_org_test_misdetected_006
```
This will generate a folder:
- ```resnet50_r51_org_test_misdetected_006_with_coordinate```


#### 6.2 Perform image inpainting
Go to ```./Pluralistic-Inpainting``` and run:

```
python test.py --name imagenet_random --mask_type 4 --img_file ../resnet50_r51_adv_test_detected_006_with_coordinate \
            --results_dir ./resnet50_r51_adv_test_detected_006/ --mask_pert 1. --is_square_patch 1

python test.py --name imagenet_random --mask_type 4 --img_file ../resnet50_r51_org_test_misdetected_006_with_coordinate \
            --results_dir ./resnet50_r51_org_test_misdetected_006/ --mask_pert 1. --is_square_patch 1        
```

- ```--img_file```: images on which we perform inpainting
- ```--results_dir```: output directory for the inpainting output. For each image, it will generate 3 different files: 1) the original image (```xx_yy=pp=qq_truth.npy```); 2) the image placed with a mask (```xx_yy=pp=qq_mask.npy```); 3) the image after being inpainted (```xx_yy=pp=qq_out_0.npy```)
- ```--mask_pert```: percentage of pixels to mask. E.g., 0.5 means we mask 50% of the pixels within the salient feature region.
- ```--is_square_patch```: indicate whether to draw a square mask or rectangular mask.

This will generate 2 output folders:
- ```resnet50_r51_adv_test_detected_006```
- ```resnet50_r51_org_test_misdetected_006```


### 7. Attack mitigation and reduce false positive

#### 7.1 Attack mitigation

Go back to the main directory and run:
```
python mitigation.py --img_folder Pluralistic-Inpainting/resnet50_r51_adv_test_detected_006 \
    --input_tag out_0 --normalize 1 --input_size 0 --extract_label 1 --model resnet50 
```
- ```--img_folder```: output folder from image inpainting (step 6.2)
- ```--input_tag```: set ```out_0``` if you want to compare the original image with the *inpainted* image; set ```mask``` if you want to compare the original image with the *masked image without inpainting*
- ```--normalize```: whether to perform image normalization before inference
- ```--extract_label```: whether to extract the ground-truth label, which can be used to determine whether the mitigation is successful or not.

#### 7.2. Reduce false positive
You should also run the following command for the benign images that are *incorrectly* detected as adversarial:
```
python mitigation.py --img_folder Pluralistic-Inpainting/resnet50_r51_org_test_misdetected_006 \
    --input_tag out_0 --normalize 1 --input_size 0 --extract_label 0 --model resnet50 
```
- In this case we set ```--extract_label 0``` because we want to know if the prediction label changes after masking/inpainting regardless of the ground-truth label. If the label *does not* change, we can determine this is a benign image.

#### 7.3. Compute the amount of detected adversarial samples that will be *incorrectly* flagged as benign 
```
python mitigation.py --img_folder Pluralistic-Inpainting/resnet50_r51_adv_test_detected_006 \
    --input_tag out_0 --normalize 1 --input_size 0 --extract_label 0 --model resnet50 
```
- In this case we set ```--extract_label 0``` because we want to know if the prediction label changes after masking/inpainting regardless of the ground-truth label. If the label *does not* change, we would *falsely* determine this is a benign image, and thus the adversarial sample will be *incorrectly* flagged as benign.


# Acknowledgment
We acknowledge the use of codes from the following repositories:
- https://github.com/A-LinCui/Adversarial_Patch_Attack
- https://github.com/kazuto1011/smoothgrad-pytorch
- https://github.com/lyndonzheng/Pluralistic-Inpainting
- https://github.com/garrisongys/STRIP
- https://github.com/metallurk/local_gradients_smoothing


# Citation
If you find this code useful, please consider citing our paper

```
@article{chen2021turning,
  title={Turning Your Strength against You: Detecting and Mitigating Robust and Universal Adversarial Patch Attack},
  author={Chen, Zitao and Dash, Pritam and Pattabiraman, Karthik},
  journal={arXiv preprint arXiv:2108.05075},
  year={2021}
}
```










