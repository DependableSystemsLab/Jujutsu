# Detailed usage of each Python file


## 1. Generate adversarial patch

```
python imagenette-attack.py --target 5 --noise_percentage 0.07 \
              --train_size 2000 --model_path /imagenette-model/resnet18-imagenette.pt \
              --epochs 30 --data_dir ./imagenette-model/imagenette2-320 \
              --patch_type square 
```

- ```--noise_percentage```: size of adversarial patch (e.g., 0.06 means a patch occupying 6% of image pixels)
- ```--target```: target label 
- ```--patch_type```: shape of the patch (square or rectangle)
- ```--data_dir```: image folder
- ```--train_size```: num of images to train the adversarial patch 
- ```--epoch```: epoch to train the patch 

## 2. Generate adversarial samples using the adversarial patch and derive their saliency maps

```
python imagenette_create_patched_input_with_saliency.py --noise_percentage 0.07  \
  --patch_file patch_files/imagenette_square_best_org_patch_007_5.npy --target 5 \
   --patch_type square \
   --model_path ./imagenette-model/resnet18-imagenette.pt \
   --data_dir ./imagenette-model/imagenette2-320 --test_size 3500
```

- ```--test_size```: number of adversarial samples to generate
- ```--patch_file```: path to the file of adversarial patch
- ```model_path```: path to the model's checkpoint


## 3. Prepare held-out images and their saliency maps

```
python get_saliency.py --model resnet18 \
    --img_folder ./imagenette-model/imagenette_hold_out \
    --output_folder ./imagenette-model/imagenette_hold_out_saliency \
    --dataset imagenette --model_path ./imagenette-model/resnet18-imagenette.pt
```

- ```--img_folder```: folder containing the hold-out images. 
- ```--output_folder```: output folder to save the saliency maps. 
- ```--dataset```: name of the dataset

## 4. Perform feature transplantation 

```
python feature_transfer.py --radius 51 --saliency_folder imagenette_square_5_npy_test_adv_saliency_007 \
    --img_folder imagenette_square_5_npy_test_adv_result_007 --adv_input 1 \
    --held_out_input_folder ./imagenette-model/imagenette_hold_out --held_out_saliency ./imagenette-model/imagenette_hold_out_saliency \
    --noise_percentage 0.07  --target 5 --patch_type square \
    --num_of_feature_trans 1 --random_seed 10 --dataset imagenette
```
- ```--radius```: parameter for performing average filter over the saliency map and for extracting the salient features in the saliency map.
- ```--adv_input```: indicate whether the source images are adversarial or not. This is for labeling the output folder only.
- ```--noise_percentage```: indicate the size of the adversarial patch in the images. This is for labeling the output folder only.
- ```--num_of_feature_trans```: a tag to label the output folder only.
- ```--random_seed```: random seed for initializing the random number generator.


## 5. Attack detection based on prediction consistency

```
python imagenette-detection.py --is_adv 1 --num_input 0 --img_folder imagenette_5_square_1feature_transfer_adv_comp_007  \
        --source_folder imagenette_square_5_npy_test_adv_result_007 --noise_percentage 0.07 \
        --radius 51 --second_folder imagenette_5_square_2feature_transfer_adv_comp_007 --target 5 \
        --save_folder ./ --patch_type square --model_path ./imagenette-model/resnet18-imagenette.pt
```

- ```--source_folder```: The test images that we want to test whether they are adversarial or not
- ```--img_folder```: first folder that contains the hold-out images transplanted with salient features from the test images
- ```--second_folder```: second folder that contains *another* set of hold-out images transplanted with salient features from the test images
- ```--radius```: a tag for labeling the output folder
- ```--noise_percentage```: a tag for labeling the output folder



## 6. Identify the image regions for inpainting (attack mitigation)

```
python get_coordinate.py --saliency_folder ./imagenette_square_5_npy_test_adv_saliency_007 \
                                          --img_folder ./imagenette_5_square_r51_adv_test_detected_007 
```
- ```--img_folder```: Test images on which we want to perform image inpainting
- ```--saliency_folder```: Saliency maps of the test images, used for locating the regions to perform inpainting



## 7. Perform image inpainting

Go to ```./Pluralistic-Inpainting``` and run:

```
python test.py --name imagenet_random --mask_type 4 --img_file ../imagenette_5_square_r51_adv_test_detected_007_with_coordinate \
            --results_dir ./imagenette_square_5_r51_adv_test_detected_007_100/ --mask_pert 1. --is_square_patch 1 --batchSize 16 
```
- ```--img_file```: images on which we perform inpainting
- ```--results_dir```: output directory for the inpainting output. For each image, it will generate 3 different files: 1) the original image (```xx_yy=pp=qq_truth.npy```); 2) the image placed with a mask (```xx_yy=pp=qq_mask.npy```); 3) the image after being inpainted (```xx_yy=pp=qq_out_0.npy```)
- ```--mask_pert```: percentage of pixels to mask. E.g., 0.5 means we mask 50% of the pixels within the salient feature region.
- ```--is_square_patch```: indicate whether to draw a square mask or rectangular mask.

## 8. Attack mitigation

Go back to the main directory and run:

```
python imagenette-mitigation.py --img_folder Pluralistic-Inpainting/imagenette_square_5_r51_adv_test_detected_007_100 \
    --input_tag out_0 --normalize 1 --input_size 0 --extract_label 1  --model_path ./imagenette-model/resnet18-imagenette.pt 

python imagenette-mitigation.py --img_folder Pluralistic-Inpainting/imagenette_square_5_r51_org_test_misdetected_007_100 \
    --input_tag out_0 --normalize 1 --input_size 0 --extract_label 0  --model_path ./imagenette-model/resnet18-imagenette.pt
```

- ```--img_folder```: output folder from image inpainting (step 6.2)
- ```--input_tag```: set ```out_0``` if you want to compare the original image with the *inpainted* image; set ```mask``` if you want to compare the original image with the *masked image without inpainting*
- ```--normalize```: whether to perform image normalization before inference
- ```--extract_label```: whether to extract the ground-truth label, which can be used to determine whether the mitigation is successful or not.
















