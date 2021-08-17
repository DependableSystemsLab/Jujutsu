# Evaluation on 4 related defences

## local gradient smoothing (https://arxiv.org/abs/1807.01216)

Go to *./local_gradients_smoothing*

```
python lgs.py --model resnet50 --benign_img_folder ../../resnet50_square_859_npy_test_org_result_006 --adv_img_folder ../../resnet50_square_859_npy_test_adv_result_006
```

- ```--benign_img_folder```: benign images to be tested for computing false positive 
- ```--adv_img_folder```: adversarial samples 


## STRIP (https://arxiv.org/abs/1902.06531)
Go to *./STRIP-master*
```
python detection.py --model resnet50 --train_img_folder ../../held_out_inputs --benign_img_folder ../../resnet50_square_859_npy_test_org_result_006 --adv_img_folder ../../resnet50_square_859_npy_test_adv_result_006 --held_out_img_folder ../../held_out_inputs
```
- ```--benign_img_folder```: benign images to be tested for computing false positive 
- ```--adv_img_folder```: adversarial samples 
- ```--train_img_folder```: a set of images to derive the detection boundary 
- ```--held_out_img_folder```: a set of images to be superimposed. Each image in ```train_img_folder``` will be blended with the images in this folder, and the resulting images will be used for computing the prediction entropy. 

## SentiNet (https://arxiv.org/abs/1812.00292)
Go to *./Sentinet*
```
python sentinet.py -t ref_img_folder -b ben_img_folder -a adv_img_folder 
```
- ```-t```: reference images to map salient features
- ```-b```: benign images to test false positives
- ```-a```: adversarial samples 


## Adversarial training 
