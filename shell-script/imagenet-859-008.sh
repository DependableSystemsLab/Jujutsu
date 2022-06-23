#!/bin/bash 

#python imagenet-attack.py --target 859 --noise_percentage 0.08 \
#              --train_size 2000 --model resnet50 \
#              --epochs 10 --data_dir ./imagenet-model/datasets/IMAGENET-UNCROPPED/val/ \
#              --patch_type square  --test_size 10000 



python imagenet-create_patched_input_with_saliency.py --noise_percentage 0.08  \
  --patch_file patch_files/imagenet_square_best_org_patch_008_859.npy --target 859 \
   --patch_type square --save_folder . \
   --model resnet50 \
   --data_dir ./imagenet-model/datasets/IMAGENET-UNCROPPED/val/ --test_size 10000


python feature_transfer.py --radius 51 --saliency_folder imagenet_square_859_npy_test_adv_saliency_008 \
    --img_folder imagenet_square_859_npy_test_adv_result_008 --adv_input 1 \
    --held_out_input_folder ./imagenet-model/hold_out_inputs --held_out_saliency ./imagenet-model/hold_out_saliency \
    --noise_percentage 0.08  --target 859 --patch_type square \
    --num_of_feature_trans 1 --random_seed 10 --dataset imagenet --save_folder .

python feature_transfer.py --radius 51 --saliency_folder imagenet_square_859_npy_test_adv_saliency_008 \
    --img_folder imagenet_square_859_npy_test_adv_result_008 --adv_input 1 \
    --held_out_input_folder ./imagenet-model/hold_out_inputs --held_out_saliency ./imagenet-model/hold_out_saliency \
    --noise_percentage 0.08  --target 859 --patch_type square \
    --num_of_feature_trans 2 --random_seed 20 --dataset imagenet --save_folder .


python feature_transfer.py --radius 51 --saliency_folder imagenet_square_859_npy_test_org_saliency_008 \
    --img_folder imagenet_square_859_npy_test_org_result_008 --adv_input 0 \
    --held_out_input_folder ./imagenet-model/hold_out_inputs --held_out_saliency ./imagenet-model/hold_out_saliency \
    --noise_percentage 0.08  --target 859 --patch_type square \
    --num_of_feature_trans 1 --random_seed 10 --dataset imagenet --save_folder .
python feature_transfer.py --radius 51 --saliency_folder imagenet_square_859_npy_test_org_saliency_008 \
    --img_folder imagenet_square_859_npy_test_org_result_008 --adv_input 0 \
    --held_out_input_folder ./imagenet-model/hold_out_inputs --held_out_saliency ./imagenet-model/hold_out_saliency \
    --noise_percentage 0.08  --target 859 --patch_type square \
    --num_of_feature_trans 2 --random_seed 20 --dataset imagenet --save_folder .


echo "===> Detection Output"
python imagenet-detection.py --is_adv 1 --num_input 0 --img_folder imagenet_859_square_1feature_transfer_adv_comp_008  \
        --source_folder imagenet_square_859_npy_test_adv_result_008 --noise_percentage 0.08 \
        --radius 51 --second_folder imagenet_859_square_2feature_transfer_adv_comp_008 --target 859 \
        --save_folder ./ --patch_type square --model resnet50


python imagenet-detection.py --is_adv 0 --num_input 0 --img_folder imagenet_859_square_1feature_transfer_org_comp_008  \
        --source_folder imagenet_square_859_npy_test_org_result_008 --noise_percentage 0.08 \
        --radius 51 --second_folder imagenet_859_square_2feature_transfer_org_comp_008 --target 859 \
        --save_folder ./ --patch_type square --model resnet50






#!/bin/sh
python get_coordinate.py --saliency_folder ./imagenet_square_859_npy_test_adv_saliency_008 \
                                          --img_folder ./imagenet_859_square_r51_adv_test_detected_008 
python get_coordinate.py --saliency_folder ./imagenet_square_859_npy_test_org_saliency_008 \
                                          --img_folder ./imagenet_859_square_r51_org_test_misdetected_008

cd ./Pluralistic-Inpainting 
python test.py --gpu_ids 0 --name imagenet_random --mask_type 4 --img_file ../imagenet_859_square_r51_adv_test_detected_008_with_coordinate \
            --results_dir ./imagenet_square_859_r51_adv_test_detected_008_100/ --mask_pert 1. --is_square_patch 1 --batchSize 16 
python test.py --gpu_ids 0 --name imagenet_random --mask_type 4 --img_file ../imagenet_859_square_r51_org_test_misdetected_008_with_coordinate \
            --results_dir ./imagenet_square_859_r51_org_test_misdetected_008_100/ --mask_pert 1. --is_square_patch 1 --batchSize 16


echo "===> Mitigation Output"
cd ../ 
python imagenet-mitigation.py --img_folder Pluralistic-Inpainting/imagenet_square_859_r51_adv_test_detected_008_100 \
    --input_tag out_0 --normalize 1 --input_size 0 --extract_label 1   --model resnet50

python imagenet-mitigation.py --img_folder Pluralistic-Inpainting/imagenet_square_859_r51_org_test_misdetected_008_100 \
    --input_tag out_0 --normalize 1 --input_size 0 --extract_label 0   --model resnet50


rm -rf imagenet_square_859_npy_test_adv_saliency_008
rm -rf imagenet_square_859_npy_test_adv_result_008
rm -rf imagenet_square_859_npy_test_org_saliency_008
rm -rf imagenet_square_859_npy_test_org_result_008
rm -rf imagenet_859_square_2feature_transfer_adv_comp_008
rm -rf imagenet_859_square_1feature_transfer_adv_comp_008
rm -rf imagenet_859_square_2feature_transfer_org_comp_008
rm -rf imagenet_859_square_1feature_transfer_org_comp_008

rm -rf imagenet_859_square_r51_adv_test_detected_008_with_coordinate
rm -rf imagenet_859_square_r51_adv_test_detected_008
rm -rf imagenet_859_square_r51_org_test_misdetected_008_with_coordinate
rm -rf imagenet_859_square_r51_org_test_misdetected_008

rm -rf Pluralistic-Inpainting/imagenet_square_859_r51_adv_test_detected_008_100
rm -rf Pluralistic-Inpainting/imagenet_square_859_r51_org_test_misdetected_008_100






