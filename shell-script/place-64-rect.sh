#!/bin/bash 

python place-create_patched_input_with_saliency.py --noise_percentage 0.07  \
  --patch_file ./patch_files/place_rectangle_best_org_patch_007_64.npy --target 64 \
   --patch_type rectangle --save_folder . \
   --train_data_dir ./place-model/place_dataset/train \
   --test_data_dir ./place-model/place_dataset/test \
    --train_size 2000 --test_size 3000 


python feature_transfer.py --radius 51 --saliency_folder place_rectangle_64_npy_test_adv_saliency_007 \
    --img_folder place_rectangle_64_npy_test_adv_result_007 --adv_input 1 \
    --held_out_input_folder ./place-model/place_hold_out_inputs --held_out_saliency ./place-model/place_hold_out_saliency \
    --noise_percentage 0.07  --target 64 --patch_type rectangle \
    --num_of_feature_trans 1 --random_seed 10 --dataset place --save_folder .

python feature_transfer.py --radius 51 --saliency_folder place_rectangle_64_npy_test_adv_saliency_007 \
    --img_folder place_rectangle_64_npy_test_adv_result_007 --adv_input 1 \
    --held_out_input_folder ./place-model/place_hold_out_inputs --held_out_saliency ./place-model/place_hold_out_saliency \
    --noise_percentage 0.07  --target 64 --patch_type rectangle \
    --num_of_feature_trans 2 --random_seed 20 --dataset place --save_folder .


python feature_transfer.py --radius 51 --saliency_folder place_rectangle_64_npy_test_org_saliency_007 \
    --img_folder place_rectangle_64_npy_test_org_result_007 --adv_input 0 \
    --held_out_input_folder ./place-model/place_hold_out_inputs --held_out_saliency ./place-model/place_hold_out_saliency \
    --noise_percentage 0.07  --target 64 --patch_type rectangle \
    --num_of_feature_trans 1 --random_seed 10 --dataset place --save_folder .

python feature_transfer.py --radius 51 --saliency_folder place_rectangle_64_npy_test_org_saliency_007 \
    --img_folder place_rectangle_64_npy_test_org_result_007 --adv_input 0 \
    --held_out_input_folder ./place-model/place_hold_out_inputs --held_out_saliency ./place-model/place_hold_out_saliency \
    --noise_percentage 0.07  --target 64 --patch_type rectangle \
    --num_of_feature_trans 2 --random_seed 20 --dataset place --save_folder .


echo "===> Detection Output"
python place-detection.py --is_adv 1 --num_input 0 --img_folder place_64_rectangle_1feature_transfer_adv_comp_007  \
        --source_folder place_rectangle_64_npy_test_adv_result_007 --noise_percentage 0.07 \
        --radius 51 --second_folder place_64_rectangle_2feature_transfer_adv_comp_007 --target 64 \
        --save_folder ./ --patch_type rectangle --model_path ./place-model/resnet50_places365.pth.tar


python place-detection.py --is_adv 0 --num_input 0 --img_folder place_64_rectangle_1feature_transfer_org_comp_007  \
        --source_folder place_rectangle_64_npy_test_org_result_007 --noise_percentage 0.07 \
        --radius 51 --second_folder place_64_rectangle_2feature_transfer_org_comp_007 --target 64 \
        --save_folder ./ --patch_type rectangle --model_path ./place-model/resnet50_places365.pth.tar





#!/bin/sh
python get_coordinate.py --saliency_folder ./place_rectangle_64_npy_test_adv_saliency_007 \
                                          --img_folder ./place_64_rectangle_r51_adv_test_detected_007 
python get_coordinate.py --saliency_folder ./place_rectangle_64_npy_test_org_saliency_007 \
                                          --img_folder ./place_64_rectangle_r51_org_test_misdetected_007

cd ./Pluralistic-Inpainting 
python test.py --name place_random --mask_type 4 --img_file ../place_64_rectangle_r51_adv_test_detected_007_with_coordinate \
            --results_dir ./place_rectangle_64_r51_adv_test_detected_007_100/ --mask_pert 1. --is_square_patch 0 --batchSize 16 
python test.py --name place_random --mask_type 4 --img_file ../place_64_rectangle_r51_org_test_misdetected_007_with_coordinate \
            --results_dir ./place_rectangle_64_r51_org_test_misdetected_007_100/ --mask_pert 1. --is_square_patch 0 --batchSize 16


echo "===> Mitigation Output"
cd ../ 
python place-mitigation.py --img_folder Pluralistic-Inpainting/place_rectangle_64_r51_adv_test_detected_007_100 \
    --input_tag out_0 --normalize 1 --input_size 0 --extract_label 1 --model_path ./place-model/resnet50_places365.pth.tar
    
python place-mitigation.py --img_folder Pluralistic-Inpainting/place_rectangle_64_r51_org_test_misdetected_007_100 \
    --input_tag out_0 --normalize 1 --input_size 0 --extract_label 0  --model_path ./place-model/resnet50_places365.pth.tar


rm -rf place_rectangle_64_npy_test_adv_saliency_007
rm -rf place_rectangle_64_npy_test_adv_result_007
rm -rf place_rectangle_64_npy_test_org_saliency_007
rm -rf place_rectangle_64_npy_test_org_result_007
rm -rf place_64_rectangle_2feature_transfer_adv_comp_007
rm -rf place_64_rectangle_1feature_transfer_adv_comp_007
rm -rf place_64_rectangle_2feature_transfer_org_comp_007
rm -rf place_64_rectangle_1feature_transfer_org_comp_007

rm -rf place_64_rectangle_r51_adv_test_detected_007_with_coordinate
rm -rf place_64_rectangle_r51_adv_test_detected_007
rm -rf place_64_rectangle_r51_org_test_misdetected_007_with_coordinate
rm -rf place_64_rectangle_r51_org_test_misdetected_007

rm -rf Pluralistic-Inpainting/place_rectangle_64_r51_adv_test_detected_007_100
rm -rf Pluralistic-Inpainting/place_rectangle_64_r51_org_test_misdetected_007_100





