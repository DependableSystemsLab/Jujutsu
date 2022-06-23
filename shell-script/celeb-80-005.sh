#!/bin/bash 

python celeb-create_patched_input_with_saliency.py --noise_percentage 0.05  \
  --patch_file patch_files/celeb_square_best_org_patch_005_80.npy --target 80 \
   --patch_type square --save_folder . \
   --model_path ./celeb-model/facial_identity_ResNet18.pth \
   --data_dir ./celeb-model/CelebA_HQ_facial_identity_dataset


python feature_transfer.py --radius 51 --saliency_folder celeb_square_80_npy_test_adv_saliency_005 \
    --img_folder celeb_square_80_npy_test_adv_result_005 --adv_input 1 \
    --held_out_input_folder ./celeb-model/celeb_hold_out --held_out_saliency ./celeb-model/celeb_hold_out_saliency \
    --noise_percentage 0.05  --target 80 --patch_type square \
    --num_of_feature_trans 1 --random_seed 10 --dataset celeb --save_folder .

python feature_transfer.py --radius 51 --saliency_folder celeb_square_80_npy_test_adv_saliency_005 \
    --img_folder celeb_square_80_npy_test_adv_result_005 --adv_input 1 \
    --held_out_input_folder ./celeb-model/celeb_hold_out --held_out_saliency ./celeb-model/celeb_hold_out_saliency \
    --noise_percentage 0.05  --target 80 --patch_type square \
    --num_of_feature_trans 2 --random_seed 20 --dataset celeb --save_folder .


python feature_transfer.py --radius 51 --saliency_folder celeb_square_80_npy_test_org_saliency_005 \
    --img_folder celeb_square_80_npy_test_org_result_005 --adv_input 0 \
    --held_out_input_folder ./celeb-model/celeb_hold_out --held_out_saliency ./celeb-model/celeb_hold_out_saliency \
    --noise_percentage 0.05  --target 80 --patch_type square \
    --num_of_feature_trans 1 --random_seed 10 --dataset celeb --save_folder .

python feature_transfer.py --radius 51 --saliency_folder celeb_square_80_npy_test_org_saliency_005 \
    --img_folder celeb_square_80_npy_test_org_result_005 --adv_input 0 \
    --held_out_input_folder ./celeb-model/celeb_hold_out --held_out_saliency ./celeb-model/celeb_hold_out_saliency \
    --noise_percentage 0.05  --target 80 --patch_type square \
    --num_of_feature_trans 2 --random_seed 20 --dataset celeb --save_folder .



echo "===> Detection Output"
python celeb-detection.py --is_adv 1 --num_input 0 --img_folder celeb_80_square_1feature_transfer_adv_comp_005  \
        --source_folder celeb_square_80_npy_test_adv_result_005 --noise_percentage 0.05 \
        --radius 51 --second_folder celeb_80_square_2feature_transfer_adv_comp_005 --target 80 \
        --save_folder ./ --patch_type square --model_path ./celeb-model/facial_identity_ResNet18.pth


python celeb-detection.py --is_adv 0 --num_input 0 --img_folder celeb_80_square_1feature_transfer_org_comp_005  \
        --source_folder celeb_square_80_npy_test_org_result_005 --noise_percentage 0.05 \
        --radius 51 --second_folder celeb_80_square_2feature_transfer_org_comp_005 --target 80 \
        --save_folder ./ --patch_type square --model_path ./celeb-model/facial_identity_ResNet18.pth



#!/bin/sh
python get_coordinate.py --saliency_folder ./celeb_square_80_npy_test_adv_saliency_005 \
                                          --img_folder ./celeb_80_square_r51_adv_test_detected_005 
python get_coordinate.py --saliency_folder ./celeb_square_80_npy_test_org_saliency_005 \
                                          --img_folder ./celeb_80_square_r51_org_test_misdetected_005

cd ./Pluralistic-Inpainting 
python test.py --name celeb_random --mask_type 4 --img_file ../celeb_80_square_r51_adv_test_detected_005_with_coordinate \
            --results_dir ./celeb_square_80_r51_adv_test_detected_005_100/ --mask_pert 1. --is_square_patch 1 --batchSize 16 
python test.py --name celeb_random --mask_type 4 --img_file ../celeb_80_square_r51_org_test_misdetected_005_with_coordinate \
            --results_dir ./celeb_square_80_r51_org_test_misdetected_005_100/ --mask_pert 1. --is_square_patch 1 --batchSize 16


echo "===> Mitigation Output"
cd ../ 
python celeb-mitigation.py --img_folder Pluralistic-Inpainting/celeb_square_80_r51_adv_test_detected_005_100 \
    --input_tag out_0 --normalize 1 --input_size 0 --extract_label 1   --model_path ./celeb-model/facial_identity_ResNet18.pth
    
python celeb-mitigation.py --img_folder Pluralistic-Inpainting/celeb_square_80_r51_org_test_misdetected_005_100 \
    --input_tag out_0 --normalize 1 --input_size 0 --extract_label 0   --model_path ./celeb-model/facial_identity_ResNet18.pth


rm -rf celeb_square_80_npy_test_adv_saliency_005
rm -rf celeb_square_80_npy_test_adv_result_005
rm -rf celeb_square_80_npy_test_org_saliency_005
rm -rf celeb_square_80_npy_test_org_result_005
rm -rf celeb_80_square_2feature_transfer_adv_comp_005
rm -rf celeb_80_square_1feature_transfer_adv_comp_005
rm -rf celeb_80_square_2feature_transfer_org_comp_005
rm -rf celeb_80_square_1feature_transfer_org_comp_005

rm -rf celeb_80_square_r51_adv_test_detected_005_with_coordinate
rm -rf celeb_80_square_r51_adv_test_detected_005
rm -rf celeb_80_square_r51_org_test_misdetected_005_with_coordinate
rm -rf celeb_80_square_r51_org_test_misdetected_005

rm -rf Pluralistic-Inpainting/celeb_square_80_r51_adv_test_detected_005_100
rm -rf Pluralistic-Inpainting/celeb_square_80_r51_org_test_misdetected_005_100















