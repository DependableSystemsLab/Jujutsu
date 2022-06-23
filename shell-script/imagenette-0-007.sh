#!/bin/bash 

python imagenette_create_patched_input_with_saliency.py --noise_percentage 0.07  \
  --patch_file patch_files/imagenette_square_best_org_patch_007_0.npy --target 0 \
   --patch_type square --save_folder . \
   --model_path ./imagenette-model/resnet18-imagenette.pt \
   --data_dir ./imagenette-model/imagenette2-320 --test_size 3500


python feature_transfer.py --radius 51 --saliency_folder imagenette_square_0_npy_test_adv_saliency_007 \
    --img_folder imagenette_square_0_npy_test_adv_result_007 --adv_input 1 \
    --held_out_input_folder ./imagenette-model/imagenette_hold_out --held_out_saliency ./imagenette-model/imagenette_hold_out_saliency \
    --noise_percentage 0.07  --target 0 --patch_type square \
    --num_of_feature_trans 1 --random_seed 10 --dataset imagenette --save_folder .
python feature_transfer.py --radius 51 --saliency_folder imagenette_square_0_npy_test_adv_saliency_007 \
    --img_folder imagenette_square_0_npy_test_adv_result_007 --adv_input 1 \
    --held_out_input_folder ./imagenette-model/imagenette_hold_out --held_out_saliency ./imagenette-model/imagenette_hold_out_saliency \
    --noise_percentage 0.07  --target 0 --patch_type square \
    --num_of_feature_trans 2 --random_seed 20 --dataset imagenette --save_folder .


python feature_transfer.py --radius 51 --saliency_folder imagenette_square_0_npy_test_org_saliency_007 \
    --img_folder imagenette_square_0_npy_test_org_result_007 --adv_input 0 \
    --held_out_input_folder ./imagenette-model/imagenette_hold_out --held_out_saliency ./imagenette-model/imagenette_hold_out_saliency \
    --noise_percentage 0.07  --target 0 --patch_type square \
    --num_of_feature_trans 1 --random_seed 10 --dataset imagenette --save_folder .
python feature_transfer.py --radius 51 --saliency_folder imagenette_square_0_npy_test_org_saliency_007 \
    --img_folder imagenette_square_0_npy_test_org_result_007 --adv_input 0 \
    --held_out_input_folder ./imagenette-model/imagenette_hold_out --held_out_saliency ./imagenette-model/imagenette_hold_out_saliency \
    --noise_percentage 0.07  --target 0 --patch_type square \
    --num_of_feature_trans 2 --random_seed 20 --dataset imagenette --save_folder .


echo "===> Detection Output"
python imagenette-detection.py --is_adv 1 --num_input 0 --img_folder imagenette_0_square_1feature_transfer_adv_comp_007  \
        --source_folder imagenette_square_0_npy_test_adv_result_007 --noise_percentage 0.07 \
        --radius 51 --second_folder imagenette_0_square_2feature_transfer_adv_comp_007 --target 0 \
        --save_folder ./ --patch_type square --model_path ./imagenette-model/resnet18-imagenette.pt
python imagenette-detection.py --is_adv 0 --num_input 0 --img_folder imagenette_0_square_1feature_transfer_org_comp_007  \
        --source_folder imagenette_square_0_npy_test_org_result_007 --noise_percentage 0.07 \
        --radius 51 --second_folder imagenette_0_square_2feature_transfer_org_comp_007 --target 0 \
        --save_folder ./ --patch_type square --model_path ./imagenette-model/resnet18-imagenette.pt



#!/bin/sh
python get_coordinate.py --saliency_folder ./imagenette_square_0_npy_test_adv_saliency_007 \
                                          --img_folder ./imagenette_0_square_r51_adv_test_detected_007 
python get_coordinate.py --saliency_folder ./imagenette_square_0_npy_test_org_saliency_007 \
                                          --img_folder ./imagenette_0_square_r51_org_test_misdetected_007

cd ./Pluralistic-Inpainting 
python test.py --gpu_ids 0 --name imagenet_random --mask_type 4 --img_file ../imagenette_0_square_r51_adv_test_detected_007_with_coordinate \
            --results_dir ./imagenette_square_0_r51_adv_test_detected_007_100/ --mask_pert 1. --is_square_patch 1 --batchSize 16 
python test.py --gpu_ids 0 --name imagenet_random --mask_type 4 --img_file ../imagenette_0_square_r51_org_test_misdetected_007_with_coordinate \
            --results_dir ./imagenette_square_0_r51_org_test_misdetected_007_100/ --mask_pert 1. --is_square_patch 1 --batchSize 16


echo "===> Mitigation Output"
cd ../ 
python imagenette-mitigation.py --img_folder Pluralistic-Inpainting/imagenette_square_0_r51_adv_test_detected_007_100 \
    --input_tag out_0 --normalize 1 --input_size 0 --extract_label 1  --model_path ./imagenette-model/resnet18-imagenette.pt 
python imagenette-mitigation.py --img_folder Pluralistic-Inpainting/imagenette_square_0_r51_org_test_misdetected_007_100 \
    --input_tag out_0 --normalize 1 --input_size 0 --extract_label 0  --model_path ./imagenette-model/resnet18-imagenette.pt



rm -rf imagenette_square_0_npy_test_adv_saliency_007
rm -rf imagenette_square_0_npy_test_adv_result_007
rm -rf imagenette_square_0_npy_test_org_saliency_007
rm -rf imagenette_square_0_npy_test_org_result_007
rm -rf imagenette_0_square_2feature_transfer_adv_comp_007
rm -rf imagenette_0_square_1feature_transfer_adv_comp_007
rm -rf imagenette_0_square_2feature_transfer_org_comp_007
rm -rf imagenette_0_square_1feature_transfer_org_comp_007

rm -rf imagenette_0_square_r51_adv_test_detected_007_with_coordinate
rm -rf imagenette_0_square_r51_adv_test_detected_007
rm -rf imagenette_0_square_r51_org_test_misdetected_007_with_coordinate
rm -rf imagenette_0_square_r51_org_test_misdetected_007

rm -rf Pluralistic-Inpainting/imagenette_square_0_r51_adv_test_detected_007_100
rm -rf Pluralistic-Inpainting/imagenette_square_0_r51_org_test_misdetected_007_100






