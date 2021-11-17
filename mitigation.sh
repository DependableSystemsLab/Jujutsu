#!/bin/sh
python get_coordinate.py --saliency_folder ./imagenette_square_5_npy_test_adv_saliency_007 \
                                          --img_folder ./imagenette_5_square_r51_adv_test_detected_007 
python get_coordinate.py --saliency_folder ./imagenette_square_5_npy_test_org_saliency_007 \
                                          --img_folder ./imagenette_5_square_r51_org_test_misdetected_007

cd ./Pluralistic-Inpainting 
python test.py --name imagenet_random --mask_type 4 --img_file ../imagenette_5_square_r51_adv_test_detected_007_with_coordinate \
            --results_dir ./imagenette_square_5_r51_adv_test_detected_007_100/ --mask_pert 1. --is_square_patch 1 --batchSize 16 
python test.py --name imagenet_random --mask_type 4 --img_file ../imagenette_5_square_r51_org_test_misdetected_007_with_coordinate \
            --results_dir ./imagenette_square_5_r51_org_test_misdetected_007_100/ --mask_pert 1. --is_square_patch 1 --batchSize 16



cd ../ 
python imagenette-mitigation.py --img_folder Pluralistic-Inpainting/imagenette_square_5_r51_adv_test_detected_007_100 \
    --input_tag out_0 --normalize 1 --input_size 0 --extract_label 1  --model_path ./imagenette/resnet18-imagenette.pt 
python imagenette-mitigation.py --img_folder Pluralistic-Inpainting/imagenette_square_5_r51_org_test_misdetected_007_100 \
    --input_tag out_0 --normalize 1 --input_size 0 --extract_label 0  --model_path ./imagenette/resnet18-imagenette.pt





