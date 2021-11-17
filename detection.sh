#!/bin/bash 

python imagenette_create_patched_input_with_saliency.py --noise_percentage 0.07  \
  --patch_file patch_files/imagenette_square_best_org_patch_007_5.npy --target 5 \
   --patch_type square --save_folder . \
   --model_path ./imagenette/resnet18-imagenette.pt \
   --data_dir ./imagenette/imagenette2-320 --test_size 5000


python feature_transfer.py --radius 51 --saliency_folder imagenette_square_5_npy_test_adv_saliency_007 \
    --img_folder imagenette_square_5_npy_test_adv_result_007 --adv_input 1 \
    --held_out_input_folder ./imagenette/imagenette_hold_out --held_out_saliency ./imagenette/imagenette_hold_out_saliency \
    --noise_percentage 0.07  --target 5 --patch_type square \
    --num_of_feature_trans 1 --random_seed 10 --dataset imagenette --save_folder .
python feature_transfer.py --radius 51 --saliency_folder imagenette_square_5_npy_test_adv_saliency_007 \
    --img_folder imagenette_square_5_npy_test_adv_result_007 --adv_input 1 \
    --held_out_input_folder ./imagenette/imagenette_hold_out --held_out_saliency ./imagenette/imagenette_hold_out_saliency \
    --noise_percentage 0.07  --target 5 --patch_type square \
    --num_of_feature_trans 2 --random_seed 20 --dataset imagenette --save_folder .


python feature_transfer.py --radius 51 --saliency_folder imagenette_square_5_npy_test_org_saliency_007 \
    --img_folder imagenette_square_5_npy_test_org_result_007 --adv_input 0 \
    --held_out_input_folder ./imagenette/imagenette_hold_out --held_out_saliency ./imagenette/imagenette_hold_out_saliency \
    --noise_percentage 0.07  --target 5 --patch_type square \
    --num_of_feature_trans 1 --random_seed 10 --dataset imagenette --save_folder .
python feature_transfer.py --radius 51 --saliency_folder imagenette_square_5_npy_test_org_saliency_007 \
    --img_folder imagenette_square_5_npy_test_org_result_007 --adv_input 0 \
    --held_out_input_folder ./imagenette/imagenette_hold_out --held_out_saliency ./imagenette/imagenette_hold_out_saliency \
    --noise_percentage 0.07  --target 5 --patch_type square \
    --num_of_feature_trans 2 --random_seed 20 --dataset imagenette --save_folder .



python imagenette-detection.py --is_adv 1 --num_input 0 --img_folder imagenette_5_square_1feature_transfer_adv_comp_007  \
        --source_folder imagenette_square_5_npy_test_adv_result_007 --noise_percentage 0.07 \
        --radius 51 --second_folder imagenette_5_square_2feature_transfer_adv_comp_007 --target 5 \
        --save_folder ./ --patch_type square --model_path ./imagenette/resnet18-imagenette.pt
python imagenette-detection.py --is_adv 0 --num_input 0 --img_folder imagenette_5_square_1feature_transfer_org_comp_007  \
        --source_folder imagenette_square_5_npy_test_org_result_007 --noise_percentage 0.07 \
        --radius 51 --second_folder imagenette_5_square_2feature_transfer_org_comp_007 --target 5 \
        --save_folder ./ --patch_type square --model_path ./imagenette/resnet18-imagenette.pt






