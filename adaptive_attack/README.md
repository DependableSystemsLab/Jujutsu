# Two adaptive attacks against our defence

## Adaptive attack 1

This attack aims to evade the detection by reducing the patch's influence to the output. We perform detection by first finding regions that have high influence to the output based on the saliency map. So the attacker's goal is to manipulate the saliency map in order to evade the detection (details in paper).

To study how would the attacker generate the adversarial patch in this case, go to ```./adaptive_attack``` and run:
```
python adaptive_attack_evade_detection.py --GPU 0 --data_dir /xxxx/ --n_samples 5 --train_size 200 --test_size 200 --model resnet50 --log_dir adaptive_attack_evade_detection_log.csv --max_iteration 500 --epochs 20

``` 
- ```--data_dir```: image folder
- ```--n_samples```: parameter for the saliency method (SmoothGrad)

Note that training under this new objective function is much more time-consuming.

You can compare the attack success rate of the resulting patch with that of the original attack. The attack success rate will drop significantly.

## Adaptive attack 2
This attack aims to cause targeted misclassification even after being detected.
We model this by checking whether the attack would succeed if we mask 50% or 75% of the pixels within the patch region.

```
python adaptive_attack_evade_mitigation.py  --data_dir /xxxx/ --mask_percentage 0.75 --GPU 0 --n_samples 5 --train_size 2000 --test_size 2000 --model resnet50 --log_dir mask_adaptive_attack_evade_mitigation_log.csv --max_iteration 1000 --epochs 30
```
- ```--mask_percentage```: percentage of pixels to mask within the patch region.

You can compare the attack success rate of the resulting patch with that of the original attack. The attack success rate will also degrade significantly.










