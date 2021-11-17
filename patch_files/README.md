This folder contains some adversarial patches generated from the ResNet-18 model used in our paper.

The default shell script in the main directory use ```imagenette_square_best_org_patch_007_5.npy```, which is a ```square``` patch occupying ```7%``` of the image pixels, and targeting ```class 5```.

You can try out other patches or generate your own patches using ```imagenette-attack.py``` in the main directory, e.g.,

```
python imagenette-attack.py --target 9 --noise_percentage 0.07 \
              --train_size 2000 --model_path /imagenette/resnet18-imagenette.pt
```