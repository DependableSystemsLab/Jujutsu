# Code for the paper - *Turning Your Strength against You: Detecting and Mitigating Robust and Universal Adversarial Patch Attack* 

**Update 2022 Jun**: Upload the code and data for all 4 datasets, you can now easily reproduce our results using all the scripts provided.

This repo contains the code and data to reproduce the main results in our [paper](https://arxiv.org/abs/2108.05075) on all 4 datasets (you'll need to download the data first - see below).

```./imagenette-model``` contains the ImageNette dataset and its hold-out inputs. 
```./imagenet-model``` contains the ImageNet dataset and its hold-out inputs. 
```./celeb-model``` contains the CelebA dataset and its hold-out inputs. 
```./place-model``` contains Place365 dataset and its hold-out inputs. 
```./patch_files``` contains the adversarial patches on all datasets.
```./Pluralistic-Inpainting``` is for image inpainting.
```./shell-script``` contains scripts for more experiments.


## Requirements
- PyTorch (tested on 1.7.1)
- torchvision (tested on 0.8.2)
- OpenCV 
- tpdm
- imageio
- dominate
- visdom
- tested on Python 3.8.8



## How to run

First download the data [here](https://drive.google.com/file/d/1OwOS_x2bvW0w-VziVI9QkSTjNEmsSAjr/view?usp=sharing) and move all 5 folders to the current directory: 4 folders for 4 datasets and 1 for image inpainting. 

Then you can run all the scripts as shown below. Each script performs both attack detection and mitigation on one setting (i.e., one adversarial patch). 

Each datasets contain 7 different evaluation settings, 3 for patches in different sizes (5%-7%) and 4 for patches targeting different labels. 

```
## For ImageNet dataset
./imagenet-859-007.sh >> R-imagenet-859-007 2>&1 
./imagenet-859-006.sh >> R-imagenet-859-006 2>&1 
./imagenet-859-005.sh >> R-imagenet-859-005 2>&1 
./imagenet-849-007.sh >> R-imagenet-849-007 2>&1 
./imagenet-513-007.sh >> R-imagenet-513-007 2>&1 
./imagenet-768-007.sh >> R-imagenet-768-007 2>&1 
./imagenet-954-007.sh  >> R-imagenet-954-007 2>&1 

## For CelebA dataset
./celeb-80-007.sh >> R-celeb-80-007 2>&1 
./celeb-80-006.sh >> R-celeb-80-006 2>&1 
./celeb-80-005.sh >> R-celeb-80-005 2>&1 
./celeb-20-007.sh >> R-celeb-20-007 2>&1 
./celeb-3-007.sh >> R-celeb-3-007 2>&1 
./celeb-53-007.sh >> R-celeb-53-007 2>&1 
./celeb-230-007.sh  >> R-celeb-230-007 2>&1 

## For ImageNette dataset
./imagenette-5-007.sh >> R-imagenette-5-007 2>&1 
./imagenette-5-006.sh >> R-imagenette-5-006 2>&1 
./imagenette-5-005.sh >> R-imagenette-5-005 2>&1 
./imagenette-0-007.sh >> R-imagenette-0-007 2>&1 
./imagenette-6-007.sh >> R-imagenette-6-007 2>&1 
./imagenette-8-007.sh >> R-imagenette-8-007 2>&1 
./imagenette-9-007.sh  >> R-imagenette-9-007 2>&1 

## For Place365 dataset
./place-64-007.sh >> R-place-64-007 2>&1 
./place-64-006.sh >> R-place-64-006 2>&1 
./place-64-005.sh >> R-place-64-005 2>&1 
./place-34-007.sh >> R-place-34-007 2>&1 
./place-158-007.sh >> R-place-158-007 2>&1 
./place-214-007.sh >> R-place-214-007 2>&1 
./place-354-007.sh  >> R-place-354-007 2>&1 
```

For example, ```./imagenet-859-007.sh  >> R-imagenet-859-007 2>&1``` evaluates a 7\% patch with the target label as 859 on ImageNet, and save the result in the file ```R-imagenet-859-007```. We recommend saving the output in a local file as the experiments might take some times to complete. 

**NOTE**: Running each script will generate a large number of files and occupy space. You may consider running the above scripts sequentially if you have space/file limit in your machine. The script will *delete* all the generated files after completing the experiment.

**More experiments**: you can find more scripts in ```./shell-script``` to run more experiments, including evaluation on *larger* patches (8%-10%) and on rectangular patch. Move these scripts to the main directory and run similarly as above.

*```detailed-usage.md``` explains the usage of different Python scripts.*
 





### Finding the detection output

Search for ```Detection Output``` in the output file (e.g., ```R-imagenet-859-007``` above), around which you'll see the following results.
 

- ```Detection success recall```: Amount of adversarial samples detected.
- ```False positive (FP)```: Amount of benign samples mis-detected as adversarial ones.


### Finding the mitigation output

Search for ```Mitigation Output``` in the output file, around which you'll see the following results.

- ```Robust accuracy```: Accuracy on adversarial samples (for mitigation)
- ```(Reduced) detection success recall```: This shows out of all the detected adversarial samples, how many of them will be **mis-identified** as benign samples (*the lower the better*). This gives the final amount of detected adversarial samples.
- ```Reduced (FP)```: This shows out of all the FPs on benign samples, how many of them will be **correctly identified** as benign (*the higher the better*). This gives the final FP of our technique.

### An example

Assume you obtain the following results after running the script:
```
#===> Detection Output
1) Detection success recall: 1576.0 out of 1592 adv inputs maintain the adv label: recall = 0.9899497487437185 
2) False positive: 1587.0 out of 1592 org inputs change the label: FP = 0.003140703517587995

#===> Mitigation Output
3) Recovered 1316.0 out of 1576, Robust accuracy = 0.8350
4) (Reduced) detection success recall: 1 out of 1576 adv images that remain the targeted adv label after mitigation (the lower the better) 
5) Reduced FP on benign samples: 4.0 out of 5 benign images that remain the same label before and after masking/inpainting (the higher the better)
```

In this example:

1. ```Final detection success recall``` is: (1576-1) / 1592. Minus 1 is from line 4) above (it means 1 adv sample was incorrectly determined as a benign sample after mitigation). 
2. ```Final false positive``` is (1592-1587-4) / 1592. Minus 1587 is from line 2) and Minus 4 is from line 5).
3. ```Final robust accuracy``` is 1316 / 1592 



# Citation
If you find this code useful, please consider citing our paper

```
@article{chen2021turning,
  title={Turning Your Strength against You: Detecting and Mitigating Robust and Universal Adversarial Patch Attacks},
  author={Chen, Zitao and Dash, Pritam and Pattabiraman, Karthik},
  journal={arXiv preprint arXiv:2108.05075},
  year={2021}
}
```






