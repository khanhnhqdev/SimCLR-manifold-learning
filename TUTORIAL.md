# Tutorial: Self-supervised learning with SimCLR and manifold learning

You can follow this guide to obtain the semantic clusters with SCAN on the STL-10 dataset. The procedure is equivalent for the other datasets. 

## Contents
1. [Preparation](#preparation)
0. [Self-supervised task](#pretext-task)


## Preparation
### Repository
Clone the repository and navigate to the directory:
```bash
git clone https://github.com/khanhnhqdev/SimCLR_manifold_learning
cd  SimCLR_manifold_learning
checkout branch SimCLR
```

### Environment
Activate your python environment containing the packages in the README.md.
Make sure you have a GPU available (ideally a 1080TI or better) and set $gpu_ids to your desired gpu number(s):
```bash
conda activate your_anaconda_env
export CUDA_VISIBLE_DEVICES=$gpu_ids
```
I will use an environment with Python 3.7, Pytorch 1.6, CUDA 10.2 and CUDNN 7.5.6 for this example.

### Paths
Adapt the path in `configs/env.yml` to `saved-models/`, since this directory will be used in this tutorial. 
Make the following directories. The models will be saved there, other directories will be made on the fly if necessary.
```bash
mkdir saved-models/
mkdir saved-models/cifar-10/
mkdir saved-models/cifar-10/pretext/
```
Set the path in `utils/mypath.py` to your dataset root path as mentioned in the README.md

## Pretext task
First we will run the pretext task (i.e. SimCLR) on the train+unlabeled set of cifar-10. 
Feel free to run this task with the correct config file:
```
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10.yml 
```

In order to save time, we provide pretrained models in the README.md for all the datasets discussed in the paper. 
First, download the pretrained model [here](https://drive.google.com/file/d/1261NDFfXuKR2Dh4RWHYYhcicdcPag9NZ/view?usp=sharing) and save it in your experiments directory. Then, move the downloaded model to the correct location (i.e. `repository_eccv/stl-10/pretext/`) and calculate the nearest neighbors. This can be achieved by running the following commands:
```bash
mv simclr_stl-10.pth.tar repository_eccv/stl-10/pretext/checkpoint.pth.tar  # Move model to correct location
python tutorial_nn.py --config_env configs/env.yml --config_exp configs/pretext/simclr_stl10.yml    # Compute neighbors
```

You should get the following results:
```
> Restart from checkpoint repository_eccv/stl-10/pretext/checkpoint.pth.tar
> Fill memory bank for mining the nearest neighbors (train) ...
> Fill Memory Bank [0/10]
> Mine the nearest neighbors (Top-20)
> Accuracy of top-20 nearest ne ighbors on train set is 72.81
> Fill memory bank for mining the nearest neighbors (val) ...
> Fill Memory Bank [0/16]
> Mine the nearest neighbors (Top-5)
> Accuracy of top-5 nearest neighbors on val set is 79.85
```
Now, the model has been correctly saved for the clustering step and the nearest neighbors were computed automatically. 
