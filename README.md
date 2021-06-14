# SimCLR manifold learning
SimCLR is one of the SOTA methods in Contrastive Learning. In SimCLR, the model forces representations of positive pairs as close as possible and pushes the representations of negative pairs as far as possible. So essentially, SimCLR considers a little information in 'view' space (space of transformations after original image pass through Data Augmentation pipeline). We hypothesis that the model can learn better presentations by giving it more information about the manifold property of 'view' space. Institute by Local Linear Embedding algorithms, an old method in manifold learning, we provide information about the local region in 'view' space for the model and systematically study how different structures of the local region affect the strength of latent representation. We call our method Local Contrastive Learning and this repository is the official implementation in [paper](https://github.com/khanhnhqdev/SimCLR-manifold-learning/blob/main/TUTORIAL.md). This code is based on [SCAN githup](https://github.com/wvangansbeke/Unsupervised-Classification)   

### Repository
Clone the repository and navigate to the directory:
```bash
git clone https://github.com/khanhnhqdev/SimCLR_manifold_learning
cd  SimCLR_manifold_learning
```

### Environment
Activate your python environment using conda. Install necessary package in requirements.txt 
```bash
conda acivate <your_environment_name>
```


### How to run code
```bash
python main.py --config_exp <your_config_experiment_yml_file>
```
### Tutorial

If you want to see another (more detailed) example for CIFAR-10, checkout [TUTORIAL.md](https://github.com/khanhnhqdev/SimCLR-manifold-learning/blob/main/TUTORIAL.md). It provides a detailed guide and includes visualizations and log files with the training progress.
