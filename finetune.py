"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import numpy as np

from models.models import FinetuneModel
from utils.config import create_config, create_exp_config
from utils.common_config import get_criterion, get_model, get_train_dataset, \
    get_val_dataset, get_train_dataloader, \
    get_val_dataloader, get_train_transformations, \
    get_val_transformations, get_optimizer, \
    adjust_learning_rate
from utils.evaluate_utils import contrastive_evaluate
from utils.memory import MemoryBank
from utils.train_utils import simclr_train, simclr_manifold_train
from utils.utils import fill_memory_bank
from termcolor import colored

# Parser
parser = argparse.ArgumentParser(description='Finetune model')
parser.add_argument('--config_exp', help='Config file for the experiment')
args = parser.parse_args()


def main():
    # Retrieve config file
    p = create_exp_config(args.config_exp)
    print(colored(p, 'red'))

    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_val_transformations(p)
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)

    train_dataset = get_train_dataset(p, train_transforms)
    val_dataset = get_val_dataset(p, val_transforms)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Checkpoint
    print(colored('Load checkpoint file at {}'.format(p['pretext_checkpoint']), 'blue'))
    start_epoch = 0
    checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    fine_tune_model = FinetuneModel(base_model=model, p=p).cuda()

    print('FinetuneModel is {}'.format(fine_tune_model.__class__.__name__))
    print('FinetuneModel parameters: {:.2f}M'.format(sum(p.numel() for p in fine_tune_model.parameters()) / 1e6))
    print(fine_tune_model)
    fine_tune_model = fine_tune_model.cuda()
    # Freeze all but last layer or not
    if p['freeze_layer'] == True:
        for name, param in fine_tune_model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    criterion = torch.nn.CrossEntropyLoss()

    # Training
    f = open(os.path.join(p['log_dir'], p['fine_tune_acc']), "a")
    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' % (epoch, p['epochs']), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        for batch in train_dataloader:
            images = batch['image'].cuda(non_blocking=True)
            target = batch['target'].cuda(non_blocking=True)
            optimizer.zero_grad()
            output = fine_tune_model(images)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Test on valid data
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_dataloader:
                images = batch['image'].cuda(non_blocking=True)
                target = batch['target'].cuda(non_blocking=True)
                # inputs, labels = inputs.to(device), labels.to(device)
                output = fine_tune_model(images)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        top1 = (100 * correct) / total
        f.write(str(top1))
        f.write('\n')
        print('Accuracy of the finetune model on the 10000 test images: %.3f %%' % top1)
        # Checkpoint
        print('Checkpoint ...')

if __name__ == '__main__':
    main()
