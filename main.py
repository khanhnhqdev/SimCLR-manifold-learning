"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch
import numpy as np

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
parser = argparse.ArgumentParser(description='SimCLR-manifold')
parser.add_argument('--config_exp', help='Config file for the experiment')
args = parser.parse_args()


def main():
    # Retrieve config file
    p = create_exp_config(args.config_exp)
    print(colored(p, 'red'))

    # Model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print('Model is {}'.format(model.__class__.__name__))
    print('Model parameters: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print(model)
    model = model.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    print('Train transforms:', train_transforms)
    val_transforms = get_val_transformations(p)
    print('Validation transforms:', val_transforms)
    if p['criterion'] == 'simclr-manifold':
        train_dataset = get_train_dataset(p, train_transforms, to_many_views=True,
                                      split='train+unlabeled')  # Split is for stl-10
    elif p['criterion'] == 'simclr':
        train_dataset = get_train_dataset(p, train_transforms, to_augmented_dataset=True,
                                      split='train+unlabeled')  # Split is for stl-10
    val_dataset = get_val_dataset(p, val_transforms)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Dataset contains {}/{} train/val samples'.format(len(train_dataset), len(val_dataset)))

    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    base_dataset = get_train_dataset(p, val_transforms, split='train')  # Dataset w/o augs for knn eval
    base_dataloader = get_val_dataloader(p, base_dataset)
    memory_bank_base = MemoryBank(len(base_dataset),
                                  p['model_kwargs']['features_dim'],
                                  p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_base.cuda()
    memory_bank_val = MemoryBank(len(val_dataset),
                                 p['model_kwargs']['features_dim'],
                                 p['num_classes'], p['criterion_kwargs']['temperature'])
    memory_bank_val.cuda()

    # Criterion
    print(colored('Retrieve criterion', 'blue'))
    criterion = get_criterion(p)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.cuda()

    # Optimizer and scheduler
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Checkpoint
    if os.path.exists(p['pretext_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['pretext_checkpoint']), 'blue'))
        checkpoint = torch.load(p['pretext_checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['pretext_checkpoint']), 'blue'))
        start_epoch = 0
        model = model.cuda()

    # Training

    print(colored('Starting main loop', 'blue'))
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' % (epoch, p['epochs']), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        if p['criterion'] == 'simclr-manifold':
            simclr_manifold_train(p, train_dataloader, model, criterion, optimizer, epoch)
        elif p['criterion'] == 'simclr':
            simclr_train(train_dataloader, model, criterion, optimizer, epoch)
        # Fill memory bank
        print('Fill memory bank for kNN...')
        fill_memory_bank(base_dataloader, model, memory_bank_base)

        # Evaluate (To monitor progress - Not for validation)
        print('Evaluate ...')
        top1 = contrastive_evaluate(val_dataloader, model, memory_bank_base)
        print('Result of kNN evaluation is %.2f' % (top1))
        f = open(os.path.join(p['log_dir'], p['KNN_acc']), "a")
        f.write(str(top1))
        f.write('\n')
        f.close()
        # Checkpoint
        print('Checkpoint ...')

        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
        'epoch': epoch + 1}, p['pretext_checkpoint'])

    # Save final model
    torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                'epoch': p['epochs'] + 1}, p['pretext_checkpoint'])

if __name__ == '__main__':
    main()
