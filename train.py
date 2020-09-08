import os
import json
import argparse
import torch
import dataloaders
import models
import inspect
import math
from utils import losses
from utils import Logger
from utils.torchsummary import summary
from trainer import Trainer


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


def main(config, resume):
    train_logger = Logger()

    # DATA LOADERS
    # 相当于：train_loader = dataloaders.ADE20K(**config[train_loader]['args'])
    train_loader = get_instance(dataloaders, 'train_loader', config)
    # 相当于：val_loader = dataloaders.ADE20K(**config[val_loader]['args'])
    val_loader = get_instance(dataloaders, 'val_loader', config)

    # MODEL
    # 相当于：val_loader = models.PSPNet(train_loader.dataset.num_classes, **config[arch]['args'])
    model = get_instance(models, 'arch', config, train_loader.dataset.num_classes)
    # print(f'\n{model}\n')
    print('---------------------------------')
    # print('ffffffffff')
    # LOSS
    loss = getattr(losses, config['loss'])(ignore_index=config['ignore_index'])

    # TRAINING
    trainer = Trainer(
        model=model,
        loss=loss,
        resume=resume,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # config.json文件位置
    parser.add_argument('-c', '--config', default='config.json', type=str,
                        help='Path to the config file (default: config.json)')
    # 继续训练
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    # 设置GPU
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    # 读取配置文件
    config = json.load(open(args.config))
    if args.resume:
        config = torch.load(args.resume)['config']
    if args.device:
        print('using GPUs:', args.device)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'

    main(config, args.resume)
