import torch
from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *
import os

def test():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, None)
    exp_path = args.test_path
    trainer.test(exp_path)


if __name__ == '__main__':
    test()
