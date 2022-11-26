import os
import logging

import torchvision
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from PIL import Image
import yaml
from torch.utils.tensorboard import SummaryWriter

from models import UnetV1
from diffusion import Diffusion


def setup_dataset(dataset_path: str, img_size: int = 256, batch_size: int = 1) -> DataLoader:
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size),
                                                 torchvision.transforms.ToTensor()
                                                 ])
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transforms)
    dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return dl


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def my_save_images(images, path):
    for i in range(len(images)):
        ndarr = images[i].permute(1, 2, 0).to('cpu').numpy()
        full_path = path + "/" + str(i) + '.jpg'
        im = Image.fromarray(ndarr)
        im.save(full_path)


def load_config(conf_path: str):
    with open(conf_path, "r") as f:
        cfg = yaml.safe_load(f.read())
    cfg['model']['lr'] = float(cfg['model']['lr'])
    cfg['diffusion']['beta_upper'] = float(cfg['diffusion']['beta_upper'])
    cfg['diffusion']['beta_lower'] = float(cfg['diffusion']['beta_lower'])
    return cfg


def setup_model(model_args: dict):
    model = UnetV1(channels=model_args['input-channels'],
                   time_dim=model_args['timestep-embedding-dim'],
                   device=model_args['device'])
    return model


def setup_diffusion(diffusion_args: dict):
    diffusion = Diffusion(b_lower=diffusion_args['beta_lower'],
                          b_upper=diffusion_args['beta_upper'],
                          steps=diffusion_args['steps'],
                          device=diffusion_args['device'])
    return diffusion


def setup_writer(cfg: dict):
    path = os.path.join(cfg['data']['logs-path'], cfg['run_name'])
    os.mkdir(path)
    tfwriter = SummaryWriter(log_dir=path)
    return tfwriter


def setup_logging(cfg: dict):
    logging.basicConfig(
        level=cfg['logging']['log_level'],
        format=cfg['logging']['log_format']
    )


def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)
