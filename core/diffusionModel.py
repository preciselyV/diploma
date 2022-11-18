import logging
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import UnetV1
from diffusion import Diffusion
from utils import (setup_dataset, load_config, setup_writer, setup_diffusion, setup_model,
                   setup_logging)


class DiffusionUNet:
    def __init__(self, model: nn.Module = None, diffusion: nn.Module = None,
                 device: str = 'cpu', img_size: int = 256, writer: SummaryWriter = None):

        self.img_size = img_size
        self.device = device
        self.writer = writer
        if model is None:
            self.model = UnetV1(channels=3, device=self.device, time_dim=256)
        else:
            self.model = model
        self.model = self.model.to(self.device)
        if diffusion is None:
            self.diffusion = Diffusion(b_lower=1e-4, b_upper=0.02, steps=3, device=self.device)
        else:
            self.diffusion = diffusion

    # Algorithm 2 Sampling from DDPM
    def sample(self, amount: int) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            # this is the noise that we expect to convert to img
            # now t = T
            x_t = torch.randn((amount, self.model.channels, self.img_size, self.img_size))
            for ind in tqdm(reversed(range(self.diffusion.steps)), total=self.diffusion.steps,
                            desc='sampling'):
                x_t = x_t.to(self.device)
                t = (torch.ones(amount) * ind).long().to(self.device)
                alpha = self.diffusion.a[t][:, None, None, None]
                alpha_hat = self.diffusion.a_hat[t][:, None, None, None]
                # "Experimentally, both σ^2_t = βt and
                # σ^2_t = β˜t = (1−α(t−1)/1−αt) * βt had similar results"
                beta = self.diffusion.a_hat[t][:, None, None, None]
                sigma = torch.sqrt(beta)
                pred_noise = self.model(x_t, t)
                if ind > 1:
                    z = torch.randn_like(x_t)
                else:
                    z = torch.zeros_like(x_t)

                x_t = ((1 / torch.sqrt(alpha_hat))
                       * (x_t - (1-alpha)/(torch.sqrt(1-alpha)) * pred_noise)) + z * sigma
        self.model.train()
        return x_t

    def sample_img(self, amount: int) -> torch.Tensor:
        img = self.sample(amount=amount)
        img = (img.clamp(-1, 1) + 1) / 2
        img = (img * 255).type(torch.uint8)
        return img

    def train(self, dataloader: DataLoader, optim: Optimizer, lossfunc: nn.Module, epochs: int):
        for i in range(epochs):
            avg_loss = 0
            for (img, _) in tqdm(dataloader, desc='epoch Progress'):
                t = torch.randint(low=1, high=self.diffusion.steps, size=(dataloader.batch_size,))

                t = t.to(self.device)
                img = img.to(self.device)

                imgs, noise = self.diffusion.noise_image(img, t)
                optim.zero_grad()
                predicted_noise = self.model(imgs, t)
                loss = lossfunc(noise, predicted_noise)
                avg_loss += loss.item()

                loss.backward()
                optim.step()
            avg_loss /= len(dataloader)
            logging.info(f'epoch {i} loss is {avg_loss}')
            self.writer.add_scalar('Loss/train', avg_loss, i)
            sampled_imgs = self.sample(5).to('cpu')
            self.writer.add_images('generated_images', sampled_imgs, i)


def dry_run(cfg: dict):
    tfwriter = setup_writer(cfg)
    diff = DiffusionUNet(device=cfg['model']['device'], img_size=cfg['data']['image-size'],
                         writer=tfwriter)
    noise = diff.sample_img(1)
    print(noise.dim())
    mse = nn.MSELoss()
    optim = Adam(diff.model.parameters(), lr=1e-4)
    dl = setup_dataset(dataset_path=cfg['data']['dataset-path'],
                       img_size=cfg['data']['image-size'],
                       batch_size=1)
    epochs = 5
    diff.train(dataloader=dl, optim=optim, lossfunc=mse, epochs=epochs)
    tfwriter.close()


def train(cfg: dict):
    model = setup_model(cfg['model'])
    diffusion = setup_diffusion(cfg['diffusion'])
    dl = setup_dataset(dataset_path=cfg['data']['dataset-path'],
                       img_size=cfg['data']['image-size'],
                       batch_size=cfg['data']['batch-size'])
    tfwriter = setup_writer(cfg)
    diffusionModel = DiffusionUNet(model=model, diffusion=diffusion,
                                   device=cfg['model']['device'],
                                   img_size=cfg['data']['image-size'],
                                   writer=tfwriter)
    mse = nn.MSELoss()
    optim = Adam(diffusionModel.model.parameters(),
                 lr=cfg['model']['lr'])
    diffusionModel.train(dataloader=dl, optim=optim, lossfunc=mse,
                         epochs=cfg['model']['epochs'])
    tfwriter.close()


def main():
    argparser = ArgumentParser()
    argparser.add_argument('mode', choices=['dry-run', 'sample', 'train'])
    argparser.add_argument("--config_path", default='configs/conf.yml')
    argparser.add_argument('--name', default='run_0')
    args = argparser.parse_args()
    cfg = load_config(args.config_path)
    cfg['run_name'] = args.name
    setup_logging(cfg)
    logging.info(f'config:{cfg}')
    if args.mode == 'train':
        logging.info('running in train mode')
        train(cfg)
    elif args.mode == 'dry-run':
        logging.info('running in dry-run mode')
        dry_run(cfg)


if __name__ == '__main__':
    main()
