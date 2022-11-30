import logging
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.optim.lr_scheduler import CyclicLR, _LRScheduler
from torchvision.utils import save_image

from models import UnetV1
from diffusion import Diffusion
from utils import (setup_dataset, load_config, setup_writer, setup_diffusion, setup_model,
                   setup_logging, save_model)


class DiffusionUNet:
    def __init__(self, model: nn.Module = None, diffusion: nn.Module = None,
                 device: str = 'cpu', img_size: int = 256, writer: SummaryWriter = None,
                 cfg: dict = None):

        self.device = device
        self.fid = FrechetInceptionDistance(feature=192).to(self.device)

        # found a silly big in torchmetrics: all first fid.compute() calls
        # will result in ValueError. Prolly gonna dig into it later. For now
        # lets just abuse this bug and make a dummy call, so it won't bother us later
        try:
            b = torch.randint(0, 200, (1, 3, 128, 128), dtype=torch.uint8).to(self.device)
            a = torch.randint(0, 200, (1, 3, 128, 128), dtype=torch.uint8).to(self.device)
            self.fid.update(a, real=True)
            self.fid.update(b, real=False)
            self.fid.compute()
        except ValueError:
            pass

        self.data_conf = cfg['data']
        self.model_conf = cfg['model']
        self.run_name = cfg['run_name']
        self.img_size = img_size
        self.writer = writer
        if model is None:
            self.model = UnetV1(channels=3, time_dim=256)
        else:
            self.model = model
        self.model = self.model.to(self.device)
        if diffusion is None:
            self.diffusion = Diffusion(b_lower=1e-4, b_upper=0.02, steps=3, device=self.device)
        else:
            self.diffusion = diffusion

    # Algorithm 2 Sampling from DDPM
    def sample(self, amount: int, x_t: Tensor = None) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            if x_t is None:
                # this is the noise that we expect to convert to img
                # now t = T
                x_t = torch.randn((amount, self.model.channels, self.img_size, self.img_size))
            for ind in tqdm(reversed(range(1, self.diffusion.steps)), total=self.diffusion.steps,
                            desc='sampling'):
                x_t = x_t.to(self.device)
                t = (torch.ones(amount) * ind).long().to(self.device)
                alpha = self.diffusion.a[t][:, None, None, None]
                alpha_hat = self.diffusion.a_hat[t][:, None, None, None]
                # "Experimentally, both σ^2_t = βt and
                # σ^2_t = β˜t = (1−α(t−1)/1−αt) * βt had similar results"
                beta = self.diffusion.beta[t][:, None, None, None]
                sigma = torch.sqrt(beta)
                pred_noise = self.model(x_t, t)
                if ind > 1:
                    z = torch.randn_like(x_t)
                else:
                    z = torch.zeros_like(x_t)

                x_t = ((1 / torch.sqrt(alpha))
                       * (x_t - ((1-alpha)/(torch.sqrt(1-alpha_hat))) * pred_noise)) + z * sigma
        self.model.train()
        return x_t

    def convert_tensor(self, tens: Tensor) -> Tensor:
        tens = (tens + 1) / 2
        return tens

    def sample_img(self, amount: int, noise: Tensor = None) -> torch.Tensor:
        img = self.sample(amount=amount, x_t=noise)
        return self.convert_tensor(img)

    def train(self, dataloader: DataLoader, optim: Optimizer, lossfunc: nn.Module, epochs: int):
        mixed_precision = self.model_conf.get('mixed_precision', False)
        scheduler = None
        if self.model_conf.get('cyclicLR', False):
            max_lr = self.model_conf.get('max_lr')
            base_lr = self.model_conf.get('lr')
            scheduler = CyclicLR(optimizer=optim, base_lr=base_lr, max_lr=max_lr,
                                 step_size_up=len(dataloader), cycle_momentum=False)
        if mixed_precision:
            self.train_mixed(dataloader, optim, lossfunc, epochs, scheduler)
        else:
            self.train_(dataloader, optim, lossfunc, epochs, scheduler)

    def write_metrics(self, avg_loss: int, dataloader: DataLoader, epoch: int, lr: int):
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        self.writer.add_scalar('LearningRate', lr, epoch)
        logging.info(f'epoch {epoch} loss is {avg_loss}')
        self.model.eval()

        with torch.no_grad():
            sampled_imgs = self.sample(16).to('cpu')
            self.writer.add_images('generated_images', sampled_imgs, epoch)
            real_img, _ = next(iter(dataloader))
            timesteps = (torch.ones((real_img.shape[0])) * (self.diffusion.steps - 1)).long()
            real_img = real_img.to(self.device)
            timesteps = timesteps.to(self.device)
            noised, _ = self.diffusion.noise_image(real_img, timesteps)
            image = self.sample(noised.shape[0], noised)
            image = image.to(self.device)
            self.fid.update(self.convert_tensor(real_img), real=True)
            self.fid.update(self.convert_tensor(image), real=False)
            val = self.fid.compute()
            self.writer.add_scalar('Metric/FID', val, epoch)
            self.writer.add_images('FID/original', real_img, epoch)
            self.writer.add_images('FID/generated', image, epoch)

        self.model.train()

    def train_mixed(self, dataloader: DataLoader, optim: Optimizer, lossfunc: nn.Module,
                    epochs: int, scheduler: _LRScheduler = None):
        scaler = torch.cuda.amp.GradScaler()
        for i in range(epochs):
            avg_loss = 0
            for (img, _) in tqdm(dataloader, desc='epoch Progress'):
                t = torch.randint(low=1, high=self.diffusion.steps, size=(img.shape[0],))

                t = t.to(self.device)
                img = img.to(self.device)

                imgs, noise = self.diffusion.noise_image(img, t)
                optim.zero_grad()
                with torch.cuda.amp.autocast():
                    predicted_noise = self.model(imgs, t)
                    loss = lossfunc(noise, predicted_noise)
                scaler.scale(loss).backward()
                avg_loss += loss.detach().cpu().item()
                scaler.step(optim)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

            avg_loss /= len(dataloader)
            self.write_metrics(avg_loss=avg_loss, dataloader=dataloader, epoch=i,
                               lr=optim.param_groups[0]['lr'])

            if i % 10 == 0:
                self.model = self.model.to('cpu')
                self.model.eval()
                save_model(self.model, self.data_conf['checkpoints-path'], self.run_name, i)
                self.model = self.model.to(self.device)
                self.model.train()

    def train_(self, dataloader: DataLoader, optim: Optimizer, lossfunc: nn.Module, epochs: int,
               scheduler: _LRScheduler = None):
        for i in range(epochs):
            avg_loss = 0
            for (img, _) in tqdm(dataloader, desc='epoch Progress'):
                t = torch.randint(low=1, high=self.diffusion.steps, size=(img.shape[0],))

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
            self.write_metrics(avg_loss=avg_loss, dataloader=dataloader, epoch=i,
                               lr=optim.param_groups[0]['lr'])

            if i % 10 == 0:
                self.model = self.model.to('cpu')
                self.model.eval()
                save_model(self.model, self.data_conf['checkpoints-path'], self.run_name, i)
                self.model = self.model.to(self.device)
                self.model.train()


def dry_run(cfg: dict):
    tfwriter = setup_writer(cfg)
    diff = DiffusionUNet(device=cfg['model']['device'], img_size=cfg['data']['image-size'],
                         writer=tfwriter, cfg=cfg)
    noise = diff.sample_img(1)
    print(noise.dim())
    mse = nn.MSELoss()
    optim = AdamW(diff.model.parameters(), lr=1e-4)
    dl = setup_dataset(dataset_path=cfg['data']['dataset-path'],
                       img_size=cfg['data']['image-size'],
                       batch_size=2)
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
                                   writer=tfwriter, cfg=cfg)
    mse = nn.MSELoss()
    optim = AdamW(diffusionModel.model.parameters(),
                  lr=cfg['model']['lr'])
    diffusionModel.train(dataloader=dl, optim=optim, lossfunc=mse,
                         epochs=cfg['model']['epochs'])
    tfwriter.close()


def sample(cfg: dict):
    model = setup_model(cfg['model'])
    model.load_state_dict(torch.load(cfg['checkpoint_path']))
    model.eval()
    diffusion = setup_diffusion(cfg['diffusion'])
    diffusionModel = DiffusionUNet(model=model, diffusion=diffusion,
                                   device=cfg['model']['device'],
                                   img_size=cfg['data']['image-size'],
                                   cfg=cfg)
    imgs = diffusionModel.sample_img(cfg['data']['batch-size'])
    save_image(imgs, cfg['save_path'])


def main():
    argparser = ArgumentParser()
    argparser.add_argument('mode', choices=['dry-run', 'sample', 'train'])
    argparser.add_argument("--config_path", default='configs/conf.yml')
    argparser.add_argument('--name', default='run_0')
    argparser.add_argument('--checkpoint_path', default='')
    argparser.add_argument('--save_path', default='')
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
    elif args.mode == 'sample':
        logging.info('running in sample mode')
        cfg['checkpoint_path'] = args.checkpoint_path
        cfg['save_path'] = args.save_path
        sample(cfg)


if __name__ == '__main__':
    main()
