from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.optim import Optimizer, Adam

from tqdm import tqdm
from torch.utils.data import DataLoader

from models import UnetV1
from diffusion import Diffusion
from utils import prepare_dataset, load_config


class DiffusionUNet:
    def __init__(self, model: nn.Module = None, diffusion: nn.Module = None,
                 device: str = 'cpu', img_size: int = 256):

        self.img_size = img_size
        self.device = device
        if model is None:
            self.model = UnetV1(channels=3, device=self.device, time_dim=256)
        else:
            self.model = model
        self.model = self.model.to(self.device)
        if diffusion is None:
            self.diffusion = Diffusion(b_lower=1e-4, b_upper=0.02, steps=3, device=self.device)

    # Algorithm 2 Sampling from DDPM
    def sample(self, amount: int) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            # this is the noise that we expect to convert to img
            # now t = T
            x_t = torch.randn((amount, self.model.channels, self.img_size, self.img_size))
            for ind in tqdm(reversed(range(self.diffusion.steps))):
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
        noise = self.sample(amount=amount)
        noise = (noise.clamp(-1, 1) + 1) / 2
        noise = (noise * 255).type(torch.uint8)
        return noise

    def train(self, dataloader: DataLoader, optim: Optimizer, lossfunc: nn.Module, epochs: int):
        for i in range(epochs):
            print(f"epoch {i}")
            for (img, _) in dataloader:
                t = torch.randint(low=1, high=self.diffusion.steps, size=(dataloader.batch_size,))
                t = t.to(self.device)

                img = img.to(self.device)
                imgs, noise = self.diffusion.noise_image(img, t)
                predicted_noise = self.model(imgs, t)
                loss = lossfunc(noise, predicted_noise)

                optim.zero_grad()
                loss.backward()
                optim.step()


def dry_run(cfg: dict):
    diff = DiffusionUNet(device='cuda', img_size=80)
    noise = diff.sample_img(1)
    print(noise.dim())
    mse = nn.MSELoss()
    optim = Adam(diff.model.parameters(), lr=1e-4)
    dl = prepare_dataset(dataset_path=cfg['data']['dataset-path'],
                         img_size=cfg['data']['image-size'],
                         batch_size=cfg['data']['batch-size'])
    epochs = 2
    diff.train(dataloader=dl, optim=optim, lossfunc=mse, epochs=epochs)


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


def main():
    argparser = ArgumentParser()
    argparser.add_argument('mode', choices=['dry-run', 'sample', 'train'])
    argparser.add_argument("--config_path", default='configs/conf.yml')
    args = argparser.parse_args()
    cfg = load_config(args.config_path)
    print(cfg)
    if args.mode == 'train':
        model = setup_model(cfg['model'])
        diffusion = setup_diffusion(cfg['diffusion'])
        dl = prepare_dataset(dataset_path=cfg['data']['dataset-path'],
                             img_size=cfg['data']['image-size'],
                             batch_size=cfg['data']['batch-size'])
        diffusionModel = DiffusionUNet(model=model, diffusion=diffusion,
                                       device=cfg['model']['device'],
                                       img_size=cfg['data']['image-size'])
        mse = nn.MSELoss()
        optim = Adam(diffusionModel.model.parameters(),
                     lr=cfg['model']['lr'])
        diffusionModel.train(dataloader=dl, optim=optim, lossfunc=mse,
                             epochs=cfg['model']['epochs'])
    elif args.mode == 'dry-run':
        dry_run(cfg)


if __name__ == '__main__':
    main()
