import torch
import torch.nn as nn

from tqdm import tqdm

from models import UnetV1
from diffusion import Diffusion


class DiffusionUNet:
    def __init__(self, model: nn.Module = None, diffusion: nn.Module = None,
                 device: str = 'cpu', img_size: int = 256):

        self.img_size = img_size
        self.device = device
        if model is None:
            self.model = UnetV1(channels=3)
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
                pred_noise = self.model(x_t)
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


if __name__ == '__main__':
    diff = DiffusionUNet(device='cuda')
    print(diff.diffusion.a.dim())
    noise = diff.sample_img(1)
    print(noise.shape)
