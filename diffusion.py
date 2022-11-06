import torch
from argparse import ArgumentParser

from utils import prepare_dataset
from torchvision.utils import save_image


class Diffusion:
    def __init__(self, b_upper, b_lower, steps, device='cuda'):
        # lower and upper bounds for noise schedule
        self.b_upper = b_upper
        self.b_lower = b_lower
        self.steps = steps
        self.device = device

        # a stands for alpha in DDPM paper
        self.a = 1. - torch.linspace(self.b_lower, self.b_upper, self.steps)
        self.a = self.a.to(self.device)
        # cumulative product of all a. We'll need it to calculate noise
        self.a_hat = torch.cumprod(self.a, dim=0)

    def noise_image(self, img, t):
        sqrt_a = torch.sqrt(self.a_hat[t]).to(self.device)[:, None, None, None]
        rev_sqrt_a = torch.sqrt(1. - self.a_hat[t]).to(self.device)[:, None, None, None]

        noise = torch.randn_like(img)
        return sqrt_a * img + rev_sqrt_a * noise


def main():
    argparser = ArgumentParser()
    argparser.add_argument("--dataset_path", default='images/')
    argparser.add_argument("--save_path", default='../results/')
    device = 'cuda'
    args = argparser.parse_args()
    dl = prepare_dataset(args.dataset_path)
    steps = 1000
    diffusion = Diffusion(b_upper=0.02, b_lower=1e-4, steps=steps, device=device)
    img, _ = next(iter(dl))
    img = img.to(device)
    t = torch.Tensor([1, 50, 100, 150, 200, 300, 600, 700, 999]).long().to(device)
    noised = diffusion.noise_image(img, t)
    save_image(noised.add(1).mul(0.5), args.save_path + "noise.jpg")


if __name__ == '__main__':
    main()
