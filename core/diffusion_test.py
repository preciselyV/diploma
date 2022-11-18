from argparse import ArgumentParser
from utils import setup_dataset
from torchvision.utils import save_image

import torch

from diffusion import Diffusion


def test_diffusion(args: dict):
    device = 'cuda'
    dl = setup_dataset(args.dataset_path)
    steps = 1000
    diffusion = Diffusion(b_upper=0.02, b_lower=1e-4, steps=steps, device=device)
    img, _ = next(iter(dl))
    img = img.to(device)
    t = torch.Tensor([1, 50, 100, 150, 200, 300, 600, 700, 999]).long().to(device)
    noised, _ = diffusion.noise_image(img, t)
    save_image(noised.add(1).mul(0.5), args.save_path + "noise.jpg")


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--dataset_path", default='images/')
    argparser.add_argument("--save_path", default='../results/')
    args = argparser.parse_args()
    test_diffusion()
