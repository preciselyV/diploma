import torch


class Diffusion:
    def __init__(self, b_upper, b_lower, steps, device='cuda'):
        # lower and upper bounds for noise schedule
        self.b_upper = b_upper
        self.b_lower = b_lower
        self.steps = steps
        self.device = device

        # a stands for alpha in DDPM paper
        self.beta = torch.linspace(self.b_lower, self.b_upper, self.steps).to(self.device)
        self.a = 1. - self.beta
        self.a = self.a.to(self.device)
        # cumulative product of all a. We'll need it to calculate noise
        self.a_hat = torch.cumprod(self.a, dim=0)

    def noise_image(self, img, t):
        sqrt_a = torch.sqrt(self.a_hat[t]).to(self.device)[:, None, None, None]
        rev_sqrt_a = torch.sqrt(1. - self.a_hat[t]).to(self.device)[:, None, None, None]

        noise = torch.randn_like(img)
        return sqrt_a * img + rev_sqrt_a * noise, noise
