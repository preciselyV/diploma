import torchvision

from torch.utils.data import DataLoader
from PIL import Image
import yaml


def prepare_dataset(dataset_path: str, img_size: int = 256, batch_size: int = 1) -> DataLoader:
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(img_size),
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                  (0.5, 0.5, 0.5))
                                                 ])
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transforms)
    dl = DataLoader(dataset=dataset, batch_size=batch_size)

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
