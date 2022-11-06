import torchvision

from torch.utils.data import DataLoader
from PIL import Image


def prepare_dataset(dataset_path: str):
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(80),
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                  (0.5, 0.5, 0.5))
                                                 ])
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transforms)
    dl = DataLoader(dataset=dataset, batch_size=1)

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
