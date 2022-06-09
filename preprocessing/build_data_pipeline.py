import os

import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import root_dir


def build_data_pipeline(config: dict) -> (DataLoader, DataLoader):
    """
    reference: https://github.com/pytorch/examples/blob/main/imagenet/main.py
    """
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']

    dataset_dir = config['dataset']['dataset_dir']
    traindir = root_dir.joinpath(dataset_dir, 'train')
    valdir = root_dir.joinpath(dataset_dir, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    resize = config['dataset']['resize']
    train_dataset = datasets.ImageFolder(traindir,
                                         transforms.Compose([transforms.Resize([resize, resize]),
                                                             transforms.ToTensor(),
                                                             normalize]))
    val_dataset = datasets.ImageFolder(valdir,
                                       transforms.Compose([transforms.Resize([resize, resize]),
                                                           transforms.ToTensor(),
                                                           normalize]))

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=num_workers,
                                   pin_memory=True if num_workers > 0 else False)
    val_data_loader = DataLoader(val_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=num_workers,
                                 pin_memory=True if num_workers > 0 else False)

    return train_data_loader, val_data_loader


if __name__ == '__main__':
    import einops
    import matplotlib.pyplot as plt

    traindir = root_dir.joinpath('datasets/imagenet-mini', 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([transforms.Resize([128, 128]),
                                                                       transforms.ToTensor(),
                                                                       normalize]))
    print('len(train_dataset):', len(train_dataset))

    n_samples = 5
    for i in range(n_samples):
        x, y = train_dataset[i]
        x = einops.rearrange(x, 'c h w -> h w c')
        print(x.shape)
        print(x.min(), x.max())
        print(y)

        plt.imshow(x)
        plt.show()
