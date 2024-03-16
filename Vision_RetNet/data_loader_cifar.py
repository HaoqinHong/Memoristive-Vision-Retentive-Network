import torch
from torchvision import datasets
from torchvision import transforms
import os

def get_loader(args):
    if args.dset == 'mnist':
        tr_transform = transforms.Compose([
            transforms.RandomCrop(args.img_size, padding=2), 
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])
        ])
        train = datasets.MNIST(os.path.join(args.data_path, args.dset), train=True, download=True, transform=tr_transform)

        te_transform = transforms.Compose([
            transforms.Resize([args.img_size, args.img_size]), 
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])
        ])
        test = datasets.MNIST(os.path.join(args.data_path, args.dset), train=False, download=True, transform=te_transform)

    elif args.dset == 'fmnist':
        tr_transform = transforms.Compose([
            transforms.RandomCrop(args.img_size, padding=2), 
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])
        ])
        train = datasets.FashionMNIST(os.path.join(args.data_path, args.dset), train=True, download=True, transform=tr_transform)

        te_transform = transforms.Compose([
            transforms.Resize([args.img_size, args.img_size]), 
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])
        ])
        test = datasets.FashionMNIST(os.path.join(args.data_path, args.dset), train=False, download=True, transform=te_transform)

    elif args.dset == 'cifar10':
        tr_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), # 将图像转换为灰度图像
            transforms.RandomCrop(args.img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        te_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize([args.img_size, args.img_size]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        train = datasets.CIFAR10(os.path.join(args.data_path, args.dset), train=True, download=True, transform=tr_transform)
        test = datasets.CIFAR10(os.path.join(args.data_path, args.dset), train=False, download=True, transform=te_transform)

    else:
        print("Unknown dataset")
        exit(0)

    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=args.batch_size * 2,
                                              shuffle=False,
                                              num_workers=args.num_workers,
                                              drop_last=False)

    return train_loader, test_loader
