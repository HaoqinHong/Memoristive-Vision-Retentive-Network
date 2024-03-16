import torch as th
from torchvision import datasets, transforms


def get_loader(dataset, size_batch):
    transform = transforms.Compose([transforms.ToTensor(),
                                    # transforms.Resize((8, 8), antialias=None),
                                    transforms.Normalize([0.5], [0.5])])

    if dataset == 'mnist':
        set_train = datasets.MNIST('data/mnist', train=True, download=True, transform=transform)
        set_test = datasets.MNIST('data/mnist', train=False, download=True, transform=transform)
    elif dataset == 'fmnist':
        set_train = datasets.FashionMNIST('data/fmnist', train=True, download=True, transform=transform)
        set_test = datasets.FashionMNIST('data/fmnist', train=False, download=True, transform=transform)
    elif dataset == 'cifar':
        set_train = datasets.CIFAR10('data/cifar', train=True, download=True, transform=transform)
        set_test = datasets.CIFAR10('data/cifar', train=False, download=True, transform=transform)
    else:
        print("Unknown dataset")
        exit(0)

    loader_train = th.utils.data.DataLoader(dataset=set_train, batch_size=size_batch, shuffle=True, drop_last=True)
    loader_test = th.utils.data.DataLoader(dataset=set_test, batch_size=size_batch * 2, shuffle=False, drop_last=False)

    return loader_train, loader_test


if __name__ == '__main__':
    a, b = get_loader('cifar', 10)
    for k, l in a:
        print(k[0])
