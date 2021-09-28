from torchvision import transforms, datasets

from torch.utils.data import DataLoader

from util.net.cut_out import CutOut


class CIFAR10:
    MEAN = [x / 255 for x in [125.3, 123.0, 113.9]]
    STD = [x / 255 for x in [63.0, 62.1, 66.7]]

    def __init__(self,
                 root,
                 loader_kwargs,
                 cutout=0) -> None:
                 
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD)
        ])

        if cutout > 0:
            train_transform.transforms += [CutOut(cutout)]

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD)
        ])

        trainset = datasets.CIFAR10(
            root, 
            train=True, 
            transform=train_transform, 
            download=True
        )
        testset = datasets.CIFAR10(
            root,
            train=False,
            transform=test_transform,
            download=False
        )
        self.loaders = {}
        train_loader = DataLoader(trainset, shuffle=True, **loader_kwargs)
        self.loaders['train'] = train_loader
        test_loader = DataLoader(testset, shuffle=False, **loader_kwargs)
        self.loaders['test'] = test_loader

    @property
    def get_loaders(self):
        return self.loaders