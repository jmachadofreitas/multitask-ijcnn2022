from typing import Optional, Callable
from pathlib import Path
import random
from filelock import FileLock

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as T
import pytorch_lightning as pl


TRANSFORMS = T.Compose([
    # T.Resize(size=(14, 14)),
    # T.RandomRotation(degrees=(-60, 60)),
    # T.RandomPerspective(distortion_scale=0.6, p=.1),
    # T.RandomInvert(p=0.5),
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])


class GroupMembershipMNIST(MNIST):
    """ Toy example for binary multitask classification """

    def __init__(
            self,
            num_groups: int = 4,
            num_elements: int = 4,
            root: str = "datasets/",
            train: bool = True,
            transform: Optional[Callable] = None,
            download: bool = False,
            seed: int = 0
    ):
        Path(root).mkdir(parents=True, exist_ok=True)
        with FileLock(f"{root}/mnist.lock"):
            super().__init__(
                root=root,
                train=train,
                transform=transform,
                target_transform=None,
                download=download,
            )

        self.img_shape = (784,)
        self.num_classes = len(self.classes)
        self.transform = transform
        self.seed = seed

        # Generate subgroups
        self.num_groups = num_groups
        self.num_elements = num_elements
        self.groups = self.create_subgroups(num_groups, num_elements)
        self.target_types = ["cat" for _ in range(self.num_tasks)]

        # Subgroup membership
        self.digit_membership = dict()
        for target in range(self.num_classes):
            group_idxs = list()
            for idx, subgroup in enumerate(self.groups):
                if target in subgroup:
                    group_idxs.append(idx)
            self.digit_membership[target] = group_idxs

    def create_subgroups(self, num_groups, num_elements):
        t = list(range(len(self.classes)))
        random.seed(self.seed)
        random.shuffle(t)
        if num_groups == 4 and num_elements == 4:
            groups = [(t[0], t[1], t[2], t[3]),
                      (t[0], t[1], t[2], t[4]),
                      (t[5], t[6], t[7], t[8]),
                      (t[5], t[6], t[7], t[9])]
        elif num_groups == 4 and num_elements == 3:
            groups = [(t[0], t[1], t[2]),
                      (t[0], t[1], t[3]),
                      (t[5], t[6], t[7]),
                      (t[5], t[6], t[8])]
        elif num_groups == 6 and num_elements == 4:
            groups = [(t[1], t[2], t[0], t[3]),
                      (t[1], t[2], t[0], t[8]),
                      (t[1], t[2], t[4], t[9]),
                      (t[5], t[6], t[4], t[9]),
                      (t[5], t[6], t[7], t[3]),
                      (t[5], t[6], t[7], t[8])]
        elif num_groups == 1 and num_elements == 4:
            groups = [(t[0], t[1], t[2], t[3])]
        else:
            raise NotImplementedError(f"num_groups={num_groups} and num_elements={num_elements}")
        return groups

    def __getitem__(self, idx: int):
        img, digit = self.data[idx], int(self.targets[idx])
        groups = self.digit_membership[digit]
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        target = torch.zeros(self.num_tasks)
        target[groups] = 1

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img.view(*self.img_shape), target.long()

    @property
    def num_tasks(self):
        return len(self.groups)

    def get_model_kwargs(self):
        input_shape, target_dims = self.img_shape, [2] * self.num_tasks
        return dict(
            input_shape=input_shape,
            target_names=self.groups,
            target_dims=target_dims,
            target_types=self.target_types,
            # num_tasks=self.num_tasks
        )


class GroupPositionMNIST(GroupMembershipMNIST):
    """ Toy example for multi-class multitask classification """

    def __init__(
            self,
            num_groups: int = 4,
            num_elements: int = 4,
            root: str = "datasets/",
            train: bool = True,
            transform: Optional[Callable] = None,
            download: bool = False,
            seed: int = 0
    ):
        super().__init__(num_groups, num_elements, root, train, transform, download, seed)
        self.target_types = ["cat" for _ in range(self.num_tasks)]

    def __getitem__(self, idx: int):
        img, digit = self.data[idx], int(self.targets[idx])
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        target = torch.zeros(self.num_groups, dtype=torch.int64)
        if self.num_groups == 16:
            g_idxs, pos_idxs = np.where(np.array(self.groups[:6]) == digit)
            target[6 + digit] = 1
        elif self.num_groups == 6 or self.num_groups == 4 or self.num_groups == 1:
            g_idxs, pos_idxs = np.where(np.array(self.groups) == digit)
        else:
            raise ValueError

        target[g_idxs] = torch.tensor(pos_idxs + 1, dtype=torch.int64)

        return img.view(*self.img_shape), target.long()

    def get_model_kwargs(self):
        input_shape, target_dims = self.img_shape, [len(sg) + 1 for sg in self.groups]
        return dict(
            input_shape=input_shape,
            target_names=self.groups,
            target_dims=target_dims,
            target_types=self.target_types,
            # num_tasks=self.num_tasks
        )


class MultiMNIST(Dataset):
    """
     References:
        https://arxiv.org/abs/1710.09829
        https://olaralex.com/capsule-networks/
        https://github.com/qbeer/capsule-net/blob/master/src/capsule_net/multimnist_dataset.py
    """

    def __init__(
            self,
            root,
            transforms=None,
            target_transform=None,
            image_transforms=None,
            train=True,
            deterministic=False
    ):
        self.deterministic = deterministic
        if self.deterministic:
            self.seed = 42
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

        # self.transforms = transforms
        self.transforms = torchvision.transforms.ToTensor()
        self.target_transform = target_transform
        self.image_transforms = torchvision.transforms.Compose(transforms=[
            torchvision.transforms.RandomAffine(
                degrees=(-5, 5),
                translate=(.2, .2),
                scale=(.5, .7)
            ),
            torchvision.transforms.RandomCrop(size=(26, 26)),
            torchvision.transforms.Resize(size=(28, 28))
        ])
        self.train = train
        self.mnist = torchvision.datasets.MNIST(
            root=root, download=True, train=self.train
        )

        self.img_shape = (784,)
        self.num_tasks = 2
        self.target_types = ["cat", "cat"]

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        index1 = index
        index2 = random.randint(0, len(self) - 1)

        image1, label1 = self.mnist[index1]
        image2, label2 = self.mnist[index2]

        if self.deterministic:
            random.seed(self.seed)

        while label1 == label2:
            index2 = random.randint(0, len(self) - 1)
            image2, label2 = self.mnist[index2]

        if self.deterministic:
            random.seed(self.seed)

        if self.image_transforms:
            image1 = self.image_transforms(image1)
            image2 = self.image_transforms(image2)

        x = np.array(image1)
        y = np.array(image2)
        blend = np.where(x > y, x, y)
        image = Image.fromarray(blend)

        if self.deterministic:
            random.seed(self.seed)

        if self.transforms:
            image = self.transforms(image)

        label = np.zeros(shape=(1, 10), dtype=np.float32)
        label[:, label1] = 1
        label[:, label2] = 1

        if self.deterministic:
            random.seed(self.seed)

        if self.target_transform:
            label = self.target_transform(label)

        return image.view(*self.img_shape), label.squeeze().astype(int)

    def get_model_kwargs(self):
        input_shape, target_dims = self.img_shape, (2, 2)
        return dict(
            input_shape=input_shape,
            target_names=["Digit 1", "Digit 2"],
            target_dims=target_dims,
            target_types=self.target_types,
            # num_tasks=self.num_tasks
        )


class GroupedMNISTDataModule(pl.LightningDataModule):
    name = "grouped_mnist"

    def __init__(
            self,
            num_groups: int = 4,
            num_elements: int = 3,
            kind: str = "pos",
            val_prop: float = 0.1,
            data_dir: str = "datasets/",
            num_workers: int = 4,
            batch_size: int = 128,
            pin_memory: bool = False,
            seed: int = 0,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        assert 0 <= val_prop <= 1
        assert batch_size > 1
        self.num_groups = num_groups
        self.num_elements = num_elements
        self.data_dir = data_dir
        self.val_prop = val_prop
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

        self.dataset_cls = GroupPositionMNIST if kind == "pos" else GroupMembershipMNIST
        self.datasets = dict()

        self.transform = TRANSFORMS

        self.dl_kwargs = dict(batch_size=self.batch_size,
                              pin_memory=self.pin_memory,
                              drop_last=True,
                              num_workers=self.num_workers)

    def setup(self, stage: Optional[str] = None) -> None:
        #  Create preprocessed datasets for all splits
        ds_kwargs = dict(
            num_groups=self.num_groups,
            num_elements=self.num_elements,
            root=self.data_dir,
            download=True,
            seed=self.seed
        )
        train = self.dataset_cls(train=True, transform=self.transform, **ds_kwargs)
        test = self.dataset_cls(train=False, transform=T.ToTensor(), **ds_kwargs)

        train_len = int((1 - self.val_prop) * len(train))
        lengths = [train_len, len(train) - train_len]
        train, val = random_split(train, lengths)

        self.datasets["train"] = train
        self.datasets["val"] = val
        self.datasets["test"] = test

    def train_dataloader(self):
        return DataLoader(self.datasets["train"], **self.dl_kwargs)

    def val_dataloader(self):
        return DataLoader(self.datasets["val"], shuffle=False, **self.dl_kwargs)

    def test_dataloader(self):
        return DataLoader(self.datasets["test"], shuffle=False, **self.dl_kwargs)

    def predict_dataloader(self):
        return DataLoader(self.datasets["test"], shuffle=False, **self.dl_kwargs)

    def get_model_kwargs(self):
        if "train" not in self.datasets:
            self.setup()
        return self.datasets["train"].dataset.get_model_kwargs()


class MultiMNISTDataModule(pl.LightningDataModule):
    name = "multimnist"

    def __init__(
            self,
            val_prop: float = 0.1,
            data_dir: str = "datasets/",
            num_workers: int = 42,
            batch_size: int = 32,
            pin_memory: bool = False,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        assert 0 <= val_prop <= 1
        assert batch_size > 1
        self.data_dir = data_dir
        self.val_prop = val_prop
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.dataset_cls = MultiMNIST
        self.datasets = dict()

        self.transform = TRANSFORMS

        self.dl_kwargs = dict(
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            drop_last=True,
            num_workers=self.num_workers
        )

    def setup(self, stage: Optional[str] = None) -> None:
        #  Create preprocessed datasets for all splits
        kwargs = dict(root=self.data_dir)
        train = self.dataset_cls(train=True, **kwargs)
        test = self.dataset_cls(train=False, **kwargs)

        train_len = int((1 - self.val_prop) * len(train))
        lengths = [train_len, len(train) - train_len]
        train, val = random_split(train, lengths)

        self.datasets["train"] = train
        self.datasets["val"] = val
        self.datasets["test"] = test

    def train_dataloader(self):
        return DataLoader(self.datasets["train"], **self.dl_kwargs)

    def val_dataloader(self):
        return DataLoader(self.datasets["val"], shuffle=False, **self.dl_kwargs)

    def test_dataloader(self):
        return DataLoader(self.datasets["test"], shuffle=False, **self.dl_kwargs)

    def predict_dataloader(self):
        return DataLoader(self.datasets["test"], shuffle=False, **self.dl_kwargs)

    def get_model_kwargs(self):
        if "train" not in self.datasets:
            self.setup()
        return self.datasets["train"].dataset.get_model_kwargs()
