from typing import Optional, Callable
from pathlib import Path

import numpy as np
import pandas as pd

from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
from PIL import Image
import torch

import pytorch_lightning as pl


def split_train_val(dataset, val_prop=0.2):
    assert 0 < val_prop < 1
    num_samples = len(dataset)
    idxs = list(range(num_samples))
    np.random.shuffle(idxs)
    train_end = int((1 - val_prop) * num_samples)
    train_idxs = idxs[:train_end]
    val_idxs = idxs[train_end:]
    assert len(train_idxs) + len(val_idxs) == num_samples, "the split is not valid"
    return train_idxs, val_idxs


class MTFL(Dataset):

    def __init__(
            self,
            data_file
    ):
        super().__init__()
        self.root_dir = Path(data_file).parent
        try:
            self.df = pd.read_csv(data_file, sep=" ", header=None, skipinitialspace=True, skipfooter=1, engine="python")
        except FileNotFoundError:
            raise FileNotFoundError("Download datamodule from:"
                                    "https://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html"
                                    " to 'datasets/MTFL/'")

        self.df.iloc[:, 0] = self.df.iloc[:, 0].str.replace("\\", "/", regex=False)  # Pandas 1.3.4

        self.transforms = transforms.Compose([
            # transforms.Resize((150, 150)),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        # self.img_shape = (3, 150, 150)
        self.img_shape = (3, 256, 256)
        self._target_dims = (2, 2, 2, 5)
        self._num_tasks = len(self._target_dims)
        self._task_names = ("gender", "smile", "glasses", "head pose")
        self._task_types = ("cat", "cat", "cat", "cat")

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        img_name = self.root_dir / str(item[0])
        labels = (item[11:] - 1)  # 1-indexed to 0-indexed
        img = Image.open(img_name).convert("RGB")
        img = self.transforms(img)
        return img, torch.from_numpy(np.array(labels, dtype=np.float32)).long()

    def __len__(self):
        return len(self.df)

    def get_model_kwargs(self):
        input_shape, target_dims = self.img_shape, self._target_dims
        return dict(
            input_shape=input_shape,
            target_dims=target_dims,
            target_types=self._task_types,
            target_names=self._task_names,
        )


class MTFLDataModule(pl.LightningDataModule):
    name = "mtfl"

    def __init__(
            self,
            data_dir: str = "datasets/",
            val_prop: float = 0.1,
            num_workers: int = 4,
            batch_size: int = 128,
            pin_memory: bool = False,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        assert 0 <= val_prop <= 1
        assert batch_size > 1
        self.root_dir = Path(data_dir) / "MTFL"
        self.data_dir = data_dir
        self.val_prop = val_prop
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataloaders = dict()
        self.dataset_cls = MTFL

    def setup(self, stage: Optional[str] = None) -> None:
        #  Create preprocessed datasets for all splits
        train_dataset = self.dataset_cls(data_file=self.root_dir / "training.txt")
        valid_dataset = self.dataset_cls(data_file=self.root_dir / "training.txt")
        test_dataset = self.dataset_cls(data_file=self.root_dir / "testing.txt")

        train_idxs, val_idxs = split_train_val(train_dataset, val_prop=self.val_prop)

        train_sampler = SubsetRandomSampler(train_idxs)
        val_sampler = SubsetRandomSampler(val_idxs)

        dl_kwargs = dict(
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            drop_last=True,
            num_workers=self.num_workers
        )
        self.dataloaders["train"] = DataLoader(train_dataset, sampler=train_sampler, **dl_kwargs)
        self.dataloaders["val"] = DataLoader(valid_dataset, sampler=val_sampler, **dl_kwargs)
        self.dataloaders["test"] = DataLoader(test_dataset, shuffle=False, **dl_kwargs)

    def train_dataloader(self):
        return self.dataloaders["train"]

    def val_dataloader(self):
        return self.dataloaders["val"]

    def test_dataloader(self):
        return self.dataloaders["test"]

    def predict_dataloader(self):
        pass

    def get_model_kwargs(self):
        if "train" not in self.dataloaders:
            self.setup()
        return self.dataloaders["train"].dataset.get_model_kwargs()
