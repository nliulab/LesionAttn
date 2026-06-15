import torch
import os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import PIL

from configs.dataset_configs import BCNDatasetConfig
from utils.mapper import map_sex
from utils.random_fn import random_split_dataset
from datasets.my_transforms import get_image_transform


class BCNImageDataset(Dataset):
    def __init__(self, config: BCNDatasetConfig,
                 set_type, image_transform=None):
        self.config = config
        self.image_dir = self.config.image_dir
        self.meta_fpath = self.config.meta_fpath
        self.set_type = set_type

        if image_transform is None:
            self.image_transform = get_image_transform(self.config.image_size)
        else:
            self.image_transform = image_transform

        self.sensitive_name = self.config.sensitive_name
        if self.sensitive_name != "sex":
            raise ValueError("LesionAttn only supports sex as the sensitive attribute.")
        self.sensitives_map_fn = map_sex

        self.label_type = self.config.label_type

        image_names, image_paths, sensitives, labels = self._load_data()
        self.image_names, self.image_paths, self.sensitives, self.labels = random_split_dataset(
           image_names, image_paths, sensitives, labels, set_type=self.set_type, seed=self.config.seed)
        max_samples = getattr(self.config, "max_samples", None)
        if max_samples is not None:
            self.image_names = self.image_names[:max_samples]
            self.image_paths = self.image_paths[:max_samples]
            self.sensitives = self.sensitives[:max_samples]
            self.labels = self.labels[:max_samples]

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = PIL.Image.open(img_path, mode='r').convert('RGB')
        image = self.image_transform(image)

        y = self.labels[index]

        a = self.sensitives[index]
        a = self.sensitives_map_fn(a)
        return image, y, a

    def get_labels(self):
        return self.labels

    def get_sensitives(self):
        return self.sensitives

    def get_image_size(self):
        return self.config.image_size

    def __len__(self):
        return len(self.image_paths)

    def _load_data(self):
        meta_data = pd.read_csv(self.meta_fpath,
                                usecols=['image', self.sensitive_name, "sex", "dx"],
                                encoding="utf-8")
        meta_data = meta_data[meta_data["sex"].notna()]
        meta_data = meta_data[meta_data["sex"] != "unknown"]

        image_names = meta_data["image"].values
        image_paths = np.array([os.path.join(self.image_dir, f"{image_name}.jpg") for image_name in image_names])
        sensitives = meta_data[self.sensitive_name].values
        labels = meta_data["dx"].values
        labels = torch.tensor([1 if label in ['ak', "bcc", "mel", "scc"] else 0 for label in labels])

        if self.config.max_samples is not None:
            rng = np.random.default_rng(self.config.seed)
            indices = rng.permutation(len(image_names))[:self.config.max_samples]
            image_names = image_names[indices]
            image_paths = image_paths[indices]
            sensitives = sensitives[indices]
            labels = labels[indices]

        return image_names, image_paths, sensitives, labels
