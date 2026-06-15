from dataclasses import dataclass
from typing import Optional


@dataclass
class HAMDatasetConfig:
    seed: int = 1
    image_size: int = 256
    label_type: str = "dx"
    sensitive_name: str = "sex"
    image_dir: str = "data/HAM10000/images"
    mask_dir: str = "data/HAM10000/masks"
    meta_fpath: str = "data/HAM10000/HAM10000_metadata.csv"
    max_samples: Optional[int] = None


@dataclass
class BCNDatasetConfig:
    seed: int = 1
    image_size: int = 256
    label_type: str = "dx"
    sensitive_name: str = "sex"
    image_dir: str = "data/BCN20000/images"
    meta_fpath: str = "data/BCN20000/BCN20000_metadata.csv"
    max_samples: Optional[int] = None
