from dataclasses import dataclass
from typing import Optional

from utils.misc import get_device


@dataclass
class LesionAttnConfig:
    seed: int = 0
    epoch_num: int = 100
    num_classes: int = 2
    num_sens: int = 2
    model_name: str = "lesionattn"
    backbone: str = "resnet18-attn"
    dataloader_mode: str = "random"
    track_mode: str = "both"
    early_stop_epochs: int = 10
    class_loss_fn: str = "ce"
    attn_loss_fn: str = "cos"
    attn_type: str = "bahdanau"
    batch_size: int = 64
    num_workers: int = 4
    class_coeff: float = 1.0
    used_ratio: float = 1.0
    model_lr: float = 1e-5
    soften_value: float = 0.5
    attn_coeff: float = 2.0
    cutoff: Optional[float] = None
    device: str = get_device()
    log_dir: str = "log/lesionattn"
