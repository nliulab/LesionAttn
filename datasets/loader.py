from torch.utils.data import DataLoader

from configs.dataset_configs import BCNDatasetConfig, HAMDatasetConfig
from datasets import BCNImageDataset, HAMImageDataset
from .sampler import RandomLabelSampler


num_sens_dict = {"sex": 2}


def _update_ham_config_from_args(config, args):
    config.seed = args.seed
    if hasattr(args, "image_size"):
        config.image_size = args.image_size
    if hasattr(args, "max_samples"):
        config.max_samples = args.max_samples
    if hasattr(args, "ham_image_dir") and args.ham_image_dir is not None:
        config.image_dir = args.ham_image_dir
    if hasattr(args, "ham_mask_dir") and args.ham_mask_dir is not None:
        config.mask_dir = args.ham_mask_dir
    if hasattr(args, "ham_meta_fpath") and args.ham_meta_fpath is not None:
        config.meta_fpath = args.ham_meta_fpath
    return config


def _update_bcn_config_from_args(config, args):
    config.seed = args.seed
    if hasattr(args, "image_size"):
        config.image_size = args.image_size
    if hasattr(args, "max_samples"):
        config.max_samples = args.max_samples
    if hasattr(args, "bcn_image_dir") and args.bcn_image_dir is not None:
        config.image_dir = args.bcn_image_dir
    if hasattr(args, "bcn_meta_fpath") and args.bcn_meta_fpath is not None:
        config.meta_fpath = args.bcn_meta_fpath
    return config


def get_dataloader(dataset, mode: str, num_classes: int, batch_size: int,
                   num_workers: int = 4, used_ratio: float = 1.0):
    if mode == "random":
        sampler = RandomLabelSampler(labels=dataset.get_labels(),
                                     num_classes=num_classes,
                                     batch_size=batch_size,
                                     used_ratio=used_ratio)
        loader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=num_workers)
    elif mode == "original":
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    else:
        raise NotImplementedError(f"Unknown mode: {mode}")
    return loader


def init_datasets(args):
    if args.dataset == "ham":
        load_mask = args.model == "lesionattn"
        dataset_configs = _update_ham_config_from_args(HAMDatasetConfig(), args)
        train_dataset = HAMImageDataset(config=dataset_configs, set_type="train", load_mask=load_mask,
                                        mask_used_ratio=args.mask_used_ratio)
        val_dataset = HAMImageDataset(config=dataset_configs, set_type="val")
        test_dataset = HAMImageDataset(config=dataset_configs, set_type="test")
    elif args.dataset == "bcn":
        val_dataset_configs = _update_ham_config_from_args(HAMDatasetConfig(), args)
        dataset_configs = _update_bcn_config_from_args(BCNDatasetConfig(), args)
        train_dataset = None
        val_dataset = HAMImageDataset(config=val_dataset_configs, set_type="val")
        test_dataset = BCNImageDataset(config=dataset_configs, set_type="all")
    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}")

    return dataset_configs, train_dataset, val_dataset, test_dataset
