import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from configs.train_configs import LesionAttnConfig
from datasets.loader import get_dataloader, init_datasets
from models import LesionAttn
from run_fn import test_fn
from utils.misc import resolve_device
from utils.random_fn import set_random_seed


SEEDS = [51344, 45585, 96152, 79697, 18087]


def parse_args():
    parser = argparse.ArgumentParser(description="Test LesionAttn on HAM10000 or BCN20000.")
    parser.add_argument("--dataset", type=str, default="ham", choices=["ham", "bcn"])
    parser.add_argument("--sensitive_name", type=str, default="sex", choices=["sex"])
    parser.add_argument("--model", type=str, default="lesionattn", choices=["lesionattn"])
    parser.add_argument("--backbone", type=str, default="resnet18-attn", choices=["resnet18-attn"])
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--used_ratio", type=float, default=1.0)
    parser.add_argument("--mask_used_ratio", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds. Default uses the paper seeds.")
    parser.add_argument("--ham_image_dir", type=str, default=None)
    parser.add_argument("--ham_mask_dir", type=str, default=None)
    parser.add_argument("--ham_meta_fpath", type=str, default=None)
    parser.add_argument("--bcn_image_dir", type=str, default=None)
    parser.add_argument("--bcn_meta_fpath", type=str, default=None)
    return parser.parse_args()


def training_log_dir(args):
    base = f"{args.log_dir}/ham-sex({args.used_ratio})/{args.model}-{args.backbone}"
    if args.mask_used_ratio < 1.0:
        base = f"{base}-mask_{args.mask_used_ratio}"
    return Path(base)


def result_prefix(args):
    return "test_ham" if args.dataset == "ham" else "test_bcn_external"


def parse_seeds(seeds_arg):
    if seeds_arg is None:
        return SEEDS
    return [int(seed.strip()) for seed in seeds_arg.split(",") if seed.strip()]


def apply_runtime_args(train_config, args):
    train_config.device = resolve_device(args.device)
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.num_workers is not None:
        train_config.num_workers = args.num_workers
    return train_config


def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    log_dir = training_log_dir(args)
    test_results_collections = {}

    for seed in seeds:
        args.seed = seed
        set_random_seed(seed)

        weights_path = log_dir / str(seed) / "lesionattn.pth"
        state_dict = torch.load(weights_path, map_location=torch.device("cpu"))["model"]

        dataset_config, _, val_dataset, test_dataset = init_datasets(args)
        train_config = LesionAttnConfig()
        train_config.seed = seed
        train_config = apply_runtime_args(train_config, args)

        model = LesionAttn(
            backbone=args.backbone,
            num_classes=train_config.num_classes,
            pretrained=False,
            attn_type=train_config.attn_type,
        )
        model.load_state_dict(state_dict)

        val_loader = get_dataloader(
            val_dataset,
            mode="original",
            batch_size=train_config.batch_size,
            num_classes=train_config.num_classes,
            num_workers=train_config.num_workers,
            used_ratio=1.0,
        )
        test_loader = get_dataloader(
            test_dataset,
            mode="original",
            batch_size=train_config.batch_size,
            num_classes=train_config.num_classes,
            num_workers=train_config.num_workers,
            used_ratio=1.0,
        )

        pbar = tqdm(total=1, desc=f"Testing seed {seed} on {args.dataset.upper()}")
        test_result = test_fn(model, train_config, val_loader=val_loader, test_loader=test_loader)
        pbar.update(1)
        pbar.close()

        print(test_result)
        test_result["seed"] = seed
        for key, value in test_result.items():
            test_results_collections.setdefault(key, []).append(value)

    test_results_stats = {"mean": {}, "std": {}, "ci": {}}
    for key, values in test_results_collections.items():
        if key == "seed":
            continue
        mean = np.mean(values)
        std = np.std(values)
        test_results_stats["mean"][key] = mean
        test_results_stats["std"][key] = std
        test_results_stats["ci"][key] = 1.96 * std / np.sqrt(len(values))

    prefix = result_prefix(args)
    pd.DataFrame(test_results_collections).transpose().round(3).to_csv(
        log_dir / f"{prefix}_results.csv", index_label="Metric"
    )
    pd.DataFrame(test_results_stats).round(3).to_csv(
        log_dir / f"{prefix}_stats.csv", index_label="Metric"
    )


if __name__ == "__main__":
    main()
