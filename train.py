import argparse
import os

import numpy as np
import pandas as pd

from configs.train_configs import LesionAttnConfig
from datasets.loader import init_datasets
from models import LesionAttn
from run_fn import lesionattn_train_fn
from utils.hparams import get_db_storage_path, load_best_hparams
from utils.misc import resolve_device
from utils.random_fn import set_random_seed


SEEDS = [51344, 45585, 96152, 79697, 18087]


def parse_args():
    parser = argparse.ArgumentParser(description="Train LesionAttn on HAM10000.")
    parser.add_argument("--dataset", type=str, default="ham", choices=["ham"])
    parser.add_argument("--sensitive_name", type=str, default="sex", choices=["sex"])
    parser.add_argument("--model", type=str, default="lesionattn", choices=["lesionattn"])
    parser.add_argument("--backbone", type=str, default="resnet18-attn", choices=["resnet18-attn"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--use_best_hparams", action="store_true")
    parser.add_argument("--db_dir", type=str, default="optuna_db")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--used_ratio", type=float, default=1.0)
    parser.add_argument("--mask_used_ratio", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epoch_num", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds. Default uses the paper seeds.")
    parser.add_argument("--ham_image_dir", type=str, default=None)
    parser.add_argument("--ham_mask_dir", type=str, default=None)
    parser.add_argument("--ham_meta_fpath", type=str, default=None)
    return parser.parse_args()


def build_log_dir(args, seed):
    base = f"{args.log_dir}/ham-sex({args.used_ratio})/{args.model}-{args.backbone}"
    if args.mask_used_ratio < 1.0:
        base = f"{base}-mask_{args.mask_used_ratio}"
    return f"{base}/{seed}"


def parse_seeds(seeds_arg):
    if seeds_arg is None:
        return SEEDS
    return [int(seed.strip()) for seed in seeds_arg.split(",") if seed.strip()]


def apply_runtime_args(train_config, args):
    train_config.device = resolve_device(args.device)
    if args.epoch_num is not None:
        train_config.epoch_num = args.epoch_num
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.num_workers is not None:
        train_config.num_workers = args.num_workers
    return train_config


def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    val_results_collections = {}

    for seed in seeds:
        args.seed = seed
        set_random_seed(seed)

        train_config = LesionAttnConfig()
        train_config.seed = seed
        train_config.log_dir = build_log_dir(args, seed)
        train_config.used_ratio = args.used_ratio

        if args.use_best_hparams:
            train_config = load_best_hparams(args.model, get_db_storage_path(args), train_config)
        train_config = apply_runtime_args(train_config, args)

        _, train_dataset, val_dataset, _ = init_datasets(args)
        model = LesionAttn(
            backbone=args.backbone,
            num_classes=train_config.num_classes,
            pretrained=args.pretrained,
            attn_type=train_config.attn_type,
        )
        val_result = lesionattn_train_fn(
            model,
            train_config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            write_log=True,
        )

        val_result["seed"] = seed
        for key, value in val_result.items():
            val_results_collections.setdefault(key, []).append(value)

    val_results_stats = {"mean": {}, "ci": {}}
    for key, values in val_results_collections.items():
        if key == "seed":
            continue
        mean = np.mean(values)
        std = np.std(values)
        val_results_stats["mean"][key] = mean
        val_results_stats["ci"][key] = 1.96 * std / np.sqrt(len(values))

    result_dir = os.path.dirname(build_log_dir(args, seeds[-1]))
    pd.DataFrame(val_results_collections).transpose().round(3).to_csv(
        f"{result_dir}/val_results_collections.csv", index_label="Metric"
    )
    pd.DataFrame(val_results_stats).round(3).to_csv(
        f"{result_dir}/val_results_stats.csv", index_label="Metric"
    )


if __name__ == "__main__":
    main()
