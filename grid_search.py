import argparse
from functools import partial
from pathlib import Path

import optuna

from configs.grid_search_configs import SearchSpaceConfigs
from run_fn.objectives import lesionattn_objective
from utils.hparams import get_db_storage_path
from utils.random_fn import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search LesionAttn hyperparameters on HAM10000.")
    parser.add_argument("--dataset", type=str, default="ham", choices=["ham"])
    parser.add_argument("--sensitive_name", type=str, default="sex", choices=["sex"])
    parser.add_argument("--backbone", type=str, default="resnet18-attn", choices=["resnet18-attn"])
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--db_dir", type=str, default="optuna_db")
    parser.add_argument("--model", type=str, default="lesionattn", choices=["lesionattn"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--used_ratio", type=float, default=1.0)
    parser.add_argument("--mask_used_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)

    Path(args.db_dir).mkdir(exist_ok=True, parents=True)
    search_space = SearchSpaceConfigs().lesionattn_search_space
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        storage=get_db_storage_path(args),
        sampler=sampler,
        study_name=args.model,
        load_if_exists=True,
    )
    study.optimize(partial(lesionattn_objective, args=args), n_trials=args.n_trials)


if __name__ == "__main__":
    main()
