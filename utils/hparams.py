import optuna


def get_db_storage_path(args):
    return f"sqlite:///{args.db_dir}/{args.dataset}-sex_{args.model}-{args.backbone}.db"


def load_best_hparams(model_name, db_storage_path, config):
    best_trial = get_best_trial(model_name, db_storage_path)
    for key, value in best_trial.params.items():
        setattr(config, key, value)
    return config


def get_best_trial(model_name, db_storage_path):
    study = optuna.load_study(study_name=model_name, storage=db_storage_path)
    best_trials = study.best_trials
    if len(best_trials) == 1:
        return best_trials[0]

    auprc_scores = [trial.user_attrs["val:auprc"] for trial in best_trials]
    eo_scores = [trial.user_attrs["val:eo"] for trial in best_trials]
    auprc_rank = sorted(range(len(auprc_scores)), key=lambda i: auprc_scores[i], reverse=True)
    eo_rank = sorted(range(len(eo_scores)), key=lambda i: eo_scores[i])
    combined_rank = [auprc_rank.index(i) + eo_rank.index(i) for i in range(len(best_trials))]
    return best_trials[combined_rank.index(min(combined_rank))]
