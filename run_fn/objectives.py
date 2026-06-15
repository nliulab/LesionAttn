from configs.train_configs import LesionAttnConfig
from datasets.loader import get_dataloader, init_datasets
from models import LesionAttn
from run_fn import lesionattn_train_fn, test_fn
from utils.metrics import get_selecting_metrics


def lesionattn_objective(trial, args):
    model_lr = trial.suggest_float("model_lr", 1e-5, 1e-3)
    attn_coeff = trial.suggest_float("attn_coeff", 0.1, 10.0)
    soften_value = trial.suggest_float("soften_value", 0.0, 1.0)

    train_config = LesionAttnConfig()
    train_config.seed = args.seed
    train_config.model_lr = model_lr
    train_config.attn_coeff = attn_coeff
    train_config.soften_value = soften_value

    _, train_dataset, val_dataset, test_dataset = init_datasets(args)
    val_loader = get_dataloader(
        val_dataset,
        mode="original",
        batch_size=train_config.batch_size,
        num_classes=train_config.num_classes,
        used_ratio=1.0,
    )
    test_loader = get_dataloader(
        test_dataset,
        mode="original",
        batch_size=train_config.batch_size,
        num_classes=train_config.num_classes,
        used_ratio=1.0,
    )

    model = LesionAttn(
        backbone=args.backbone,
        num_classes=train_config.num_classes,
        pretrained=args.pretrained,
        attn_type=train_config.attn_type,
    )
    lesionattn_train_fn(
        model,
        train_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        write_log=False,
    )
    metric_dict = test_fn(model, train_config, val_loader=val_loader, test_loader=test_loader)
    for metric, value in metric_dict.items():
        trial.set_user_attr(f"val:{metric}", float(value))

    selecting_metrics = get_selecting_metrics(metric_dict)
    return selecting_metrics["auprc"], selecting_metrics["eo"]
