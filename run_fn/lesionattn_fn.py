import torch
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
import torch.nn.functional as F

from models.lesionattn import LesionAttn
from datasets import get_dataloader
from utils.metrics import get_selecting_metrics
from utils.mapper import label2onehot
from configs.train_configs import LesionAttnConfig
from utils.loss_fn import cross_entropy_loss_fn, focal_loss_fn, l1_loss_fn, l2_loss_fn, cos_loss_fn
from models.tracker import PerformanceTracker, ParetoFrontierTracker
from run_fn.test_fn import _test_loop


def lesionattn_train_fn(model: LesionAttn, train_config: LesionAttnConfig,
                        train_dataset, val_dataset, write_log=True):
    print(f"Start Training on Seed {train_config.seed}")
    model = model.to(train_config.device)
    if train_config.track_mode == "both":
        tracker = ParetoFrontierTracker(early_stop_epochs=train_config.early_stop_epochs)
    elif train_config.track_mode == "performance":
        tracker = PerformanceTracker(early_stop_epochs=train_config.early_stop_epochs)
    else:
        raise ValueError(f"Invalid track mode {train_config.track_mode}")

    train_loader = get_dataloader(train_dataset, mode=train_config.dataloader_mode,
                                  batch_size=train_config.batch_size,
                                  num_classes=train_config.num_classes,
                                  num_workers=train_config.num_workers,
                                  used_ratio=train_config.used_ratio)

    val_loader = get_dataloader(val_dataset, mode="original",
                                batch_size=train_config.batch_size,
                                num_workers=train_config.num_workers,
                                num_classes=train_config.num_classes)

    if write_log:
        log_dir = train_config.log_dir
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    model_optim = torch.optim.Adam(model.parameters(), lr=train_config.model_lr)
    model_optim_scheduler = torch.optim.lr_scheduler.StepLR(model_optim, step_size=10, gamma=0.99)

    if train_config.class_loss_fn == "ce":
        class_loss_fn = cross_entropy_loss_fn
    elif train_config.class_loss_fn == "focal":
        class_loss_fn = focal_loss_fn
    else:
        raise ValueError(f"Invalid class loss function {train_config.class_loss_fn}")
    
    if train_config.attn_loss_fn == "l1":
        attn_loss_fn = l1_loss_fn
    elif train_config.attn_loss_fn == "l2":
        attn_loss_fn = l2_loss_fn
    elif train_config.attn_loss_fn == "cos":
        attn_loss_fn = cos_loss_fn
    else:
        raise ValueError(f"Invalid attention loss function {train_config.attn_loss_fn}")

    device = train_config.device

    epoch_num = train_config.epoch_num
    num_classes = train_config.num_classes
    num_sens = train_config.num_sens

    for i_epoch in range(epoch_num):
        _train_loop(model, model_optim, model_optim_scheduler,
                   loader=train_loader, device=device, writer=writer,
                   class_loss_fn=class_loss_fn, attn_loss_fn=attn_loss_fn,
                   class_loss_coeff=train_config.class_coeff, attn_loss_coeff=train_config.attn_coeff,
                   num_classes=num_classes, num_sens=num_sens, current_epoch=i_epoch,
                   soften_value=train_config.soften_value)
        metric_dict = _test_loop(model, loader=val_loader, device=device,
                                 num_classes=num_classes, num_sens=num_sens)
        metric_dict = get_selecting_metrics(metric_dict)
        model_state_dict = {"model": deepcopy(model.state_dict())}

        continue_flag = tracker.update(metric_dict, model_state_dict)
        if not continue_flag:
            break
        

    state_dict = tracker.export_best_model_state_dict()
    best_metric_dict = tracker.export_best_metric_dict()
    model.load_state_dict(state_dict["model"])

    # write log
    if write_log:
        torch.save(state_dict, f"{log_dir}/{train_config.model_name}.pth")
        with open(f"{log_dir}/val_log.txt", "w") as f:
            for key, value in best_metric_dict.items():
                f.write(f"{key}: {value}\n")

        writer.close()

    return best_metric_dict


def _train_loop(model: LesionAttn, model_optim, model_optim_scheduler,
                loader, writer, class_loss_fn, attn_loss_fn,
                class_loss_coeff: float, attn_loss_coeff: float,
                device: str, num_classes: int, num_sens: int, current_epoch: int,
                soften_value: float):
    print("Training Loop")
    model.train()

    num_step = current_epoch * len(loader)

    pbar = tqdm(total=len(loader), desc=f"Epoch {current_epoch}")
    for i_step, (images, labels, sensitives, masks, image_names) in enumerate(loader):
        labels = label2onehot(labels, num_classes=num_classes)
        sensitives = label2onehot(sensitives, num_classes=num_sens)
        images = images.to(device)
        labels = labels.to(device)
        sensitives = sensitives.to(device)
        masks = masks.to(device)
        
        batch_size = images.shape[0]
        is_allzero = ~masks.view(batch_size, -1).any(dim=1)

        masks = torch.where(masks == 0, soften_value, masks)

        y_logits, hidden = model.forward_with_hidden(images)
        attn_map = hidden["attn"]

        mask_size = [attn_map.shape[2], attn_map.shape[3]]
        masks = F.interpolate(masks, size=mask_size, mode="bilinear")
        
        masks = torch.softmax(masks.view(batch_size, -1), dim=1)
        masks = masks.view(batch_size, 1, mask_size[0], mask_size[1])
        for i in range(batch_size):
            if is_allzero[i]:
                zero_mask = torch.zeros_like(masks[i])
                masks[i] = zero_mask

        class_loss = class_loss_fn(y_logits, labels)
        attn_loss = attn_loss_fn(attn_map, masks, is_allzero=is_allzero)
        loss = class_loss_coeff * class_loss + attn_loss_coeff * attn_loss

        model_optim.zero_grad(set_to_none=True)
        loss.backward()
        model_optim.step()
        lr = model_optim.param_groups[0]['lr']
        model_optim_scheduler.step()

        if writer is not None:
            log_dict = {"class_loss": loss.item(),
                        "attn_loss": attn_loss.item(),
                        "lr": lr}
            for key, value in log_dict.items():
                writer.add_scalar(f"train/{key}", value, num_step)

        num_step += 1
        pbar.update()
        pbar.write(f"Step {i_step+1}:Loss {loss.item()}")
