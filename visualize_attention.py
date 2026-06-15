import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from configs.dataset_configs import BCNDatasetConfig, HAMDatasetConfig
from configs.train_configs import LesionAttnConfig
from datasets import BCNImageDataset, HAMImageDataset
from models import LesionAttn
from utils.misc import unnormalize
from utils.random_fn import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize LesionAttn attention maps.")
    parser.add_argument("--dataset", type=str, default="ham", choices=["ham", "bcn"])
    parser.add_argument("--sensitive_name", type=str, default="sex", choices=["sex"])
    parser.add_argument("--model", type=str, default="lesionattn", choices=["lesionattn"])
    parser.add_argument("--backbone", type=str, default="resnet18-attn", choices=["resnet18-attn"])
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--output_dir", type=str, default="attention_vis")
    parser.add_argument("--seed", type=int, default=51344)
    parser.add_argument("--used_ratio", type=float, default=1.0)
    parser.add_argument("--mask_used_ratio", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=16)
    return parser.parse_args()


def training_log_dir(args):
    base = f"{args.log_dir}/ham-sex({args.used_ratio})/{args.model}-{args.backbone}"
    if args.mask_used_ratio < 1.0:
        base = f"{base}-mask_{args.mask_used_ratio}"
    return Path(base)


def tensor_to_image(image_tensor):
    image = unnormalize(image_tensor.detach().cpu().clone()).clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    return (image * 255).astype(np.uint8)


def attention_to_heatmap(attn_tensor, size):
    attn = F.interpolate(attn_tensor, size=size, mode="bilinear", align_corners=False)
    attn = attn.squeeze().detach().cpu().numpy()
    attn = attn - attn.min()
    denom = attn.max()
    if denom > 0:
        attn = attn / denom

    heatmap = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    heatmap[..., 0] = (255 * attn).astype(np.uint8)
    heatmap[..., 1] = (80 * (1 - attn)).astype(np.uint8)
    return heatmap


def overlay_attention(image, heatmap, alpha=0.45):
    overlay = (image * (1 - alpha) + heatmap * alpha).clip(0, 255)
    return overlay.astype(np.uint8)


def sample_name(dataset, index):
    if hasattr(dataset, "image_names"):
        return str(dataset.image_names[index])
    return Path(dataset.image_paths[index]).stem


def load_visualization_dataset(args):
    if args.dataset == "ham":
        config = HAMDatasetConfig(seed=args.seed)
        return HAMImageDataset(config=config, set_type="test")
    if args.dataset == "bcn":
        config = BCNDatasetConfig(seed=args.seed)
        return BCNImageDataset(config=config, set_type="all")
    raise NotImplementedError(f"Unknown dataset: {args.dataset}")


@torch.no_grad()
def main():
    args = parse_args()
    set_random_seed(args.seed)

    dataset = load_visualization_dataset(args)
    output_dir = Path(args.output_dir) / args.dataset / str(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = LesionAttnConfig(seed=args.seed)
    model = LesionAttn(
        backbone=args.backbone,
        num_classes=train_config.num_classes,
        pretrained=False,
        attn_type=train_config.attn_type,
    )
    weights_path = training_log_dir(args) / str(args.seed) / "lesionattn.pth"
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))["model"]
    model.load_state_dict(state_dict)
    model.to(train_config.device)
    model.eval()

    for index in range(min(args.max_samples, len(dataset))):
        image, label, sensitive = dataset[index]
        image_batch = image.unsqueeze(0).to(train_config.device)
        _, hidden = model.forward_with_hidden(image_batch)
        image_np = tensor_to_image(image)
        heatmap = attention_to_heatmap(hidden["attn"], size=image_np.shape[:2])
        overlay = overlay_attention(image_np, heatmap)

        name = sample_name(dataset, index)
        out_path = output_dir / f"{index:04d}_{name}_y{int(label)}_sex{int(sensitive)}.png"
        Image.fromarray(overlay).save(out_path)

    print(f"Saved attention visualizations to {output_dir}")


if __name__ == "__main__":
    main()
