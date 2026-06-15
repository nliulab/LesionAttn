import torch
import copy
import torch.nn as nn

def sample_state_dict(state_dict, num_samples:int, module_prefix:str='classify_module', random_noise=0.1):
    """
    Sample parameters of a specific module in the model.

    Args:
    state_dict -- state dictionary of the model
    num_samples -- number of samples to be generated
    module_prefix -- prefix of the model module to be sampled
    random_noise -- range of the uniform distribution to sample from, centered around the original parameter value

    Returns:
    sampled_state_dicts -- list of sampled state dictionaries
    """
    sampled_state_dicts = []

    for i in range(num_samples):
        # Create a copy of the current model parameters
        sampled_state_dict = copy.deepcopy(state_dict)
        
        # Sample only the parameters of the specified module
        for key in sampled_state_dict:
            if key.startswith(module_prefix):
                param_value = sampled_state_dict[key]
                sampled_state_dict[key] = param_value + (torch.rand(1) * 2 - 1)*random_noise*param_value

        sampled_state_dicts.append(sampled_state_dict)

    return sampled_state_dicts


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=5)
        self.classify_module = nn.Linear(10, 3)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classify_module(x)
        return x

def test_sampled_models():
    model = SimpleNet()
    num_samples = 10
    original_state_dict = model.state_dict()
    module_prefix = 'classify_module'
    random_noise = 0.1

    sampled_state_dicts = sample_state_dict(original_state_dict, num_samples, module_prefix, random_noise)
    print(f"Sampled {num_samples} state dictionaries from the original state dictionary.")


def set_grad_flag(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


def resolve_device(device: str):
    if device == "auto":
        return get_device()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is False.")
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unknown device: {device}")
    return device


def unnormalize(image: torch.Tensor):
    """
    Unnormalize a tensor image with mean and std deviation.
    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    for t, m, s in zip(image, means, stds):
        t.mul_(s).add_(m)
    return image


if __name__ == "__main__":
    test_sampled_models()
