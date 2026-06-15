from dataclasses import dataclass


@dataclass
class SearchSpaceConfigs:
    lesionattn_search_space = {
        "model_lr": [1e-5],
        "attn_coeff": [0.5, 1.0, 2.0],
        "soften_value": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    }
