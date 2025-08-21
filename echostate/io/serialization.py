import json
import os
import torch
from typing import Any, Tuple
from ..core.esn import ESN

def save_esn(path_prefix: str, model: ESN) -> Tuple[str, str]:
    """
    Save an ESN as <prefix>.json (config) and <prefix>.pt (state_dict).
    """
    os.makedirs(os.path.dirname(path_prefix) or ".", exist_ok=True)
    cfg_path = f"{path_prefix}.json"
    sd_path = f"{path_prefix}.pt"

    cfg = model.get_config()
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # state_dict contains reservoir buffers (W, W_in, W_bias) and W_out
    torch.save(model.state_dict(), sd_path)
    return cfg_path, sd_path

def load_esn(path_prefix: str, map_location: str | torch.device | None = None) -> ESN:
    """
    Load ESN from <prefix>.json and <prefix>.pt. If device differs, pass map_location.
    """
    cfg_path = f"{path_prefix}.json"
    sd_path = f"{path_prefix}.pt"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Construct on target device (map_location influences buffers on load)
    model = ESN(**cfg)
    state = torch.load(sd_path, map_location=map_location or model.device)
    model.load_state_dict(state, strict=False)  # strict=False tolerates minor version upgrades
    return model
