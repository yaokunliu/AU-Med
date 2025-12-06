import os
import torch
import numpy as np
from typing import Dict
from .estimator import Estimator

class AUProbeEstimator(Estimator):
    """
    AUProbeEstimator

    Estimate uncertainty based on a pretrained Linear Probe (w, b) 
    that separates clear vs. fuzzy (aleatoric) samples in hidden states.

    The linear probe is trained with K-fold cross-validation to 
    distinguish hidden states of clear vs. fuzzy (aleatoric) inputs.

    The model computes:
        uncertainty = sigmoid(<h, w> + b)

    Args:
        base_root (str): Root directory containing saved linear probe results
        model (str): Model name (used for folder structure)
        layer (int): Layer index whose linear probe weights to load
    """

    def __init__(
        self,
        base_root: str,
        layer: int,
    ):
        dependencies = ["input_last_hidden"]
        super().__init__(dependencies, "sequence")
        
        probe_path = os.path.join(base_root, f"linearprobe_layer_{layer}.pt")

        if not os.path.exists(probe_path):
            raise FileNotFoundError(f"LinearProbe file not found: {probe_path}")

        state = torch.load(probe_path, map_location="cpu")
        if "w" not in state or "b" not in state:
            raise KeyError(f"LinearProbe file missing 'w' or 'b': {probe_path}")

        self.w = state["w"].float()
        self.b = torch.tensor(state["b"], dtype=torch.float32)
        self.layer = layer

        print(f"✅ Loaded LinearProbe (layer={layer}) from {probe_path}")

    def __str__(self):
        return f"AUProbeEstimator(L{self.layer})"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute uncertainty for each sample:
            uncertainty = sigmoid(<h, w> + b)
        """
        input_last = stats["input_last_hidden"]  # [B, H]
        if isinstance(input_last, np.ndarray):
            input_last = torch.from_numpy(input_last)
        input_last = input_last.float()

        logits = torch.matmul(input_last, self.w) + self.b
        aleatoric_uncertainty = torch.sigmoid(logits).cpu().numpy() # [B,]

        return aleatoric_uncertainty
