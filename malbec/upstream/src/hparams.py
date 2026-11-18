from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class HParams(BaseModel):
    """Configuration for training and optimizing Ribeiro's neural network."""

    # Model Architecture
    n_leads: int = 12
    n_classes: int = 6
    seq_length: int = 4096
    net_seq_length: List[int] = Field(default=[4096, 1024, 256, 64, 16])
    net_filter_size: List[int] = Field(default=[64, 128, 196, 256, 320])
    kernel_size: int = 17
    dropout_rate: float = 0.3

    # Some Training Parameters 
    freeze: int = 3 # freeze first n resnet blocks, will not be trained
    gradient_clipping: float = 0.9

    # Default Class Weights
    weights: Dict[str, float] = Field(default={
        'stemi_hd_v2': 1.0,
        'nstemi_hd_v2': 1.0,
        'nstemi_type1_hd_v2': 1.0,
        'nstemi_type2_hd_v2': 1.0,
        'other_hd_v2' : 1.0,
        'non_hd_v2' : 1.0
        })
        
    # Paths
    ecg_path: Path = None
    target_path: Path = None

    class Config:
        arbitrary_types_allowed = True  # Allow Path objects


# Initialize hparams
hparams = HParams()
hparams_dict = hparams.dict()
