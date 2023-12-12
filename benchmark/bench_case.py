from dataclasses import dataclass


@dataclass
class GPTCase:
    batch_size: int = 4
    seq_size: int = 1024
    num_layers: int = 1
    hidden_dim: int = 12288
    num_heads: int = 48
    dropout_rate: float = 0.0
    use_bias: bool = True
    dtype = "float32"


@dataclass
class ResNetCase:
    batch_size: int = 128


@dataclass
class GATCase:
    num_node: int = 4096
    in_feature: int = 12288
    out_feature: int = 12288
