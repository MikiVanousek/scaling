import math
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class DatasetConfig:
    name: str
    split: str = "train"


@dataclass
class ModelShape:
    layers: int
    d_model: int
    n_heads: int
    d_vocab: int

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())


def calculate_flops_per_token(seq_len, d_model, n_heads, layers, d_vocab, ffw_size=4):
    key_size = d_model // n_heads
    # Embeddings
    embeddings = 2 * seq_len * d_model * d_vocab
    # Attention
    qkv = 2 * 3 * seq_len * d_model * (key_size * n_heads)
    kq = 2 * seq_len * seq_len * (key_size * n_heads)
    softmax = 3 * n_heads * seq_len * seq_len
    softmax_red = 2 * seq_len * seq_len * (key_size * n_heads)
    final_linear = 2 * seq_len * (key_size * n_heads) * d_model
    attention = qkv + kq + softmax + softmax_red + final_linear
    # Dense block
    dense = 2 * seq_len * (d_model * ffw_size + d_model * ffw_size)
    # Final logits
    logits = 2 * seq_len * d_model * d_vocab
    total = embeddings + layers * (attention + dense) + logits
    assert total % seq_len == 0
    return total // seq_len  # per token


@dataclass
class DoclangConfig:
    train_dataset: Union[dict, DatasetConfig]
    validation_datasets: list[Union[dict, DatasetConfig]]
    model_shape: ModelShape

    wandb_entity: str
    wandb_project: str

    # Arguments related to training
    batch_size: int
    seq_len: int
    tokens: Optional[int] = None
    target_compute: Optional[int] = None
    gradient_accumulation_steps: int
    learning_rate: float
    lr_warmup_steps: int
    weight_decay: float
    eval_interval: -1
    validation_batch_size: int

    def __post_init__(self):
        # Convert train_dataset to DatasetConfig if needed
        if isinstance(self.train_dataset, dict):
            self.train_dataset = DatasetConfig(**self.train_dataset)

        # Convert validation_datasets to DatasetConfig objects
        converted_val_datasets = []
        for val_dataset in self.validation_datasets:
            if isinstance(val_dataset, dict):
                converted_val_datasets.append(DatasetConfig(**val_dataset))
            else:
                converted_val_datasets.append(val_dataset)
        self.validation_datasets = converted_val_datasets

        # Determine training target: tokens or compute
        if (self.tokens is None) and (self.target_compute is None):
            raise ValueError(
                "Config must set exactly one of 'tokens' or 'target_compute'."
            )
        if (self.tokens is not None) and (self.target_compute is not None):
            raise ValueError(
                "Config must set exactly one of 'tokens' or 'target_compute', not both."
            )

        if self.target_compute is not None:
            flops_per_token = calculate_flops_per_token(
                self.seq_len,
                d_model=self.model_shape.d_model,
                n_heads=self.model_shape.n_heads,
                layers=self.model_shape.layers,
                d_vocab=self.model_shape.d_vocab,
            )
            # tokens to meet/exceed compute target
            tokens_needed = math.ceil(self.target_compute / flops_per_token)
            # align to full batches so training loop doesn't under-shoot
            tokens_per_batch = self.batch_size * self.seq_len
            aligned_tokens = (
                math.ceil(tokens_needed / tokens_per_batch) * tokens_per_batch
            )
            self.tokens = aligned_tokens

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())
