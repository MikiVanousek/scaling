import math
from dataclasses import dataclass, field
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
    ffw_size: int = 4

    d_head: int = field(init=False)

    def __post_init__(self):
        self.d_head = self.d_model // self.n_heads

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    def calculate_flops_per_token(self, context_length: int):
        # ---- Embeddings ----
        embeddings = 2 * context_length * self.d_model * self.d_vocab
        # ---- Attention ----
        # Q, K, V projections
        qkv = 2 * 3 * context_length * self.d_model * (self.d_head * self.n_heads)

        # Key @ Query logits
        kq = 2 * context_length * context_length * (self.d_head * self.n_heads)

        # Softmax
        softmax = 3 * self.n_heads * context_length * context_length

        # Softmax reductions
        softmax_red = 2 * context_length * context_length * (self.d_head * self.n_heads)

        # Final linear
        final_linear = 2 * context_length * (self.d_head * self.n_heads) * self.d_model

        attention = qkv + kq + softmax + softmax_red + final_linear

        # ---- Dense block ----
        dense = (
            2
            * context_length
            * (self.d_model * self.ffw_size + self.d_model * self.ffw_size)
        )

        # ---- Final logits ----
        logits = 2 * context_length * self.d_model * self.d_vocab

        # ---- Total FLOPs ----
        total = embeddings + self.layers * (attention + dense) + logits
        assert total % context_length == 0
        return total // context_length  # per token

    @property
    def num_params(self) -> int:
        """
        Calculates the approximate number of parameters.
        Assumptions:
        1. Includes biases for all Linears and LayerNorms.
        2. Standard FFW (not SwiGLU).
        3. Weight tying: Output head shares weights with input embedding.
        """

        # 1. Embedding Layer (Input only, assuming output is tied)
        # Size: [d_vocab, d_model]
        embedding_params = self.d_vocab * self.d_model

        # 2. Per-Layer Parameters
        # A. Attention: W_q, W_k, W_v, W_o
        # Each is d_model x d_model. Plus biases (d_model).
        # Total: 4 matrices + 4 biases
        attn_weights = 4 * (self.d_model ** 2)
        attn_biases = 4 * self.d_model

        # B. Feed Forward (MLP)
        # Hidden dimension size
        d_ff = self.d_model * self.ffw_size

        # Up Projection: [d_model, d_ff] + Bias [d_ff]
        # Down Projection: [d_ff, d_model] + Bias [d_model]
        ffw_weights = 2 * (self.d_model * d_ff)
        ffw_biases = d_ff + self.d_model

        # C. Layer Normalization
        # Usually 2 per layer (pre-attn, pre-ffw).
        # Each has Scale (gamma) + Shift (beta) of size d_model.
        ln_params = 2 * (2 * self.d_model)

        # Sum of one block
        block_params = attn_weights + attn_biases + ffw_weights + ffw_biases + ln_params

        unembedding_params = self.d_model * self.d_vocab
        # 3. Final Layer Norm (Before output)
        final_ln_params = 2 * self.d_model


        # Total
        return embedding_params + (self.layers * block_params) + unembedding_params + final_ln_params


@dataclass
class DoclangConfig:
    train_dataset: Union[dict, DatasetConfig]
    validation_datasets: list[Union[dict, DatasetConfig]]
    model_shape: ModelShape

    wandb_entity: str
    wandb_project: str

    # Arguments related to training
    batch_size: int
    context_length: int

    learning_rate: float
    lr_warmup_steps: int
    weight_decay: float
    validation_batch_size: int
    tokens: Optional[int] = None
    target_compute: Optional[int] = None
    eval_interval: int = -1
    hf_upload_username: Optional[str] = None

    def __post_init__(self):
        # Convert train_dataset to DatasetConfig if needed
        if isinstance(self.train_dataset, dict):
            self.train_dataset = DatasetConfig(**self.train_dataset)

        assert (self.tokens is None) ^ (self.target_compute is None), (
            "Exactly one of tokens or target_compute must be specified"
        )
        if self.target_compute is not None:
            fpt = self.model_shape.calculate_flops_per_token(self.context_length)
            self.tokens = self.target_compute // fpt

        # Convert validation_datasets to DatasetConfig objects
        converted_val_datasets = []
        for val_dataset in self.validation_datasets:
            if isinstance(val_dataset, dict):
                converted_val_datasets.append(DatasetConfig(**val_dataset))
            else:
                converted_val_datasets.append(val_dataset)
        self.validation_datasets = converted_val_datasets

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())
