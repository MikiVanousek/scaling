from dataclasses import dataclass

@dataclass
class ModelShape:
    layers: int
    d_model: int
    n_heads: int
    d_vocab: int

    def __repr__(self):
        return '\n'.join(f"{k}: {v}" for k, v in self.__dict__.items())
@dataclass
class DoclangConfig:
    dataset: str
    model_shape: ModelShape

    wandb_entity: str
    wandb_project: str

    # Arguments related to training
    batch_size: int
    seq_len: int
    tokens: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_warmup_steps: int
    weight_decay: float
    eval_interval: int
    validation_batch_size: int

    def __repr__(self):
        return '\n'.join(f"{k}: {v}" for k, v in self.__dict__.items())
