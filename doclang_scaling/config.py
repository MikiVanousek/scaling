from dataclasses import dataclass

class NicePrint:
    def __repr__(self):
        return '\n'.join(f"{k}: {v}" for k, v in self.__dict__.items())
@dataclass
class ModelShape(NicePrint):
    layers: int
    d_model: int
    n_heads: int
    d_vocab: int

@dataclass
class TrainingConfig(NicePrint):
    batch_size: int
    seq_len: int
    tokens: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_warmup_steps: int
    weight_decay: float
    eval_interval: int
    validation_batch_size: int

@dataclass
class WandbConfig(NicePrint):
    wandb_entity: str
    wandb_project: str

@dataclass
class DoclangConfig(NicePrint):
    dataset: str
    model_shape: ModelShape
    training: TrainingConfig
    wandb: WandbConfig
