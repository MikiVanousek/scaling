from dataclasses import dataclass
from typing import Union


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
    tokens: int
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

    def __repr__(self):
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())
