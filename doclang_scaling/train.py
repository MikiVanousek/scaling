import argparse
import math
import os
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import yaml
from datasets import Dataset, load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)

import wandb
from doclang_scaling.alibi_transformer import AlibiTransformer
from doclang_scaling.config import DoclangConfig, ModelShape
from doclang_scaling.doclang_transformers import SimpleTransformer


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_safe_dataset_name(dataset_name):
    """Create a safe name for wandb logging by replacing special characters."""
    return dataset_name.replace("/", "_").replace("-", "_")


def evaluate(
    model, val_dataset_loader, n_bootstrap=5000, ci=95, criterion=nn.CrossEntropyLoss()
):
    model.eval()

    batch_losses = []
    with torch.no_grad():
        for batch in val_dataset_loader:
            input_ids = batch["input_ids"].to(next(model.parameters()).device)
            outputs = model(input_ids[:, :-1])
            targets = input_ids[:, 1:]
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            batch_losses.append(loss.item())

    batch_losses = np.array(batch_losses)

    # Bootstrapping
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(batch_losses, size=len(batch_losses), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)

    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    mean_loss = batch_losses.mean()

    return mean_loss, (lower, upper)


def main(config_path: str):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Convert model_shape dict to ModelShape object
    model_shape_dict = config_dict.pop("model_shape")
    model_shape = ModelShape(**model_shape_dict)
    config_dict["model_shape"] = model_shape

    config = DoclangConfig(**config_dict)

    flops_per_token = config.model_shape.calculate_flops_per_token(
        config.context_length
    )

    print(f"FLOPs per token: {flops_per_token}")

    # Load training dataset
    train_dataset = load_dataset(
        config.train_dataset.name, split=config.train_dataset.split, cache_dir="/tmp"
    )
    train_dataset.set_format(type="torch", columns=["input_ids"])  # type: ignore
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)  # type: ignore
    tokens = config.tokens
    batches = tokens // config.batch_size // config.context_length

    # Load validation datasets
    val_loaders = {}
    for val_dataset_config in config.validation_datasets:
        val_dataset = load_dataset(
            val_dataset_config.name, split=val_dataset_config.split, cache_dir="/tmp"
        )
        val_dataset.set_format(type="torch", columns=["input_ids"])  # type: ignore
        val_loader = DataLoader(
            val_dataset, batch_size=config.validation_batch_size, shuffle=False
        )  # type: ignore
        val_loaders[val_dataset_config.name] = val_loader

    model = AlibiTransformer(
        **config.model_shape.__dict__
    )  # , context_length=config.context_length)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_params = model.count_params()

    wandb.init(
        project=config.wandb_project, entity=config.wandb_entity, name=config_path
    )

    print(f"Using device: {device}")
    print(f"Model parameters: {model_params / 1e6:.2f}M")
    print(f"Training on {tokens / 1e6}M tokens ({batches / len(train_loader)} epochs)")
    print(f"FLOPs per token: {flops_per_token}")

    print(f"Using config:'\n{config}")

    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config.lr_warmup_steps, num_training_steps=batches
    )

    # Training loop
    model.train()

    batch_iterator = cycle(train_loader)
    for batch_num in range(batches):
        batch = next(batch_iterator)
        input_ids = batch["input_ids"].to(device)

        outputs = model(input_ids[:, :-1])
        targets = input_ids[:, 1:]

        loss = nn.CrossEntropyLoss()(
            outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1)
        )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Log to wandb
        tokens_seen = (batch_num + 1) * config.batch_size * config.context_length
        compute = tokens_seen * flops_per_token
        wandb.log(
            {
                "loss": loss.item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "tokens_seen": tokens_seen,
                "compute": compute,
            }
        )

        if config.eval_interval > 0 and (batch_num + 1) % config.eval_interval == 0:
            log_data = {
                "tokens_seen": tokens_seen,
                "compute": compute,
            }

            val_losses_str = []
            for val_dataset_name, val_loader in val_loaders.items():
                loss_val, (lower, upper) = evaluate(model, val_loader)
                safe_name = create_safe_dataset_name(val_dataset_name)
                log_data.update(
                    {
                        f"val_loss_{safe_name}": loss_val,
                        f"val_loss_ci_lower_{safe_name}": lower,
                        f"val_loss_ci_upper_{safe_name}": upper,
                    }
                )
                val_losses_str.append(
                    f"{val_dataset_name}: {loss_val:.4f} [{lower:.4f}, {upper:.4f}]"
                )

            wandb.log(log_data)
            print(
                f"Batch {batch_num + 1}/{batches}, Train Loss: {loss.item():.4f}, Val Losses: {', '.join(val_losses_str)}"
            )
            model.train()

    # final evaluation
    final_log_data = {
        "tokens_seen": tokens_seen,  # pyright: ignore[reportPossiblyUnboundVariable]
        "compute": compute,  # pyright: ignore[reportPossiblyUnboundVariable]
        "params": model.count_params(),
        "chunk_size": config.context_length,
    }

    final_val_losses_str = []
    for val_dataset_name, val_loader in val_loaders.items():
        loss_val, (lower, upper) = evaluate(model, val_loader)
        safe_name = create_safe_dataset_name(val_dataset_name)
        final_log_data.update(
            {
                f"final_val_loss_{safe_name}": loss_val,
                f"final_val_loss_ci_lower_{safe_name}": lower,
                f"final_val_loss_ci_upper_{safe_name}": upper,
            }
        )
        final_val_losses_str.append(
            f"{val_dataset_name}: {loss_val:.4f} [{lower:.4f}, {upper:.4f}]"
        )

    wandb.log(final_log_data)
    print(
        f"Final - Batch {batch_num + 1}/{batches}, Train Loss: {loss.item():.4f}, Val Losses: {', '.join(final_val_losses_str)}"
    )  # pyright: ignore[reportPossiblyUnboundVariable]

    wandb.finish()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config", type=str, default="cfgs/default.yaml", help="Path to config file"
    )
    args = argparser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found")

    # Is it a directory?
    if os.path.isdir(args.config):
        cfg_files = os.listdir(args.config)
        for cfg_file in cfg_files:
            if cfg_file.endswith(".yaml"):
                cfg_path = os.path.join(args.config, cfg_file)
                main(cfg_path)
    else:
        main(args.config)
