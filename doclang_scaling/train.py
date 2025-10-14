import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, PreTrainedTokenizerFast, get_cosine_schedule_with_warmup
from datasets import load_dataset, Dataset
import wandb
import argparse
import yaml
import numpy as np
from itertools import cycle

from doclang_scaling.config import DoclangConfig
from doclang_scaling.transformers import SimpleTransformer


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate(model, val_dataset_loader, n_bootstrap=5000, ci=95, criterion=nn.CrossEntropyLoss()):
    model.eval()

    batch_losses = []
    with torch.no_grad():
        for batch in val_dataset_loader:
            input_ids = batch['input_ids'].to(next(model.parameters()).device)
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

def calculate_flops_per_token(seq_len, d_model, n_heads, layers, d_vocab, ffw_size=4):


    key_size = d_model // n_heads

    # ---- Embeddings ----
    embeddings = 2 * seq_len * d_model * d_vocab
    # ---- Attention ----
    # Q, K, V projections
    qkv = 2 * 3 * seq_len * d_model * (key_size * n_heads)

    # Key @ Query logits
    kq = 2 * seq_len * seq_len * (key_size * n_heads)

    # Softmax
    softmax = 3 * n_heads * seq_len * seq_len

    # Softmax reductions
    softmax_red = 2 * seq_len * seq_len * (key_size * n_heads)

    # Final linear
    final_linear = 2 * seq_len * (key_size * n_heads) * d_model

    attention = qkv + kq + softmax + softmax_red + final_linear

    # ---- Dense block ----
    dense = 2 * seq_len * (d_model * ffw_size + d_model * ffw_size)

    # ---- Final logits ----
    logits = 2 * seq_len * d_model * d_vocab

    # ---- Total FLOPs ----
    total = embeddings + layers * (attention + dense) + logits
    assert total % seq_len == 0
    return total // seq_len  # per token

def main(config: DoclangConfig):
    flops_per_token = calculate_flops_per_token(config.training.seq_len, **config.model_shape.__dict__)

    dataset = load_dataset(config.dataset, cache_dir="/tmp")
    train_data = dataset['train']
    train_data.set_format(type="torch", columns=["input_ids"]) # type: ignore
    train_loader = DataLoader(train_data, batch_size=config.training.batch_size, shuffle=True) # type: ignore
    tokens = config.training.tokens
    batches = tokens // config.training.batch_size // config.training.seq_len

    val_data = dataset['test']
    val_data.set_format(type="torch", columns=["input_ids"]) # type: ignore
    val_loader = DataLoader(val_data, batch_size=config.training.validation_batch_size, shuffle=False) # type: ignore

    model = SimpleTransformer(**config.model_shape.__dict__, seq_len=config.training.seq_len)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model_params = model.count_params()

    run_name = f"model_{model_params/1e6:.1f}M_tokens_{config.training.tokens /1e6:.1f}M"
    wandb.init(project=config.wandb.wandb_project, entity=config.wandb.wandb_entity, name=run_name)

    print(f"Using device: {device}")
    print(f"Model parameters: {model_params/1e6:.2f}M")
    print(f"Training on {tokens/1e6}M tokens ({batches / len(train_loader)} epochs)")
    print(f"FLOPs per token: {flops_per_token}")


    print(f"Using config:'\n{config}")

    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), 
                     lr=float(config.training.learning_rate),
                     weight_decay=float(config.training.weight_decay))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.lr_warmup_steps,
        num_training_steps=batches
    )

    # Training loop
    model.train()

    batch_iterator = cycle(train_loader)
    for batch_num in range(batches):
        batch = next(batch_iterator)
        input_ids = batch["input_ids"].to(device)
        
        outputs = model(input_ids[:, :-1])
        targets = input_ids[:, 1:]
        
        loss = nn.CrossEntropyLoss()(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Log to wandb
        tokens_seen = (batch_num + 1) * config.training.batch_size * config.training.seq_len
        compute = tokens_seen * flops_per_token
        wandb.log({
            'loss': loss.item(),
            'learning_rate': scheduler.get_last_lr()[0],
            'tokens_seen': tokens_seen,
            'compute': compute,
        })

        if (batch_num + 1) % config.training.eval_interval == 0:
            loss_val, (lower, upper) = evaluate(model, val_loader)
            wandb.log({
                'val_loss': loss_val,
                'val_loss_ci_lower': lower,
                'val_loss_ci_upper': upper,
                'tokens_seen': tokens_seen,
                'compute': compute,
            })
            print(f"Batch {batch_num+1}/{batches}, Train Loss: {loss.item():.4f}, Val Loss: {loss_val:.4f} [{lower:.4f}, {upper:.4f}]")
            model.train()
        
    
    # final evaluation
    loss_val, (lower, upper) = evaluate(model, val_loader)
    wandb.log({
        'val_loss': loss_val,
        'val_loss_ci_lower': lower,
        'val_loss_ci_upper': upper,
        'tokens_seen': tokens_seen, # pyright: ignore[reportPossiblyUnboundVariable]
        'compute': compute, # pyright: ignore[reportPossiblyUnboundVariable]
        'params': model.count_params(),
    })
    print(f"Batch {batch_num+1}/{batches}, Train Loss: {loss.item():.4f}, Val Loss: {loss_val:.4f} [{lower:.4f}, {upper:.4f}]") # pyright: ignore[reportPossiblyUnboundVariable]

    wandb.finish()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='cfgs/default.yaml', help='Path to config file')
    args = argparser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    training_config = DoclangConfig(**config)
    main(training_config)
