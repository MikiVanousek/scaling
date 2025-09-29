import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, PreTrainedTokenizerFast, get_cosine_schedule_with_warmup
from datasets import load_dataset
import wandb
import argparse
import yaml
import numpy as np
from itertools import cycle

class SimpleTransformer(nn.Module):
    def __init__(self, d_vocab, d_model, n_heads, layers, seq_len):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.embedding = nn.Embedding(d_vocab, d_model)
        self.pos_embedding = nn.Embedding(seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, d_vocab)
        
    def forward(self, x):
        seq_len = x.size(1)
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
        
        x = self.embedding(x) + self.pos_embedding(pos_ids)
        x = self.transformer(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate(model, val_dataset_loader, n_bootstrap=1000, ci=95):
    model.eval()
    criterion = nn.CrossEntropyLoss()

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

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='cfgs/default.yaml', help='Path to config file')
    args = argparser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    wandb.init(project=config['wandb_project'], entity=config['wandb_entity'], config=config)
    print(f"Using config: {args.config}")


    flops_per_token = calculate_flops_per_token(config['seq_len'], **config['model_shape'])
    print(f"FLOPs per token (non-embedding): {flops_per_token}")

    dataset = load_dataset(config['dataset'], cache_dir="/tmp")
    train_data = dataset['train']
    train_data.set_format(type="torch", columns=["input_ids"])
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    tokens = config['tokens']
    batches = tokens // config['batch_size'] // config['seq_len']
    print(f"Training on {tokens/1e6}M tokens ({batches / len(train_loader)} epochs)")

    val_data = dataset['test']
    val_data.set_format(type="torch", columns=["input_ids"])
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)

    model = SimpleTransformer(**config['model_shape'], seq_len=config['seq_len'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    print(f"Model parameters: {model.count_params()/1e6:.2f}M")
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), 
                     lr=float(config['learning_rate']),
                     weight_decay=float(config['weight_decay']))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['lr_warmup_steps'],
        num_training_steps=config['batches']
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
        tokens_seen = (batch_num + 1) * config['batch_size'] * config['seq_len']
        compute = tokens_seen * flops_per_token
        wandb.log({
            'loss': loss.item(),
            'learning_rate': scheduler.get_last_lr()[0],
            'tokens_seen': tokens_seen,
            'compute': compute,
        })

        if (batch_num + 1) % config['eval_interval'] == 0:
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
        'tokens_seen': tokens_seen,
        'compute': compute,
    })
    print(f"Batch {batch_num+1}/{batches}, Train Loss: {loss.item():.4f}, Val Loss: {loss_val:.4f} [{lower:.4f}, {upper:.4f}]")

    wandb.finish()

if __name__ == "__main__":
    main()