import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset
import wandb
import argparse
import yaml
import numpy as np

class SimpleTransformer(nn.Module):
    def __init__(self, d_vocab, d_model, n_heads, layers, seq_len):
        super().__init__()
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

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default='cfgs/default.yaml', help='Path to config file')
    args = argparser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    wandb.init(project=config['wandb_project'], entity=config['wandb_entity'], config=config)
    print(f"Using config: {args.config}")

    dataset = load_dataset(config['dataset'])
    train_data = dataset['train']
    train_data.set_format(type="torch", columns=["input_ids"])
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)

    val_data = dataset['test']
    val_data = val_data.select(range(config['validation_batches'] * config['validation_batch_size']))
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

    total_steps = config['samples'] // config['batch_size']
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)
    
    # Training loop
    model.train()
    step = 0
    

    for batch in train_loader:
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
        wandb.log({
            'loss': loss.item(),
            'learning_rate': scheduler.get_last_lr()[0],
            'step': step
        })
        
        step += 1
        if step % config['eval_interval'] == 0:
            loss_val, (lower, upper) = evaluate(model, val_loader)
            wandb.log({
                'val_loss': loss_val,
                'val_loss_ci_lower': lower,
                'val_loss_ci_upper': upper,
                'step': step
            })
            print(f"Step {step}: Train Loss {loss.item():.4f}, Val Loss {loss_val:.4f} [{lower:.4f}, {upper:.4f}]")
            model.train()
        
    
    wandb.finish()

if __name__ == "__main__":
    main()