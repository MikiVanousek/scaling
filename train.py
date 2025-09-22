import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from datasets import load_dataset
import wandb
import argparse
import yaml

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

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    with open("training_cfg.yaml", "r") as f:
        config = yaml.safe_load(f)

    wandb.init(project=config['wandb_project'], entity=config['wandb_entity'], config=config)

    dataset = load_dataset(config['dataset'])
    train_data = dataset['train']
    
    model = SimpleTransformer(**config['model_shape'], seq_len=config['seq_len'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), 
                     lr=float(config['learning_rate']),
                     weight_decay=float(config['weight_decay']))

    total_steps = config['samples'] // config['batch_size']
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)
    
    # Training loop
    model.train()
    step = 0
    
    for i in range(0, config['samples'], config['batch_size']):
        batch_end = min(i + config['batch_size'], config['samples'])
        batch = train_data[i:batch_end]
        
        input_ids = torch.tensor(batch['input_ids']).to(device)
        
        # Forward pass
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
        
        if step % config['eval_interval'] == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
        
        step += 1
    
    wandb.finish()

if __name__ == "__main__":
    main()