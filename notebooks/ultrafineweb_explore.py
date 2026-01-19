#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm
from datasets import load_dataset, Value, Dataset, Sequence, Features
import os
import sys
from huggingface_hub import login, HfApi


# In[2]:


# 1. Load the tokenizer
# We use the "Fast" version for significantly better performance on 1M rows
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
print(f"Number of distinct tokens in tokenizer: {tokenizer.vocab_size} ")
assert tokenizer.vocab_size < 2**16, "Tokenizer vocab size exceeds uint16 capacity."

TARGET_TOKEN_LENGTH = 8192
# TARGET_ROWS = 1_000_000
TARGET_ROWS = 1_00
HF_USERNAME = "MikiV"

# Assert HF write token in env or input


INPUT_DATASET_ID = "openbmb/Ultra-FineWeb"
OUTPUT_DATASET_NAME = f"{HF_USERNAME}/{INPUT_DATASET_ID.split('/')[-1]}-chunked-{TARGET_TOKEN_LENGTH}"
print(f"Final dataset will have approximately {TARGET_ROWS * TARGET_TOKEN_LENGTH * 2 / 1e9}GB.")


# In[3]:


# Suppress tokenization warnings (e.g., token indices sequence length is longer than the specified maximum)
# We want the true document length, not truncated to 1024
tokenizer.model_max_length = 1e9 

# 2. Load the dataset in streaming mode
# "openbmb/Ultra-FineWeb" is massive, so streaming=True prevents downloading TBs of data.
# We explicitly select the 'train' split.
streaming_dataset = load_dataset(INPUT_DATASET_ID, split="en", streaming=True)
output_rows = []

pbar = tqdm(total=TARGET_ROWS, desc="Collecting Valid Samples", unit="rows")

# for row in streaming_dataset:
#     text = row["content"]
#     tokenized = tokenizer(text)
#     input_ids = tokenized["input_ids"]
#     if len(input_ids) >= TARGET_TOKEN_LENGTH:
#         input_ids = input_ids[:TARGET_TOKEN_LENGTH]
#         output_rows.append({"input_ids": input_ids})
#         pbar.update(1)
#     if len(output_rows) >= TARGET_ROWS:
#         break

# output_ds = Dataset.from_list(output_rows)
def batch_processor(batch):
    """
    Tokenizes and filters a batch of text at once.
    This runs in parallel in Rust (fast) rather than Python loop (slow).
    """
    # Tokenize the whole batch at once with truncation
    tokenized = tokenizer(
        batch["content"], 
        truncation=True, 
        max_length=TARGET_TOKEN_LENGTH,
        padding=False  # Do not pad, we just want to check length/truncate
    )
    
    # Filter: Keep only sequences that met the target length.
    # (Since we truncated, valid rows equal TARGET_TOKEN_LENGTH. Short rows are less.)
    packed_input_ids = [
        ids for ids in tokenized["input_ids"] 
        if len(ids) == TARGET_TOKEN_LENGTH
    ]
    
    return {"input_ids": packed_input_ids}

# 2. Apply Mapping (Lazy Evaluation)
# batched=True is the key speedup (10x-100x faster than loops)
processed_ds = streaming_dataset.map(
    batch_processor,
    batched=True,
    batch_size=1000,
    remove_columns=["content", "score", "source"]  # specific columns depends on your dataset
)
limmited_ds = processed_ds.take(TARGET_ROWS)

# 4. Save/Materialize efficiently
# Instead of building a list in RAM, we use a generator.
# This streams data from the map function and writes it directly to 
# the disk-backed Arrow format, keeping RAM usage low.

def gen():
    # The loop will naturally break when limited_stream runs out (at TARGET_ROWS)
    for row in tqdm(limmited_ds, total=TARGET_ROWS, desc="Writing Valid Rows"):
        yield row

# Explicitly defining features is good practice for large datasets
features = Features({"input_ids": [Value("int32")]}) # or int64 depending on vocab size

output_ds = Dataset.from_generator(
    gen, 
    features=features
)


# In[4]:



output_ds = output_ds.cast_column("input_ids", Sequence(Value("uint16"), length=TARGET_TOKEN_LENGTH))
output_ds.push_to_hub(OUTPUT_DATASET_NAME, private=False)


