#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm
from datasets import load_dataset, Value, Dataset, Sequence
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
for row in streaming_dataset:
    text = row["content"]
    tokenized = tokenizer(text)
    input_ids = tokenized["input_ids"]
    if len(input_ids) >= TARGET_TOKEN_LENGTH:
        input_ids = input_ids[:TARGET_TOKEN_LENGTH]
        output_rows.append({"input_ids": input_ids})
        pbar.update(1)
    if len(output_rows) >= TARGET_ROWS:
        break


# In[4]:


output_ds = Dataset.from_list(output_rows)

output_ds = output_ds.cast_column("input_ids", Sequence(Value("uint16"), length=TARGET_TOKEN_LENGTH))
output_ds.push_to_hub(OUTPUT_DATASET_NAME, private=False)


# In[ ]:




