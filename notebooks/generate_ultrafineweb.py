#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys
from math import ceil
from typing import Optional

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Sequence,
    Value,
    load_dataset,
)
from tqdm import tqdm
from transformers import GPT2TokenizerFast

try:
    from huggingface_hub import login as hf_login
except Exception:
    hf_login = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate chunked Ultra-FineWeb dataset with configurable constants and a 5% validation split."
    )
    parser.add_argument(
        "--target_token_length",
        type=int,
        default=8192,
        help="Target tokens per example after truncation.",
    )
    parser.add_argument(
        "--target_rows",
        type=int,
        default=1_000_000,
        help="Number of training rows to produce.",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.05,
        help="Fraction of training rows to also generate for validation. E.g. 0.05 for 5%% of train rows.",
    )
    parser.add_argument(
        "--input_dataset_id",
        type=str,
        default="openbmb/Ultra-FineWeb",
        help="Input dataset ID on Hugging Face Hub.",
    )
    parser.add_argument(
        "--input_split",
        type=str,
        default="en",
        help="Split of the input dataset to stream from (e.g., 'train' or 'en').",
    )
    parser.add_argument(
        "--hf_username",
        type=str,
        default="MikiV",
        help="Your Hugging Face username/organization for the output dataset repo path.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for tokenization mapping.",
    )
    parser.add_argument(
        "--dataset_name_override",
        type=str,
        default=None,
        help="Override output dataset repo name. If not provided, uses '<hf_username>/<input_name>-chunked-<token_length>'.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Optional Hugging Face write token. If provided, performs login() before pushing.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Push the resulting dataset as private.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="If provided, initialize Weights & Biases logging with this project name.",
    )
    return parser.parse_args()


def maybe_init_wandb(project_name: Optional[str]):
    if not project_name:
        return None
    try:
        import wandb

        return wandb.init(project=project_name)
    except Exception as e:
        print(f"W&B init skipped due to error: {e}")
        return None


def main():
    args = parse_args()

    # Optional W&B
    _wandb_run = maybe_init_wandb(args.wandb_project)

    # 1. Load the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    print(f"Number of distinct tokens in tokenizer: {tokenizer.vocab_size}")
    assert tokenizer.vocab_size < 2**16, "Tokenizer vocab size exceeds uint16 capacity."
    tokenizer.model_max_length = int(
        1e9
    )  # do not artificially cap during length inspection

    target_token_length = int(args.target_token_length)
    target_rows = int(args.target_rows)
    val_rows = max(1, int(target_rows * float(args.val_fraction)))

    # Prepare output repo name
    input_basename = args.input_dataset_id.split("/")[-1]
    if args.dataset_name_override:
        output_dataset_name = args.dataset_name_override
    else:
        output_dataset_name = (
            f"{args.hf_username}/{input_basename}-chunked-{target_token_length}"
        )

    print(
        f"Plan:\n"
        f"- Train rows: {target_rows}\n"
        f"- Validation rows (val_fraction={args.val_fraction}): {val_rows}\n"
        f"- Output HF repo: {output_dataset_name}\n"
    )

    # Approximate sizes, uint16 = 2 bytes
    train_size_gb = target_rows * target_token_length * 2 / 1e9
    val_size_gb = val_rows * target_token_length * 2 / 1e9
    total_size_gb = (target_rows + val_rows) * target_token_length * 2 / 1e9
    print(
        f"Approx sizes:\n"
        f"- Train ~ {train_size_gb:.2f} GB\n"
        f"- Val   ~ {val_size_gb:.2f} GB\n"
        f"- Total ~ {total_size_gb:.2f} GB\n"
    )

    # Optional HF login
    if args.hf_token and hf_login is not None:
        try:
            hf_login(token=args.hf_token)
        except Exception as e:
            print(f"Hugging Face login failed/skipped: {e}")

    # 2. Load the dataset in streaming mode
    streaming_dataset = load_dataset(
        args.input_dataset_id, split=args.input_split, streaming=True
    )

    # 3. Batch processor: tokenize and keep only sequences that meet the length
    def batch_processor(batch):
        """
        Tokenizes and filters a batch of text at once.
        """
        tokenized = tokenizer(
            batch["content"],
            truncation=True,
            max_length=target_token_length,
            padding=False,
        )

        packed_input_ids = [
            ids for ids in tokenized["input_ids"] if len(ids) == target_token_length
        ]
        return {"input_ids": packed_input_ids}

    processed_ds = streaming_dataset.map(
        batch_processor,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=["content", "score", "source"],
    )

    # Shared iterator over the processed stream to avoid materializing or double iteration
    processed_iter = iter(processed_ds)

    # Explicit features; start as variable-length list[int32], cast to fixed uint16 later
    features = Features({"input_ids": [Value("int32")]})

    def gen_n(n: int, desc: str):
        count = 0
        with tqdm(total=n, desc=desc, unit="rows") as pbar:
            for row in processed_iter:
                yield row
                count += 1
                pbar.update(1)
                if count >= n:
                    break

    # 4. Materialize train and validation sequentially from the single iterator
    train_ds = Dataset.from_generator(
        lambda: gen_n(target_rows, "Writing Train Rows"), features=features
    )
    val_ds = Dataset.from_generator(
        lambda: gen_n(val_rows, "Writing Validation Rows"), features=features
    )

    # 5. Cast column to fixed-length uint16
    train_ds = train_ds.cast_column(
        "input_ids", Sequence(Value("uint16"), length=target_token_length)
    )
    val_ds = val_ds.cast_column(
        "input_ids", Sequence(Value("uint16"), length=target_token_length)
    )

    # 6. Push both splits as a DatasetDict to the Hub
    ds_dict = DatasetDict({"train": train_ds, "validation": val_ds})
    print(
        f"Pushing dataset with splits (train={len(train_ds)}, validation={len(val_ds)}) to {output_dataset_name} ..."
    )
    ds_dict.push_to_hub(output_dataset_name, private=bool(args.private))
    print("Push complete.")


if __name__ == "__main__":
    main()
