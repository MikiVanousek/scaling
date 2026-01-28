#!/usr/bin/env python
# coding: utf-8

import argparse
import os

from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

# Log to wandb
import wandb

wandb.init(project="subchunk_ultrafineweb")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create sub-chunked datasets from a larger chunked dataset."
    )

    parser.add_argument(
        "--target_length",
        type=int,
        required=True,
        help="The new smaller sequence length (e.g., 2048 or 512).",
    )
    parser.add_argument(
        "--max_input_rows",
        type=int,
        default=None,
        help="Number of rows to consume from the input dataset. If None, processes all rows.",
    )
    parser.add_argument(
        "--input_dataset",
        type=str,
        default="MikiV/Ultra-FineWeb-chunked-8192",
        help="The input dataset on HF Hub.",
    )
    parser.add_argument(
        "--hf_username",
        type=str,
        default="MikiV",
        help="Your Hugging Face username for the output.",
    )
    parser.add_argument(
        "--streaming", type=bool, default=False, help="Whether to use streaming mode."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Define output name based on input logic
    # Example: MikiV/Ultra-FineWeb-chunked-2048
    base_name = args.input_dataset.split("/")[-1].split("-chunked")[0]

    original_chunk_size = int(args.input_dataset.split("-chunked-")[-1])
    assert original_chunk_size % args.target_length == 0, (
        "Target length must evenly divide the original chunk size."
    )
    subchunk_ratio = original_chunk_size // args.target_length
    output_repo_id = (
        f"{args.hf_username}/{base_name}-chunked-{subchunk_ratio}x{args.target_length}"
    )

    print(f"--- Configuration ---")
    print(f"Input Dataset:  {args.input_dataset}")
    print(f"Target Length:  {args.target_length}")
    print(f"Row Limit:      {args.max_input_rows if args.max_input_rows else 'All'}")
    print(f"Output Dataset: {output_repo_id}")
    print(f"---------------------")

    # 1. Load the input dataset in streaming mode
    print("Loading input dataset (all splits)...")
    ds_all = load_dataset(args.input_dataset, streaming=args.streaming)

    # 2. Apply limit per split below

    # 3. Generator factory to reshape data for a given split
    # This reads 1 row of 8192 and yields N rows of target_length
    def make_gen(ds, total_input):
        def gen():
            print(
                f"Detected input length: {original_chunk_size}. Target: {args.target_length}."
            )
            print(
                f"Splitting into {subchunk_ratio} passes (The dataset will be streamed {subchunk_ratio} times)."
            )

            # Step B: Outer Loop - The "Position" of the chunk
            # Pass 0 gets all 1st chunks, Pass 1 gets all 2nd chunks, etc.
            for pass_idx in range(subchunk_ratio):
                start_token = pass_idx * args.target_length
                end_token = start_token + args.target_length

                description = f"Pass {pass_idx + 1}/{subchunk_ratio} (tokens {start_token}-{end_token})"

                # Step C: Inner Loop - The Rows
                # We create a fresh iterator for 'ds' every time the outer loop runs.
                iterator = tqdm(ds, total=total_input, desc=description, unit="row")

                for row in iterator:
                    long_input_ids = row["input_ids"]

                    # Only yield if the row actually has data in this range
                    if len(long_input_ids) >= end_token:
                        chunk = long_input_ids[start_token:end_token]
                        yield {"input_ids": chunk}

        return gen

    # 4. Define Features for efficient storage
    # We maintain the uint16 optimization you used previously
    features = Features(
        {"input_ids": Sequence(Value("uint16"), length=args.target_length)}
    )

    # 5. Create the new dataset from generator
    print("Generating new datasets per split...")
    processed = {}
    for split_name, split_ds in ds_all.items():
        # 2. Apply limit if requested (per split)
        if args.max_input_rows is not None:
            split_ds = split_ds.take(args.max_input_rows)
            total_input = args.max_input_rows
        else:
            # If streaming without limit, we won't know total explicitly for tqdm,
            # but we can just let it run.
            total_input = None

        gen_fn = make_gen(split_ds, total_input)
        processed[split_name] = Dataset.from_generator(gen_fn, features=features)

    # 6. Push to Hub as a DatasetDict preserving splits
    ds_out = DatasetDict(processed)
    print(f"Pushing to Hugging Face Hub: {output_repo_id}")
    ds_out.push_to_hub(output_repo_id, private=False)

    print("\nDone! Success.")


if __name__ == "__main__":
    main()
