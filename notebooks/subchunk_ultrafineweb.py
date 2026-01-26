#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from datasets import load_dataset, Features, Value, Sequence, Dataset
from huggingface_hub import HfApi
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Create sub-chunked datasets from a larger chunked dataset.")
    
    parser.add_argument(
        "--target_length", 
        type=int, 
        required=True, 
        help="The new smaller sequence length (e.g., 2048 or 512)."
    )
    parser.add_argument(
        "--max_input_rows", 
        type=int, 
        default=None, 
        help="Number of rows to consume from the input dataset. If None, processes all rows."
    )
    parser.add_argument(
        "--input_dataset", 
        type=str, 
        default="MikiV/Ultra-FineWeb-chunked-8192", 
        help="The input dataset on HF Hub."
    )
    parser.add_argument(
        "--hf_username", 
        type=str, 
        default="MikiV", 
        help="Your Hugging Face username for the output."
    )
    parser.add_argument(
        "--streaming", 
        type=bool, 
        default=False, 
        help="Whether to use streaming mode."
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Define output name based on input logic
    # Example: MikiV/Ultra-FineWeb-chunked-2048
    base_name = args.input_dataset.split("/")[-1].split("-chunked")[0] 

    original_chunk_size = int(args.input_dataset.split("-chunked-")[-1])
    assert original_chunk_size % args.target_length == 0, 'Target length must evenly divide the original chunk size.'
    subchunk_ratio = original_chunk_size // args.target_length
    output_repo_id = f"{args.hf_username}/{base_name}-chunked-{subchunk_ratio}x{args.target_length}"

    print(f"--- Configuration ---")
    print(f"Input Dataset:  {args.input_dataset}")
    print(f"Target Length:  {args.target_length}")
    print(f"Row Limit:      {args.max_input_rows if args.max_input_rows else 'All'}")
    print(f"Output Dataset: {output_repo_id}")
    print(f"---------------------")

    # 1. Load the input dataset in streaming mode
    print("Loading input dataset (streaming)...")
    ds = load_dataset(args.input_dataset, split="train", streaming=args.streaming)
    
    # 2. Apply limit if requested
    if args.max_input_rows is not None:
        ds = ds.take(args.max_input_rows)
        total_input = args.max_input_rows
    else:
        # If streaming without limit, we won't know total explicitly for tqdm, 
        # but we can just let it run.
        total_input = None

    # 3. Generator function to reshape data
    # This reads 1 row of 8192 and yields N rows of target_length
    def gen():

        print(f"Detected input length: {original_chunk_size}. Target: {args.target_length}.")
        print(f"Splitting into {subchunk_ratio} passes (The dataset will be streamed {subchunk_ratio} times).")

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
                    chunk = long_input_ids[start_token : end_token]
                    yield {"input_ids": chunk}

    # 4. Define Features for efficient storage
    # We maintain the uint16 optimization you used previously
    features = Features({
        "input_ids": Sequence(Value("uint16"), length=args.target_length)
    })

    # 5. Create the new dataset from generator
    print("Generating new dataset...")
    new_ds = Dataset.from_generator(gen, features=features)

    # 6. Push to Hub
    print(f"Pushing to Hugging Face Hub: {output_repo_id}")
    new_ds.push_to_hub(output_repo_id, private=False)
    
    print("\nDone! Success.")

if __name__ == "__main__":
    main()