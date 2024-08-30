import argparse
import os
import shutil
import sys
import time
import datetime
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets import concatenate_datasets, load_dataset


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument("--file_name", type=str, default="")
    group.add_argument(
        "--input", type=str, required=True, help="Path to local stored dataset or repository on the Hugging Face hub"
    )
    group.add_argument("--column", type=str, default="text", help="Column to preprocess from the Dataset")
    parser.add_argument("--split", type=str, default="train", help="Which split of the data to process")
    parser.add_argument('--sample_ratio', type=float, default=1.0, help='sample ratio')
    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing vocabulary files required by the tokenizer or the model id of a predefined tokenizer hosted inside a model repo on the Hugging Face Hub.",
    )
    group.add_argument("--trust_remote_code", type=bool, default=False)
    group.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Whether or not to add special tokens when encoding the sequences. This will be passed to the Tokenizer",
    )

    group = parser.add_argument_group(title="output data")
    group.add_argument("--output-prefix", type=str, required=True, help="Path to the output processed dataset file")

    parser.add_argument("--num_proc", type=int, default=50, help="Number of processes to use for data processing")
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--total_rank", type=int, default=0)

    args = parser.parse_args()

    return args

def main(args):
    start = time.time()
    rank = int(os.environ["RANK"])

    # Check if output directory exists
    if not os.path.isdir(os.path.abspath(os.path.join(args.output_prefix, os.path.pardir))):
        print(f"Creating {os.path.abspath(os.path.join(args.output_prefix, os.path.pardir))} directory...")
        os.makedirs(os.path.abspath(os.path.join(args.output_prefix, os.path.pardir)), exist_ok=True)

    if args.input.endswith(".jsonl"):  # For processing JSON files (Cross compatibility with other projects)
        ds = load_dataset("json", data_files=args.input,cache_dir='hf_cache')
        ds = concatenate_datasets(
            [ds[splits] for splits in ds.keys()]
        )  # load_dataset returns DatasetDict and we want a Dataset
    else:
        ds = load_dataset(args.input, split=args.split)

    ds = ds.shard(num_shards=args.total_rank, index=rank, contiguous=True)
    ds = ds.select_columns(args.column)
    sample_indices=None
    if args.sample_ratio<1:
        sample_indices = random.sample(range(len(ds)),k=int(len(ds)*args.sample_ratio))
        ds = ds.select(indices=sample_indices)

    tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
            trust_remote_code=args.trust_remote_code
    )
    token_dtype = np.int32 if len(tokenizer) > np.iinfo(np.uint16).max + 1 else np.uint16
    # Create tmp directory for worker outputs
    tmp_folder = os.path.abspath(os.path.join(args.output_prefix, os.pardir, f"{args.file_name}_tmp"))
    os.makedirs(tmp_folder, exist_ok=True)

    print("Creating workers output files...")
    worker_output_file = os.path.join(tmp_folder, f"worker_{rank}_input_ids.npy")
    if sample_indices is not None:
        import pickle
        with open(os.path.join(tmp_folder,'sample_indices.pkl'),'wb') as f:
            pickle.dump(sample_indices,f)
            print(os.path.join(tmp_folder,'sample_indices.pkl'))
            print('done')
    ds = ds.map(
        lambda x: {"input_ids": tokenizer(x, add_special_tokens=args.add_special_tokens).input_ids},
        input_columns=args.column,
        batched=True,
        desc=f"Rank_{rank}: Tokenizing Dataset",
        remove_columns=[args.column],
        num_proc=args.num_proc
    )

    worker_input_ids_file = open(worker_output_file, "wb")
    for sample in tqdm(ds, desc=f"Rank_{rank}: Writing workers output files"):
        np_array = np.array(sample["input_ids"], dtype=token_dtype)
        worker_input_ids_file.write(np_array.tobytes(order="C"))
    worker_input_ids_file.close()

    end = time.time()
    print("Elapsed time: ", datetime.timedelta(seconds=end-start))
    print(type(ds))

if __name__ == "__main__":
    _args = get_args()
    main(_args)
