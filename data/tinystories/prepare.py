# Adapted from nanoGPT (https://github.com/karpathy/nanoGPT), MIT license.
# Tokenizes the TinyStories dataset into train.bin / val.bin of uint16 GPT-2
# BPE ids, matching the format expected by train.py.
#
# Usage: python prepare.py [output_dir]
#   output_dir defaults to this script's directory. On a cluster pass a
#   /scratch path and export HF_HOME there first so the raw dataset cache does
#   not land in the home quota.
#
# Dataset: https://huggingface.co/datasets/roneneldan/TinyStories
#   Ships a `train` and a `validation` split, each a single `text` column
#   (one short story per row). We tokenize with the GPT-2 BPE (50257 vocab,
#   max id 50256 < 2**16) and separate stories with the GPT-2 end-of-text
#   token, exactly as data/openwebtext/prepare.py does.

import os
import sys

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

num_proc = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # TinyStories already provides a train and a validation split.
    dataset = load_dataset("roneneldan/TinyStories", num_proc=num_proc)
    # train.py expects splits named 'train' and 'val'
    dataset['val'] = dataset.pop('validation')

    def process(example):
        ids = enc.encode_ordinary(example['text'])  # ignores special tokens
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}

    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all ids into one big memmapped file per split
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(out_dir, f'{split}.bin')
        dtype = np.uint16  # enc.max_token_value == 50256 < 2**16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        # TinyStories is small; keep shards from getting tiny on the val split
        total_batches = min(1024, len(dset))

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"{filename}: {arr_len} tokens")

    # train.bin should be ~475M tokens (~950MB), val.bin ~4.8M tokens (~9.5MB)
