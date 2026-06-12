# Adapted from nanoGPT (https://github.com/karpathy/nanoGPT), MIT license.
# Tokenizes OpenWebText into train.bin / val.bin of uint16 GPT-2 BPE ids.
#
# Usage: python prepare.py [output_dir]
#   output_dir defaults to this script's directory. On the Unicorn cluster
#   pass a /scratch path and export HF_HOME to /scratch first - the raw
#   dataset cache is ~80GB and will not fit in the home quota.
#
# Note: openwebtext is a script-based HF dataset. If load_dataset fails on
# datasets>=3.0 ("trust_remote_code" removed), pin datasets==2.20.0 in the
# hfm env: pip install "datasets==2.20.0"

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
    try:
        dataset = load_dataset("openwebtext", num_proc=num_proc, trust_remote_code=True)
    except TypeError:  # older datasets without the trust_remote_code kwarg
        dataset = load_dataset("openwebtext", num_proc=num_proc)

    # owt has only a train split; carve out a small val split
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    def process(example):
        ids = enc.encode_ordinary(example['text'])  # ignores special tokens
        ids.append(enc.eot_token)
        return {'ids': ids, 'len': len(ids)}

    tokenized = split_dataset.map(
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
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"{filename}: {arr_len} tokens")

    # train.bin should be ~17GB (~9B tokens), val.bin ~8.5MB (~4M tokens)
