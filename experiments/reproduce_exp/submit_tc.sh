#!/bin/bash
cd /home/shengyenc/workspace/research/ngpt
~/.conda/envs/hlm/bin/python submit_ngpt.py --cluster arc-tc --gpus-per-lane 4 --only ngpt_fp32_10k gpt_bf16_10k ngpt_bf16_10k
