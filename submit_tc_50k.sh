#!/bin/bash
cd /home/shengyenc/workspace/research/ngpt
~/.conda/envs/hlm/bin/python submit_ngpt.py --cluster arc-tc --gpus-per-lane 4 --only gpt_fp32_50k
