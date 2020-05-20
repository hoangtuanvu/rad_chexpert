#!/bin/bash -l
# -------------------------
# OPTIONAL
# -------------------------
# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# PyTorch comes with prebuilt NCCL support... but if you have issues with it
# you might need to load the latest version from your  modules
# module load NCCL/2.4.7-1-cuda.10.0

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo
# -------------------------

# run script from above
python train.py --save_path=ckpt_dummy --save_top_k=8 --gpus=0 --data_path=/u01/data/Dataset --log-every=50 --test-every=1000 --json_path=cfg.json --epochs=8
