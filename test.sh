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

# random port between 12k and 20k
export MASTER_PORT=$((12000 + RANDOM % 20000))

# run script from above
python train.py --load=ckpt_dpn/ckpt_dpn_6d_v2_2/_ckpt_epoch_5_v6.ckpt --gpus=1 --data_path=/u01/data/Dataset --json_path=cfg.json --infer=test --num_workers=2
