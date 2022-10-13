#!/usr/bin/env bash

PORT=${PORT:-8899}

docker run --gpus=all -it --rm -e CUDA_VISIBLE_DEVICES --ipc=host -p $PORT:$PORT -p 0.0.0.0:6005:6005 -v $PWD:/workspace/fastpitch/ -v /root/storage/data:/workspace/fastpitch/data -v /root/storage/checkpoints:/workspace/fastpitch/checkpoints -v /root/NISQA:/workspace/fastpitch/NISQA fastpitch:latest bash
