#!/usr/bin/env bash

docker run --gpus=all -it --rm -e CUDA_VISIBLE_DEVICES --ipc=host -v $PWD:/workspace/hifigan/ -v /root/storage/data:/workspace/hifigan/data -v /root/storage/checkpoints:/workspace/hifigan/checkpoints hifigan:latest bash
