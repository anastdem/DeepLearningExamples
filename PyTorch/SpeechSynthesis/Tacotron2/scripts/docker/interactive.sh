#!/bin/bash

nvidia-docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -it --rm --ipc=host -p 0.0.0.0:6006:6006 -p 8888:8888 -v $PWD:/workspace/tacotron2/ -v /root/storage/data:/workspace/tacotron2/data -v /root/storage/checkpoints:/workspace/tacotron2/checkpoints -v /root/NISQA:/workspace/tacotron2/NISQA tacotron2 bash
