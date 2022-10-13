#!/usr/bin/env bash

set -e

: ${DATA_DIR:=data/Natasha}
#: ${ARGS="--extract-mels"}

python prepare_dataset.py \
    --wav-text-filelists filelists/natasha_audio_text_val_emp.txt \
    --n-workers 4 \
    --batch-size 1 \
    --extract-pitch \
    --f0-method pyin \
    --dataset-path $DATA_DIR \
#    $ARGS
