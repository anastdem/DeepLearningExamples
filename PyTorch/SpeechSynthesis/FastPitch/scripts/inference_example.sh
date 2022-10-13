#!/usr/bin/env bash

export CUDNN_V8_API_ENABLED=1

: ${DATASET_DIR:="data/RUSLAN22"}
: ${OUTPUT_DIR:="./output/ruslan_val_gen_gst_hifi"}
: ${BATCH_SIZE:=64}
: ${FILELIST:="phrases/ruslan_val.txt"}
#: ${FILELIST:="phrases/ruslan_mel_pitch_text_val.tsv"}
: ${AMP:=false}
: ${TORCHSCRIPT:=false}
: ${WARMUP:=0}
: ${REPEATS:=1}
: ${CPU:=false}
: ${PHONE:=false}

# Mel-spectrogram generator (optional)
#: ${FASTPITCH="checkpoints/fastpitch/ruslan_v0/FastPitch_checkpoint_1000.pt"}
#: ${GST_ESTIMATOR=""}

: ${FASTPITCH="checkpoints/fastpitch/ruslan_gst_v0/FastPitch_checkpoint_1000.pt"}
: ${GST_ESTIMATOR="checkpoints/fastpitch/gst_est_ruslan_v0/GST_estimator_best_checkpoint.pt"}

# Vocoder; set only one
#: ${WAVEGLOW="checkpoints/waveglow/multispeaker_v0/checkpoint_WaveGlow_last.pt"}
: ${HIFIGAN="checkpoints/hifigan/multispeaker_v0/hifigan_gen_checkpoint_5960.pt"}

#[[ "$FASTPITCH" == "pretrained_models/fastpitch/nvidia_fastpitch_210824.pt" && ! -f "$FASTPITCH" ]] && { echo "Downloading $FASTPITCH from NGC..."; bash scripts/download_models.sh fastpitch; }
#[[ "$WAVEGLOW" == "pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt" && ! -f "$WAVEGLOW" ]] && { echo "Downloading $WAVEGLOW from NGC..."; bash scripts/download_models.sh waveglow; }

# Synthesis
: ${SPEAKER:=0}
: ${DENOISING:=0.01}

if [ ! -n "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="./output/audio_$(basename ${FILELIST} .tsv)"
    [ "$AMP" = true ]     && OUTPUT_DIR+="_fp16"
    [ "$AMP" = false ]    && OUTPUT_DIR+="_fp32"
    [ -n "$FASTPITCH" ]   && OUTPUT_DIR+="_fastpitch"
    [ ! -n "$FASTPITCH" ] && OUTPUT_DIR+="_gt-mel"
    [ -n "$WAVEGLOW" ]    && OUTPUT_DIR+="_waveglow"
    [ -n "$HIFIGAN" ]     && OUTPUT_DIR+="_hifigan"
    OUTPUT_DIR+="_denoise-"${DENOISING}
fi
: ${LOG_FILE:="$OUTPUT_DIR/nvlog_infer.json"}
mkdir -p "$OUTPUT_DIR"

echo -e "\nAMP=$AMP, batch_size=$BATCH_SIZE\n"

ARGS=""
ARGS+=" --cuda"
ARGS+=" --cudnn-benchmark"
ARGS+=" --dataset-path $DATASET_DIR"
ARGS+=" -i $FILELIST"
ARGS+=" -o $OUTPUT_DIR"
ARGS+=" --log-file $LOG_FILE"
ARGS+=" --batch-size $BATCH_SIZE"
ARGS+=" --denoising-strength $DENOISING"
ARGS+=" --warmup-steps $WARMUP"
ARGS+=" --repeats $REPEATS"
ARGS+=" --speaker $SPEAKER"
[ "$CPU" = false ]        && ARGS+=" --cuda"
[ "$CPU" = false ]        && ARGS+=" --cudnn-benchmark"
[ "$AMP" = true ]         && ARGS+=" --amp"
[ "$TORCHSCRIPT" = true ] && ARGS+=" --torchscript"
[ -n "$HIFIGAN" ]         && ARGS+=" --hifigan $HIFIGAN"
[ -n "$WAVEGLOW" ]        && ARGS+=" --waveglow $WAVEGLOW"
[ -n "$FASTPITCH" ]       && ARGS+=" --fastpitch $FASTPITCH"
[ -n "$GST_ESTIMATOR" ]   && ARGS+=" --gst_estimator $GST_ESTIMATOR"
[ "$PHONE" = true ]       && ARGS+=" --p-arpabet 1.0"

CUDA_VISIBLE_DEVICES=1 python inference.py $ARGS "$@"
