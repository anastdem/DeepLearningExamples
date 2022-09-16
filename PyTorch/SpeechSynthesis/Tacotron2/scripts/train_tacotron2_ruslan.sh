CUDA_VISIBLE_DEVICES=0,1 python -m multiproc train.py -m Tacotron2 -o ./checkpoints/output_rusln_v0/ -lr 1e-3 --epochs 1501 -bs 28 --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --anneal-steps 500 1000 1500 --anneal-factor 0.1 --load-mel-from-disk --training-files filelists/ruslan_mel_text_train.txt --validation-files filelists/ruslan_mel_text_val.txt
