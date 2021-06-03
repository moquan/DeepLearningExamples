mkdir -p output_waveglow
python -m multiproc train.py -m WaveGlow -o ./output_waveglow/ -lr 1e-4 --epochs 1501 -bs 4 --segment-length  8000 --weight-decay 0 --grad-clip-thresh 3.4028234663852886e+38 --cudnn-enabled --cudnn-benchmark \
  --log-file train_waveglow.json --load-mel-from-disk False --sampling-rate 24000 \
  --training-files VCTK-Corpus/filelists/2_stage_exp/audio_text_train_filelist.txt --validation-files VCTK-Corpus/filelists/2_stage_exp/audio_text_valid_filelist.txt
