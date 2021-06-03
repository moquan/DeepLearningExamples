mkdir -p output_tacotron2_speaker_2_stage
python -m multiproc train.py -m Tacotron2_Speaker_2_Stage -o ./output_tacotron2_speaker_2_stage/ -lr 1e-3 --epochs 1501 -bs 16 --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --anneal-steps 500 1000 1500 --anneal-factor 0.1 \
  --log-file train_tacotron2.json  --load-mel-from-disk True \
  --training-files VCTK-Corpus/filelists/2_stage_exp/mel_text_train_filelist.txt --validation-files VCTK-Corpus/filelists/2_stage_exp/mel_text_valid_filelist.txt
