mkdir -p output
python -m multiproc train.py -m Tacotron2 -o ./output/ -lr 1e-3 --epochs 1501 -bs 16 --weight-decay 1e-6 --grad-clip-thresh 1.0 --cudnn-enabled --anneal-steps 500 1000 1500 --anneal-factor 0.1 \
   --log-file train_tacotron2.json --load-mel-from-disk True --training-files filelists/ljs_mel_text_train_filelist.txt --validation-files filelists/ljs_mel_text_val_filelist.txt
