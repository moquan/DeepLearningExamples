import os
import subprocess
####################################################
# Data Prep for NVIDIA Tacotron 2 + CUED Voicebank #
####################################################


def get_spk_file_list():
  '''
  Return a dict, keys are speakers, each value is a list
  '''
  file_list_scp = '/home/dawna/tts/mw545/TorchDV/file_id_lists/file_id_list_used_cfg.scp'
  spk_file_list = {}
  with open(file_list_scp,'r') as f:
    f_list_temp = f.readlines()
    indiv_mcep_list = [x.strip() for x in f_list_temp]
    for x in indiv_mcep_list:
      spk_id = x.split('_')[0]    
      try:
        spk_file_list[spk_id].append(x)
      except:
        spk_file_list[spk_id] = [x]

  return spk_file_list

#########################################################

def convert_waveform():
  '''
  Convert 48kHz waveform to 24kHz, 1 channel and 16-bit
  '''
  Sox = '/home/dawna/tts/mw545/tools/sox-14.4.1/src/sox'
  bash_dir = '/home/dawna/tts/mw545/TorchTTS/DeepLearningExamples/PyTorch/SpeechSynthesis/CUED_Tacotron2/CUED_scripts/data_prep'
  source_dir = '/data/nst/VoiceBank/database/wav/Eng'
  target_dir = '/home/dawna/tts/mw545/TorchTTS/VCTK-Corpus/wav24'

  spk_file_list = get_spk_file_list()

  spk_list = spk_file_list.keys()
  source_wav_list = []
  for spk_id in spk_list:
    with open(spk_id+'.sh','w') as f:
      # Make a directory
      f.write('mkdir -p  '+os.path.join(target_dir, spk_id)+'\n')
      for file_id in spk_file_list[spk_id]:
        file_name = file_id + '.wav'
        source_file_name = os.path.join(source_dir, spk_id, file_name)
        target_file_name = os.path.join(target_dir, spk_id, file_name)
        if spk_id == 'p320':
            file_number = file_id.split('_')[1]
            if int(file_number) > 140:
              source_file_name = source_file_name.replace('p320', 'p320b')
        f.write(Sox+' '+source_file_name+' -r 24000 -b 16 -c 1 '+target_file_name+'\n')
    args = ('qsub', '-S', '/bin/bash','-o',bash_dir+'/log','-e',bash_dir+'/log','-l', 'queue_priority=low,tests=0,mem_grab=0M,osrel=*',spk_id+'.sh')
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)


def write_file_list_speaker():
  '''
  Write filelist files in their format
  wav
  LJSpeech-1.1/wavs/LJ050-0234.wav|It has used other Treasury law enforcement agents on special experiments in building and route surveys in places to which the President frequently travels.
  mel
  LJSpeech-1.1/mels/LJ050-0234.pt|It has used other Treasury law enforcement agents on special experiments in building and route surveys in places to which the President frequently travels.
  '''
  data_dir = '/home/dawna/tts/mw545/TorchTTS/DeepLearningExamples/PyTorch/SpeechSynthesis/CUED_Tacotron2/VCTK-Corpus'
  filelist_dir = os.path.join(data_dir, 'filelists/by_speaker')
  text_dir = '/data/nst/VoiceBank/database/txt/Eng'
  wavs_dir = os.path.join('VCTK-Corpus', 'wav24')
  mels_dir = os.path.join('VCTK-Corpus', 'mel24')

  spk_file_list = get_spk_file_list()
  spk_list = spk_file_list.keys()

  for spk_id in spk_list:
    with open(os.path.join(filelist_dir,spk_id+'_wav.txt'),'w') as f_wav:
      with open(os.path.join(filelist_dir,spk_id+'_mel.txt'),'w') as f_mel:
        for file_id in spk_file_list[spk_id]:
          txt_file_name = os.path.join(text_dir, spk_id, file_id+'.txt')
          wav_file_name = os.path.join(wavs_dir, spk_id, file_id+'.wav')
          mel_file_name = os.path.join(mels_dir, spk_id, file_id+'.pt')

          # Special handle p320: use p320_140-, and p320b-141+
          if spk_id == 'p320':
            file_number = file_id.split('_')[1]
            if int(file_number) > 140:
              txt_file_name = txt_file_name.replace('p320', 'p320b')

          with open(txt_file_name, 'r') as f_txt:
            t_line = f_txt.readlines()[0].strip()
          f_wav.write(wav_file_name+'|'+t_line+'\n')
          f_mel.write(mel_file_name+'|'+t_line+'\n')


def generate_mel():
  '''
  generate mel by speaker
  '''
  data_dir = '/home/dawna/tts/mw545/TorchTTS/DeepLearningExamples/PyTorch/SpeechSynthesis/CUED_Tacotron2/VCTK-Corpus'
  filelist_dir = os.path.join(data_dir, 'filelists/by_speaker')
  work_dir = '/home/dawna/tts/mw545/TorchTTS/DeepLearningExamples/PyTorch/SpeechSynthesis/CUED_Tacotron2'
  bash_dir = '/home/dawna/tts/mw545/TorchTTS/DeepLearningExamples/PyTorch/SpeechSynthesis/CUED_Tacotron2/CUED_scripts/data_prep'

  spk_file_list = get_spk_file_list()
  spk_list = spk_file_list.keys()

  for spk_id in spk_list:
    with open(spk_id+'.sh','w') as f:
      wav_filelist = os.path.join(filelist_dir,spk_id+'_wav.txt')
      mel_filelist = os.path.join(filelist_dir,spk_id+'_mel.txt')
      f.write('export CUDA_HOME=/usr/local/cuda-10.1\n')
      f.write('export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}\n')
      f.write('source ~/.bashrc\n')
      f.write('cd '+work_dir+'\n')
      f.write('export PYTHONPATH=${PYTHONPATH}:${PWD}\n')
      f.write('export PATH=${PATH}:${CUDA_HOME}/bin:/home/dawna/tts/mw545/tools/anaconda2/lib\n')
      f.write('export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64\n')
      f.write('unset LD_PRELOAD\n')
      f.write('source activate py36torch15cuda101\n')
      f.write('mkdir -p  '+os.path.join(data_dir, 'mel24', spk_id)+'\n')
      f.write('python preprocess_audio2mel.py --wav-files %s --mel-files %s --sampling-rate 24000\n' % (wav_filelist, mel_filelist))

    # args = ('qsub', '-S', '/bin/bash','-o',bash_dir+'/log','-e',bash_dir+'/log','-l', 'queue_priority=cuda-low,tests=0,mem_grab=0M,osrel=*',spk_id+'.sh')
    # popen = subprocess.Popen(args, stdout=subprocess.PIPE)


def run_all_bash_files():
  '''
  Run all spk_id.sh
  '''
  spk_file_list = get_spk_file_list()
  spk_list = spk_file_list.keys()

  with open('submit.sh','w') as f:
    for spk_id in spk_list:
      f.write('bash %s.sh\n' % spk_id)

  # need to run on a newer gpu machine for pytorch + tacotron
  # nohup ./submit.sh &


def write_train_valid_test_filelists():
  '''
  write 3 filelist files
  example filelist file name:
    filelists/ljs_audio_text_train_filelist.txt
  example line:
    LJSpeech-1.1/wavs/LJ050-0234.wav|It has used other Treasury law enforcement agents on special experiments in building and route surveys in places to which the President frequently travels.
  '''
  data_dir = '/home/dawna/tts/mw545/TorchTTS/DeepLearningExamples/PyTorch/SpeechSynthesis/CUED_Tacotron2/VCTK-Corpus'
  filelist_exp_dir = os.path.join(data_dir, 'filelists/2_stage_exp')
  filelist_speaker_dir = os.path.join(data_dir, 'filelists/by_speaker')

  speaker_id_list_dict = {}
  speaker_id_list_dict['all'] = ['p100', 'p101', 'p102', 'p103', 'p105', 'p106', 'p107', 'p109', 'p10', 'p110', 'p112', 'p113', 'p114', 'p116', 'p117', 'p118', 'p11', 'p120', 'p122', 'p123', 'p124', 'p125', 'p126', 'p128', 'p129', 'p130', 'p131', 'p132', 'p134', 'p135', 'p136', 'p139', 'p13', 'p140', 'p141', 'p142', 'p146', 'p147', 'p14', 'p151', 'p152', 'p153', 'p155', 'p156', 'p157', 'p158', 'p15', 'p160', 'p161', 'p162', 'p163', 'p164', 'p165', 'p166', 'p167', 'p168', 'p170', 'p171', 'p173', 'p174', 'p175', 'p176', 'p177', 'p178', 'p179', 'p17', 'p180', 'p182', 'p184', 'p187', 'p188', 'p192', 'p194', 'p197', 'p19', 'p1', 'p200', 'p201', 'p207', 'p208', 'p209', 'p210', 'p211', 'p212', 'p215', 'p216', 'p217', 'p218', 'p219', 'p21', 'p220', 'p221', 'p223', 'p224', 'p22', 'p23', 'p24', 'p26', 'p27', 'p28', 'p290', 'p293', 'p294', 'p295', 'p298', 'p299', 'p2', 'p300', 'p302', 'p303', 'p304', 'p306', 'p308', 'p30', 'p310', 'p311', 'p312', 'p313', 'p314', 'p316', 'p31', 'p320', 'p321', 'p322', 'p327', 'p32', 'p331', 'p333', 'p334', 'p336', 'p337', 'p339', 'p33', 'p340', 'p341', 'p343', 'p344', 'p347', 'p348', 'p349', 'p34', 'p350', 'p351', 'p353', 'p354', 'p356', 'p35', 'p36', 'p370', 'p375', 'p376', 'p37', 'p384', 'p386', 'p38', 'p398', 'p39', 'p3', 'p43', 'p44', 'p45', 'p47', 'p48', 'p49', 'p4', 'p52', 'p53', 'p54', 'p55', 'p56', 'p57', 'p5', 'p60', 'p61', 'p62', 'p63', 'p65', 'p67', 'p68', 'p69', 'p6', 'p70', 'p71', 'p73', 'p74', 'p75', 'p76', 'p77', 'p79', 'p7', 'p81', 'p84', 'p85', 'p87', 'p88', 'p89', 'p8', 'p90', 'p91', 'p93', 'p94', 'p95', 'p96', 'p97', 'p98', 'p99']
  # p41 has been removed, voice of a sick person
  # p202 is not in file_id_list yet, and he has same voice as p209, be careful
  speaker_id_list_dict['valid'] = ['p162', 'p2', 'p303', 'p48', 'p109', 'p153', 'p38', 'p166', 'p218', 'p70']    # Last 3 are males
  speaker_id_list_dict['test']  = ['p293', 'p210', 'p26', 'p24', 'p313', 'p223', 'p141', 'p386', 'p178', 'p290'] # Last 3 are males
  speaker_id_list_dict['not_train'] = speaker_id_list_dict['valid']+speaker_id_list_dict['test']
  speaker_id_list_dict['train'] = [spk for spk in speaker_id_list_dict['all'] if (spk not in speaker_id_list_dict['not_train'])]

  filelist_file = os.path.join(filelist_exp_dir,'audio_text_train_filelist.txt')
  with open(filelist_file,'w') as f:
    for speaker_id in speaker_id_list_dict['train']:
      speaker_file_list_file = os.path.join(filelist_speaker_dir,'%s_wav.txt' % speaker_id)
      with open(speaker_file_list_file,'r') as f_speaker:
        speaker_file_lines = f_speaker.readlines()
      for line in speaker_file_lines:
        # e.g. VCTK-Corpus/wav24/p100/p100_041.wav|What exactly were the problems?
        speaker_number = line.split('|')[0].split('/')[-1].split('.')[0].split('_')[1]
        if int(speaker_number) > 80:
          f.write(line)

  filelist_file = os.path.join(filelist_exp_dir,'audio_text_valid_filelist.txt')
  with open(filelist_file,'w') as f:
    for speaker_id in speaker_id_list_dict['valid']:
      speaker_file_list_file = os.path.join(filelist_speaker_dir,'%s_wav.txt' % speaker_id)
      with open(speaker_file_list_file,'r') as f_speaker:
        speaker_file_lines = f_speaker.readlines()
      for line in speaker_file_lines:
        # e.g. VCTK-Corpus/wav24/p100/p100_041.wav|What exactly were the problems?
        speaker_number = line.split('|')[0].split('/')[-1].split('.')[0].split('_')[1]
        if int(speaker_number) < 41:
          f.write(line)

  filelist_file = os.path.join(filelist_exp_dir,'audio_text_test_filelist.txt')
  with open(filelist_file,'w') as f:
    for speaker_id in speaker_id_list_dict['test']:
      speaker_file_list_file = os.path.join(filelist_speaker_dir,'%s_wav.txt' % speaker_id)
      with open(speaker_file_list_file,'r') as f_speaker:
        speaker_file_lines = f_speaker.readlines()
      for line in speaker_file_lines:
        # e.g. VCTK-Corpus/wav24/p100/p100_041.wav|What exactly were the problems?
        speaker_number = line.split('|')[0].split('/')[-1].split('.')[0].split('_')[1]
        if int(speaker_number) < 41:
          f.write(line)

  filelist_file = os.path.join(filelist_exp_dir,'mel_text_train_filelist.txt')
  with open(filelist_file,'w') as f:
    for speaker_id in speaker_id_list_dict['train']:
      speaker_file_list_file = os.path.join(filelist_speaker_dir,'%s_mel.txt' % speaker_id)
      with open(speaker_file_list_file,'r') as f_speaker:
        speaker_file_lines = f_speaker.readlines()
      for line in speaker_file_lines:
        # e.g. VCTK-Corpus/wav24/p100/p100_041.wav|What exactly were the problems?
        speaker_number = line.split('|')[0].split('/')[-1].split('.')[0].split('_')[1]
        if int(speaker_number) > 80:
          f.write(line)

  filelist_file = os.path.join(filelist_exp_dir,'mel_text_valid_filelist.txt')
  with open(filelist_file,'w') as f:
    for speaker_id in speaker_id_list_dict['valid']:
      speaker_file_list_file = os.path.join(filelist_speaker_dir,'%s_mel.txt' % speaker_id)
      with open(speaker_file_list_file,'r') as f_speaker:
        speaker_file_lines = f_speaker.readlines()
      for line in speaker_file_lines:
        # e.g. VCTK-Corpus/wav24/p100/p100_041.wav|What exactly were the problems?
        speaker_number = line.split('|')[0].split('/')[-1].split('.')[0].split('_')[1]
        if int(speaker_number) < 41:
          f.write(line)

  filelist_file = os.path.join(filelist_exp_dir,'mel_text_test_filelist.txt')
  with open(filelist_file,'w') as f:
    for speaker_id in speaker_id_list_dict['test']:
      speaker_file_list_file = os.path.join(filelist_speaker_dir,'%s_mel.txt' % speaker_id)
      with open(speaker_file_list_file,'r') as f_speaker:
        speaker_file_lines = f_speaker.readlines()
      for line in speaker_file_lines:
        # e.g. VCTK-Corpus/wav24/p100/p100_041.wav|What exactly were the problems?
        speaker_number = line.split('|')[0].split('/')[-1].split('.')[0].split('_')[1]
        if int(speaker_number) < 41:
          f.write(line)





if __name__ == '__main__':
    # convert_waveform()
    # write_file_list_speaker()
    generate_mel()
    # run_all_bash_files()
    # write_train_valid_test_filelists()
    pass