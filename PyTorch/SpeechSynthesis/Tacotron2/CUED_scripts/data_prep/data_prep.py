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
  bash_dir = '/home/dawna/tts/mw545/TorchTTS/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/CUED_scripts/data_prep'
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
  data_dir = '/home/dawna/tts/mw545/TorchTTS/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/VCTK-Corpus'
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
  data_dir = '/home/dawna/tts/mw545/TorchTTS/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/VCTK-Corpus'
  filelist_dir = os.path.join(data_dir, 'filelists/by_speaker')
  work_dir = '/home/dawna/tts/mw545/TorchTTS/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2'
  bash_dir = '/home/dawna/tts/mw545/TorchTTS/DeepLearningExamples/PyTorch/SpeechSynthesis/Tacotron2/CUED_scripts/data_prep'

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




if __name__ == '__main__':
    # write_file_list_speaker()
    # generate_mel()
    # run_all_bash_files()
    pass