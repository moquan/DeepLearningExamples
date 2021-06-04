# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import random
import numpy as np
import torch
import torch.utils.data

import tacotron2_common.layers as layers
from tacotron2_common.utils import load_wav_to_torch, load_filepaths_and_text, to_gpu
from tacotron2.text import text_to_sequence

import pickle
from tacotron2.data_function import TextMelLoader

class TextMelSpkLoader(TextMelLoader):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
        4) New: load speaker embedding
    """
    def __init__(self, dataset_path, audiopaths_and_text, args):
        super().__init__(dataset_path, audiopaths_and_text, args)

        self.speaker_embedding_dim  = 2048
        self.speaker_embedding_file = '/home/dawna/tts/mw545/TorchTTS/DeepLearningExamples/PyTorch/SpeechSynthesis/CUED_Tacotron2/VCTK-Corpus/spk_embedding/dv_spk_dict.dat'
        self.speaker_embedding_dict = pickle.load(open(self.speaker_embedding_file, 'rb'))
        assert len(self.speaker_embedding_dict['p100']) == self.speaker_embedding_dim

    def get_mel_text_spk_tuple(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        len_text = len(text)
        text = self.get_text(text)      # torch.Size([51])
        mel = self.get_mel(audiopath)   # torch.Size([80, 413])
        spk_embed = self.get_spk_embed(audiopath, len_text) # torch.Size([2048, 51])
        return (text, mel, len_text, spk_embed)

    def get_spk_embed(self, filename, len_text):
        spk_id = filename.split('|')[0].split('/')[-1].split('.')[0].split('_')[0]
        spk_embed = self.speaker_embedding_dict[spk_id]
        return torch.tensor(spk_embed)
        # spk_embed_TD = np.tile(spk_embed, (len_text,1))
        # spk_embed_DT = spk_embed_TD.T
        # return torch.tensor(spk_embed_DT)

    def __getitem__(self, index):
        return self.get_mel_text_spk_tuple(self.audiopaths_and_text[index])

class TextMelSpkCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized, spk_emb]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        # Right zero-pad speaker-embedding; D -> T*D
        spk_embed_dim = batch[0][3].size(0)
        spk_embed_padded = torch.FloatTensor(len(batch), max_input_len, spk_embed_dim)
        spk_embed_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            spk_embed = batch[ids_sorted_decreasing[i]][3]
            spk_embed_padded[i, :input_lengths[i]] = spk_embed.repeat(input_lengths[i], 1)

        # count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)
        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, len_x, spk_embed_padded

def batch_to_gpu(batch):
    text_padded, input_lengths, mel_padded, gate_padded, \
        output_lengths, len_x, spk_embed_padded = batch
    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    max_len = torch.max(input_lengths.data).item()
    mel_padded = to_gpu(mel_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    spk_embed_padded = to_gpu(spk_embed_padded).float()
    x = (text_padded, input_lengths, mel_padded, max_len, output_lengths, spk_embed_padded)
    y = (mel_padded, gate_padded)
    len_x = torch.sum(output_lengths)
    return (x, y, len_x)
