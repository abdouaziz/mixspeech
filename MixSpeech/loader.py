import torch
from torch.utils.data import DataLoader , Dataset
import os 
import librosa
import numpy as np



PATH_TO_FILE ="/Users/aziiz/Documents/Works/NLP/mixspeech/audio_wav_16000/"
MAX_LENGTH=768



def pad_audio(audio, max_len):
    """
    Pad audio to max_len
    """
    audio = torch.from_numpy(audio).float()
    if audio.shape[0] > max_len:
        audio = audio[:max_len]
    else:
        audio = torch.cat([audio, torch.zeros(max_len - audio.shape[0])])
    return audio


class AudioDatase(Dataset):
    """
    Custom Dataset
    """
    def __init__(self, path_to_file , max_len):
        self.path_to_file = path_to_file
        self.files = os.listdir(path_to_file)
        self.files = [self.path_to_file + file for file in self.files]
        self.max_len = max_len

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        audio, _ = librosa.load(file, sr=16000)
        audio = pad_audio(audio, max_len=self.max_len)
        return audio



def get_dataloader(path_to_file, max_len , batch_size=2):
    """
    DataLoader
    """
    dataset = AudioDatase(path_to_file , max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
   
    return dataloader 


def Mix_Loader(path_to_file , max_len , batch_size=2 , alpha=1.0):
    """
    Mix two speech
    """
    dataloader = get_dataloader(path_to_file ,max_len ,  batch_size)
    alpha = np.random.beta(alpha, alpha)
    for speech_1, speech_2 in dataloader:
        mixed_speech = alpha * speech_1 + (1 - alpha) * speech_2

        yield mixed_speech





if __name__=='__main__':
    dataloader = Mix_Loader(PATH_TO_FILE , MAX_LENGTH , batch_size=2 , alpha=1.0)

    for data in dataloader:
        print(data)
        break 

