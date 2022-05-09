import torch
from torch.utils.data import DataLoader , Dataset
import os 
import librosa
import numpy as np

PATH_TO_FILE ="/Users/aziiz/Documents/Works/NLP/mixspeech/audio_wav_16000/"
MAX_LENGTH=16000


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
    def __init__(self, path_to_file , MAX_LENGTH):
        self.path_to_file = path_to_file
        self.files = os.listdir(path_to_file)
        self.files = [self.path_to_file + file for file in self.files]
        self.max_len = MAX_LENGTH

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        audio, _ = librosa.load(file, sr=16000)
        audio = pad_audio(audio, max_len=self.max_len)
        return audio



def get_dataloader(PATH_TO_FILE, batch_size=2):
    """
    DataLoader
    """
    dataset = AudioDatase(PATH_TO_FILE , MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
   
    return dataloader 


def Mix_Speech():
    """
    Mix two speech
    """
    dataloader = get_dataloader(PATH_TO_FILE, batch_size=2)
    alpha = np.random.beta(1, 1)
    for speech_1, speech_2 in dataloader:
        mixed_speech = alpha * speech_1 + (1 - alpha) * speech_2
        yield mixed_speech
   


# if __name__ == "__main__":
#     dataloader = get_dataloader(PATH_TO_FILE, batch_size=2)
#     data = next(iter(dataloader))
#     for i in mix_speech(dataloader):
        
#         print(i)
    

  