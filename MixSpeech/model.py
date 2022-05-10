from tkinter import Y
import torch
import torch.nn as nn
from layers import TransfomerMixSpeech
from mix import Mix_Loader
from features import FeaturesEncoder
from layers import PositionalEncoding
import numpy as np 



PATH_TO_FILE ="/Users/aziiz/Documents/Works/NLP/mixspeech/audio_wav_16000/"
MAX_LENGTH=51200


dataloader = Mix_Loader(PATH_TO_FILE , MAX_LENGTH , batch_size=2 , alpha=1.0)


class Model(nn.Module):
    def __init__(self, d_model, d_ff, n_layers, n_head, dropout=0.1):
        super().__init__()
        self.features = FeaturesEncoder()
        self.position_enc = PositionalEncoding(d_model, dropout=dropout)
        self.transformer = TransfomerMixSpeech(d_model, d_ff, n_layers, n_head, dropout=dropout)
       
    def forward(self, x, mask=None):
        x = self.features(x)
        x = self.position_enc(x)
        x = self.transformer(x, mask=mask)
        return x



if __name__ == "__main__":
    model = Model(d_model=512, d_ff=2048, n_layers=6, n_head=8, dropout=0.1)
    x = torch.randn(1, 1, 512)
    y = model(x)
    print(y.shape)
    print(y)
    