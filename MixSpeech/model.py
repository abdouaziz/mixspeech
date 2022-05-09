import torch
import torch.nn as nn 
from layers import TransfomerMixSpeech
from mix import Mix_Speech
import numpy as np 






class Model(nn.Module):
    def __init__(self, d_model, d_ff, n_layers, n_head, dropout=0.1):
        super().__init__()
        self.transformer = TransfomerMixSpeech(d_model, d_ff, n_layers, n_head, dropout=dropout)
       
    def forward(self, x, mask=None):
        x = self.transformer(x, mask)
        return x
        