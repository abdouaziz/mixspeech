import torch
import torch.nn as nn
from layers import TransfomerMixSpeech
from loader import Mix_Loader
from features import FeaturesEncoder
from layers import PositionalEncoding



# Define the model configuration
CONFIG ={
    'num_epochs': 10,
    'batch_size': 2,
    'num_workers': 4,
    'learning_rate': 0.001,
    'max_length': 100000,
    'n_heads': 8,
    'n_layers': 12,
    'd_model': 768,
    'd_ff': 2048,
    'dropout': 0.05,
    'path_to_file': '/Users/aziiz/Documents/Works/NLP/mixspeech/audio_wav_16000/',
    'input_channel': 10,
   
    }

PATH_TO_FILE ="/Users/aziiz/Documents/Works/NLP/mixspeech/audio_wav_16000/"
MAX_LENGTH=100000


dataloader = Mix_Loader(PATH_TO_FILE , MAX_LENGTH , batch_size=2 , alpha=1.0)


class Model(nn.Module):
    def __init__(self, d_model, d_ff, n_layers, n_head, input_chanel ,  dropout=0.1):
        super().__init__()
        self.features = FeaturesEncoder(input_chanel)
        self.position_enc = PositionalEncoding(d_model, dropout=dropout)
        self.transformer = TransfomerMixSpeech(d_model, d_ff, n_layers, n_head, dropout=dropout)
       
    def forward(self, x, mask=None):
        x = self.features(x)
        #print("the shape after the ouput is here : ", x.shape)
        x = self.position_enc(x.transpose(1, 2))
        x = self.transformer(x, mask=mask)
        return x




if __name__ == "__main__":
    model = Model(d_model=768, d_ff=3072, n_layers=12, n_head=8, input_chanel=10, dropout=0.05)
    dataloader = Mix_Loader(PATH_TO_FILE , MAX_LENGTH , batch_size=2 , alpha=1.0)
    for x  in dataloader:
        x = x.view(1,-1,10000)
        y = model(x)
        print("the output of shape ",y.shape)
        print(x.shape)
        break 
 

 