import torch
import torch.nn as nn
from Layers import TransfomerMixSpeech
from Loader import Mix_Loader
from Features import FeaturesEncoder
from Layers import PositionalEncoding



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
MAX_LENGTH=768


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
        #Pooling 
        x = x.mean(dim=1)
        return x



#Contrastive learning loss function
def loss_function (outputs, labels):
    """
    Args:

        outputs: [batch_size, num_classes]
        labels: [batch_size]
    Returns:
    
            loss: [1]   
    """
    





if __name__ == "__main__":
    model = Model(d_model=768, d_ff=3072, n_layers=12, n_head=8, input_chanel=1, dropout=0.05)
    dataloader = Mix_Loader(PATH_TO_FILE , MAX_LENGTH , batch_size=2 , alpha=1.0)
    for x  in dataloader:
        x = x.view(1,-1,768)
        print("The input shape ",x.shape)
        outputs = model(x)
        print("The output shape ",outputs.shape)

        print("the output type is here : ", type(outputs))
        print("the input type is here : ", type(x))

        loss = loss_function(x, outputs)
        print("The loss is here : ", loss)

       
      
       

      
        break 
 

 