import torch
import torch.nn as nn
from Layers import TransfomerMixSpeech
from Loader import Mix_Loader
from Features import FeaturesEncoder
from Layers import PositionalEncoding
import torch.functional as F


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
    'alpha': 1.0,
   
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
        x = x.permute(0,2,1) 
        x = self.position_enc(x)
        x = self.transformer(x, mask=mask)
        #print("the shape after transformer is : ", x.shape)
        #Pooling 
        x = x.mean(dim=1)
       # print("the shape after pooling is : ", x.shape)
        return x



def loss_fn (output, target):
    return F.cross_entropy(output, target)


def train_step(model, dataloader, optimizer, loss_fn, epoch):
    model.train()
    for data in dataloader:
        data = data.view(1,-1,768)

        #Compute predicion and loss 
        output = model (data)
        loss =  loss_fn(output, data)

        #Backpropagate and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Print the loss
    return loss.item()


 # Define the loss function
loss_fn = loss_fn 



def main():

   
    # Define the model

    model = Model(d_model=CONFIG['d_model'], 
                d_ff=CONFIG['d_ff'], 
                n_layers=CONFIG['n_layers'], 
                n_head=CONFIG['n_heads'], 
                input_chanel=CONFIG['input_channel'], 
                dropout=CONFIG['dropout'] ,
                )


    dataloader = Mix_Loader(CONFIG['path_to_file'] , CONFIG['max_length'] , batch_size=CONFIG['batch_size'] , alpha=CONFIG['alpha'])

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

   
    # Train the model
    for epoch in range(CONFIG['num_epochs']):
        loss = train_step(model, dataloader, optimizer, loss_fn, epoch)
        print("Epoch: {} , Loss: {}".format(epoch, loss))



       







if __name__ == "__main__":
    main()












"""    model = Model(d_model=768, d_ff=3072, n_layers=12, n_head=8, input_chanel=1, dropout=0.05)
    dataloader = Mix_Loader(PATH_TO_FILE , MAX_LENGTH , batch_size=2 , alpha=1.0)
    for x  in dataloader:
        x = x.view(1,-1,768)
       # print("The input shape ",x.shape)
        outputs = model(x)
        print("The output shape ",outputs.shape)


        break  """
 

 