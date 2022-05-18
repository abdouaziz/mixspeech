import torch
import torch.nn as nn
from layers import Model
from loader import Mix_Loader
import numpy as np
from loss import loss_fn
import wandb


 

 





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


wandb.init(
    project="ssl-mixspeech-project",
    entity="abdouaziz",
    name="ssl-mixspeech-2",
 )

wandb.config = CONFIG




class SSLLearner(nn.Module):
    def __init__(self, d_model, d_ff, n_layers, n_head, input_chanel , dropout):
        super().__init__()
        self.model = Model(d_model, d_ff, n_layers, n_head, input_chanel , dropout)

    def forward(self, x):

        output , Y = self.model(x['mixed_speech'].view(10 , 10 , 1000)) 
        fx_1 , _ = self.model(x['speech_1'].view(10 , 10 , 1000))
        fx_2 , _ = self.model(x['speech_2'].view(10 , 10 , 1000))
        alpha = x['alpha']
        return output , Y , fx_1 , fx_2 , alpha

        
 


def train_step(model, dataloader, optimizer, loss_fn ):
    """
    Train the model for one step
    """

    model.train()

    losses = []

    for data in dataloader:
        
        #Compute predicion and loss 
        output , Y , fx_1 , fx_2 , alpha  = model(data)

        loss = loss_fn(output , Y , fx_1 , fx_2 , alpha)
 
        losses.append(loss.item())

        #Backpropagate and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      
     

    return np.mean(losses) 



def main():

    #logging.INFO("Initializing the model and Training .... ")

    print("Initializing the model and Training .... ")

    # Define the model

    loss_current = torch.inf

    model = SSLLearner(
                  d_model=CONFIG['d_model'], 
                  d_ff=CONFIG['d_ff'], 
                  n_layers=CONFIG['n_layers'], 
                  n_head=CONFIG['n_heads'], 
                  input_chanel=CONFIG['input_channel'], 
                  dropout=CONFIG['dropout'] ,
                  )


    dataloader = Mix_Loader(CONFIG['path_to_file'] ,
                            CONFIG['max_length'] , 
                            batch_size=CONFIG['batch_size'] , 
                            alpha=CONFIG['alpha']
                        )

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

   
    # Train the model
    for epoch in range(CONFIG['num_epochs']):
        loss = train_step(model, dataloader, optimizer , loss_fn=loss_fn)
                #Print the loss

        print("Epoch: {}/{} Loss: {}".format(epoch, CONFIG['num_epochs'], loss))

        wandb.log({"loss": loss})

      

        
        if loss < loss_current:
            loss_current = loss
            torch.save(model.state_dict(), "/Users/aziiz/Documents/Works/NLP/mixspeech/models/model2.bin")



if __name__ == "__main__":
    
    main()



 
    """ 
    model = SSLLearner(d_model=768, d_ff=3072, n_layers=12, n_head=8, input_chanel=10, dropout=0.05)

    dataloader = Mix_Loader(CONFIG['path_to_file'] , CONFIG['max_length'] , batch_size=2 , alpha=1.0)

    train_step(model, dataloader) 
    
    """

    #for data  in dataloader:


    """    
        x = data['mixed_speech'].view(10,-1,1000)

        X_1 = data['speech_1']

        fx_1 , _ = model (X_1.view(10,-1,1000))

        print(f"the shape of the output of FX_1 : {fx_1.shape}")

        print("The input shape ",x.shape)
        output, Y = model(x) """
       # print(f"the ouput shape is {output.shape}")
        # Y = Y.reshape(-1,1 , 768)
        # print("The Y output shape ",Y.shape)
        # print("The  final output shape ",output.shape)
       # print()
    
      #  print("the reshape of Y", Y.shape)
       

         
    
