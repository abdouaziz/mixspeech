import torch
import torch.nn as nn
from layers import Model
from loader import Mix_Loader
import numpy as np
from loss import loss_fn
import wandb
import argparse


def parse_args():

    parser = argparse.ArgumentParser(
        description="SSL MixSpeech training script")

    parser.add_argument(
        "--path_to_files",
        type=str,
        default='/Users/aziiz/Documents/Works/NLP/mixspeech/audio_wav_16000/',
        help="the path containing files."
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=100000,
        help="The maximum length of the input sequence to pad.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Total number of training steps to perform the model .",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate ",
    )

    parser.add_argument(
        "--epsilone",
        type=float,
        default=1e-8,

    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (per device) for the training dataloader.",
    )

    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="Number of heads in the multi-head attention layer.",
    )

    parser.add_argument(
        "--n_layers",
        type=int,
        default=12,
        help="Number of layers in the encoder .",
    )

    parser.add_argument(
        "--d_model",
        type=int,
        default=768,
        help="Ouput dimension of the model.",
    )

    parser.add_argument(
        "--d_ff",
        type=int,
        default=2048,
        help="Dimension of the inner hidden layer in the feedforward network.",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.01,
        help="Dropout rate.",
    )

    parser.add_argument(
        "--input_channel",
        type=float,
        default=10,
    )

    parser.add_argument(
        "--alpha",
        type=int,
        default=1.0
    )

    parser.add_argument(
        "--path_to_save",
        type=str,
        default=None,
        help="The path to save the model.",
    )

    args = parser.parse_args()

    if args.path_to_file is None:
        raise ValueError("Please specify the path to load files.")


    if args.epochs :
        print("The number of epochs is {args.epochs}")  

    return args


def print_args():
    args  =  parse_args()
    print("The path to files is {args.path_to_files}")


""" 
# Define the model configuration
CONFIG = parse_args() 

wandb.init(
    project="ssl-mixspeech-project",
    entity="abdouaziz",
    name="ssl-mixspeech-2",
)

wandb.config = CONFIG
 """


class SSLMixUpRepresentationLearner(nn.Module):
    def __init__(self, d_model, d_ff, n_layers, n_head, input_chanel, dropout):
        super().__init__()
        self.model = Model(d_model, d_ff, n_layers,
                           n_head, input_chanel, dropout)

    def forward(self, x):

        output, Y = self.model(x['mixed_speech'].view(10, 10, 1000))
        fx_1, _ = self.model(x['speech_1'].view(10, 10, 1000))
        fx_2, _ = self.model(x['speech_2'].view(10, 10, 1000))
        alpha = x['alpha']
        return output, Y, fx_1, fx_2, alpha


def train_step(model, dataloader, optimizer, loss_fn):
    """
    Train the model for one step
    """

    model.train()

    losses = []

    for data in dataloader:

        # Compute predicion and loss
        output, Y, fx_1, fx_2, alpha = model(data)
        loss = loss_fn(output, Y, fx_1, fx_2, alpha)
        # track the loss
        losses.append(loss.item())

        # Backpropagate and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.mean(losses)


def main():

    args = parse_args()
    print("the batch size from args is {args.epochs}")

    print("Initializing the model and Training .... ")

    # Define the model

    loss_current = torch.inf

    model = SSLMixUpRepresentationLearner(
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        n_head=args.n_heads,
        input_chanel=args.input_channel,
        dropout=args.dropout
    )

    dataloader = Mix_Loader(args.path_to_files,
                            args.batch_size,
                            batch_size=args.batch_size,
                            alpha=args.alpha
                            )

    # Define the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate)

    # Train the model
    for epoch in range(args.epochs):
        loss = train_step(model, dataloader, optimizer, loss_fn=loss_fn)
        # Print the loss

        print("Epoch: {}/{} Loss: {}".format(epoch,
              args.epochs, loss))

        wandb.log({"loss": loss})

        if loss < loss_current:
            loss_current = loss
            torch.save(model.state_dict(
            ), "/Users/aziiz/Documents/Works/NLP/mixspeech/models/model2.bin")


if __name__ == "__main__":

    main()
