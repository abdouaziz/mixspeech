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
        default=1.5,
         
    )

    parser.add_argument(
        "--path_to_save",
        type=str,
        default=None,
        help="The path to save the model.",
    )

    args = parser.parse_args()

    if args.path_to_files is None:
        raise ValueError("Please specify the path to load files.")


 

    return args


def print_args():
  args  =  parse_args()
  print("The path to files is ", args.path_to_files)
  print("The maximum length of the input sequence to pad is ", args.max_length)
  print("The number of epochs is ", args.epochs)
  print("The learning rate is ", args.alpha)
    




if __name__ == "__main__":
    print_args()