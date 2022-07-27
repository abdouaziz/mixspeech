from pathlib import Path
import torch
import os
import wandb
import librosa
import pandas as pd
from zmq import device
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoFeatureExtractor, Wav2Vec2ForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices 




class CustomDataset(Dataset):
    def __init__(self ,data ):
        self.data = data   
    def __len__(self):
        return len(self.data["train"])

    def __getitem__(self,idx):
        file = self.data["train"][idx]["speech"]
        return file

def my_collate_fn(batch ,model ,feature_extractor):
    input_values = feature_extractor(batch ,padding=True ,  sampling_rate=16000 , return_tensors="pt").input_values 
    
    batch_size, raw_sequence_length = input_values.shape
    sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
    mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.2, mask_length=2)
    mask_time_indices = torch.tensor(mask_time_indices, device=input_values.device, dtype=torch.long)

    return {"input_values":input_values ,
           "mask_time_indices":mask_time_indices}


def to_dataframe(path = 'ALFFA_PUBLIC/ASR/WOLOF/data/train/'):
    """
    Load the data path and return a DataFrame with the file 
    """
    paths = set()
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                audio_path = os.path.join(root, file)
                p = Path(audio_path)
                id = p.parts[-1].split('.')[0]
                paths.add(audio_path)

    dataframe =pd.DataFrame(paths , columns=["file"])

    return dataframe


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["file"], sr = 16000)
    batch["speech"] = speech_array
    batch["sampling_rate"] = sampling_rate
    return batch


def train():

  
    MODEL_NAME="facebook/wav2vec2-base"

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForPreTraining.from_pretrained(MODEL_NAME)

    dataframe = to_dataframe()

    data_ = DatasetDict({'train': Dataset.from_pandas(dataframe)})
    data_ = data_.map(speech_file_to_array_fn, remove_columns=data_.column_names["train"], num_proc=1)

    dataset = CustomDataset(data_)

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda batch: my_collate_fn(batch, model, feature_extractor))

    for step , batch in enumerate(dataloader):
        print("Training ...")
        model.train()
        model.zero_grad()

        input_values = batch["input_values"]
        mask_time_indices = batch["mask_time_indices"]
        loss = model(input_values, mask_time_indices).loss
        
        loss.backward()

    return loss


def main():
    wandb.init(project="wav2vec2_pretraining")
    loss = train()
    wandb.log({"loss":loss})
       







if __name__ == "__main__":
    main()




