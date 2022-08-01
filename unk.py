
from transformers import Wav2Vec2Processor, HubertModel
from datasets import load_dataset
import soundfile as sf

processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)

input_values = processor(ds["speech"][0], return_tensors="pt").input_values

input_values





import torch
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features) :
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature } for feature in features]
     
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )  
       
        batch["mask_time_indices"] = [ self.get_mask(i)  for i in batch["input_values"] ]

        return batch


    def get_mask (self , input_values):
     
      batch_size, raw_sequence_length = 1 , input_values.shape[0]
      sequence_length = model._get_feat_extract_output_lengths(raw_sequence_length)
      mask_time_indices = _compute_mask_indices((batch_size, sequence_length), mask_prob=0.2, mask_length=2)
      mask_time_indices = torch.tensor(mask_time_indices, device=input_values.device, dtype=torch.long)

      return  mask_time_indices

dataCollactor = DataCollatorCTCWithPadding(processor=processor, padding=True)

data =dataCollactor(ds["speech"])



from torch.utils.data import DataLoader , Dataset



# Need to override __init__, __len__, __getitem__
# as per datasets requirement
class CustomDataset(Dataset):
    def __init__(self, data):
      self.data =data
      
    def __len__(self):
        return len(self.data["input_values"])

    def __getitem__(self, idx):

      input_values = self.data["input_values"][idx]
      attention_mask = self.data["attention_mask"][idx]
      mask_time_indices = self.data["mask_time_indices"][idx]

      
      return {'input_values':input_values ,
              'attention_mask': attention_mask,
              'mask_time_indices':mask_time_indices 
              }

costum_dataset = CustomDataset(data)

train_loader =DataLoader(costum_dataset , batch_size = 4)

data = next(iter(train_loader))




data["input_values"][0].view(1,-1)

output = model (data["input_values"][0].view(1,-1) ,attention_mask=data["attention_mask"][0]  , mask_time_indices=data["mask_time_indices"][0])

print("helo",output)

