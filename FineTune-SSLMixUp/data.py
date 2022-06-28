import pandas as pd
import numpy as np
import torch


with open("/Users/aziiz/Documents/Works/NLP/mixspeech/text") as f:
    data = f.read().split("\n")

file_name=[]
sentences=[]

for i in data:
    file_name.append(str(i[:18]))
    sentences.append(i[18:])

audio_data = {
    'ID':file_name,
    'transcription':sentences
}

wolof_train = pd.DataFrame(audio_data)
wolof_train = wolof_train[:12000]

