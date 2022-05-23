# mixspeech
SSL MixUp Speech Representation Toolkit



# Training 
to run the training, you need to run the following command:

```py
python SSLMixUp/model.py  --path-to-files "/Users/aziiz/Documents/Works/NLP/mixspeech/audio_wav_16000/" \
                          --max_length 100000 \ 
                          --epochs 12 \
                          --learning_rate 0.001 \
                          --epsilone 1e-8  \
                          --batch_size 12 \
                          --n_heads 8 \
                          --n_layers 12 \
                          --d_fft 768 \
                          --d_ff 3048 \
                          --dropout 0.1 \
                          --input_channel 10 \
                          --alpha 2.0 \
                          --path-to-save "/Users/aziiz/Documents/Works/NLP/mixspeech/model/" \
```

 