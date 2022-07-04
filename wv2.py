from transformers import  Wav2Vec2Config , Wav2Vec2ForCTC

# Initializing a Wav2Vec2 facebook/wav2vec2-base-960h style configuration
configuration = Wav2Vec2Config()
configuration.num_layers = 2
configuration.hidden_size = 512
