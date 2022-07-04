from transformers import  Wav2Vec2Config , Wav2Vec2ForCTC

# Initializing a Wav2Vec2 facebook/wav2vec2-base-960h style configuration
configuration = Wav2Vec2Config()
configuration.num_layers = 2
configuration.hidden_size = 512
configuration.num_heads = 8
configuration.attention_dropout = 0.1
configuration.reformer_dropout = 0.1
configuration.learning_rate = 0.0001

