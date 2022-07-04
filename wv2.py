from transformers import  Wav2Vec2Config , Wav2Vec2ForCTC

# Initializing a Wav2Vec2 facebook/wav2vec2-base-960h style configuration
configuration = Wav2Vec2Config()
configuration.num_layers = 2
configuration.hidden_size = 512
configuration.num_heads = 8
configuration.attention_dropout = 0.1
configuration.reformer_dropout = 0.1
configuration.learning_rate = 0.0001
configuration.batch_size = 32
configuration.num_epochs = 10
configuration.max_seq_length = 128
configuration.max_num_samples = 100000
configuration.max_num_workers = 4
configuration.max_num_samples_per_worker = 100000
configuration.max_num_samples_per_epoch = 100000
configuration.max_num_epochs = 10



