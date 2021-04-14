import torch
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):
    """Initialize PyTorch Model Class.

    :params:
    ----------
    dictionary_size: integer, required. Total number of words in the dictionary.
    embedding_size: integer, required. Number of features in the embedding space, maybe 300, 100, or 50.
    number_of_layers: integer, default to 1. Number of LSTM layers.
    droupout_probability: float, default to 0.5. Probability for dropping nn dropout.
    batch_size: integer, default to 64. The batch size for this model.
    sequence_length: integer, default to 30. The token sequence length.
    max_norm: integer, default to 2. The maximum norm for back propagation.
    max_init_param: float, default to 0.01. The maximum weight after initialization.
    device: string, default to "cpu" or "gpu" if detected. The hardware device to train on.
    sequence_step_size: None, optional. Default to sequence length.
                        The step size for batching (the smaller it is, the more overlap).
    lstm_configuration: a string, default to "default". Enable future options to tinker with model.

    primary_reference: https://github.com/iryzhkov/nlp-pipeline
    secondary_reference: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
    """

    def __init__(self
               , dictionary_size
               , embedding_layer
               , embedding_size
               , number_of_layers=2
               , dropout_probability=0.5
               , batch_size=64
               , sequence_length=30
               , max_norm=2
               , max_init_param=0.01
               , device="cpu"
               , sequence_step_size=None
               , lstm_configuration="default"
                 ):

        super().__init__()
        self.dictionary_size = dictionary_size
        self.embedding_layer = embedding_layer
        self.embedding_size = embedding_size
        self.number_of_layers = number_of_layers
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_norm = max_norm
        self.max_init_param = max_init_param

        if sequence_step_size is None:
            self.sequence_step_size = sequence_length
        else:
            self.sequence_step_size = sequence_step_size

        if device == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.dropout = nn.Dropout(dropout_probability)

        self.lstm = LSTM(self.embedding_size
                         , number_of_layers
                         , dropout_probability
                         , lstm_configuration)

        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, inp, hidden):
        return self.gru(self.embedding(inp), hidden)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))


class LSTM(nn.Module):
    def __init__(self
                 , embedding_size
                 , number_of_layers
                 , dropout_probability
                 , lstm_configuration
                 ):
        """Initialization for LSTM model.

        :params:
        ----------
        embedding_size: integer, required. Number of features in the embedding space.
        number_of_layers: integer, required. Number of LSTM layers (for stacked-LSTM).
        dropout_probability: float, default to 0.5. Probability for dropping nn dropout.
        lstm_configuration: the configuration of the lstm. Possible configurations:
        Name            Description
        default         The regular stacked-lstm architecture
        *TBD other options

        primary_reference: https://github.com/iryzhkov/nlp-pipeline
        """
        super().__init__()
        configurations = {
            "default": 0
            }

        self.configuration = configurations[lstm_configuration]

        self.lstm = nn.LSTM(embedding_size
                          , embedding_size
                          , num_layers=number_of_layers
                          , dropout=dropout_probability
                            )
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, X, states=None):
        if self.configuration == 0:
            X = self.dropout(X)
            X, states = self.lstm(X, states)

        return X, states