import torch
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):
    """Initialize PyTorch Model Class.

    :param dictionary_size: integer, required. Total number of words in the dictionary.
    :param embedding_size: integer, required. Number of features in the embedding space, maybe 300, 100, or 50.
    :param number_of_layers: integer, default to 1. Number of LSTM layers.
    :param dropout_probability: float, default to 0.5. Probability for dropping nn dropout.
    :param batch_size: integer, default to 64. The batch size for this model.
    :param sequence_length: integer, default to 30. The token sequence length.
    :param max_norm: integer, default to 2. The maximum norm for back propagation.
    :param max_init_param: float, default to 0.01. The maximum weight after initialization.
    :param device: string, default to "gpu" if detected else "cpu. The hardware device for nn training.
    :param sequence_step_size: None, optional. Default to sequence length.
                        The step size for batching (the smaller it is, the more overlap).
    :params lstm_configuration: a string, default to "default". Enable future options to tinker with model.

    primary_reference: https://github.com/iryzhkov/nlp-pipeline
    secondary_reference: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
    """

    def __init__(self
                 , dictionary_size
                 , embedding_size
                 , embedding_layer=None
                 , num_layers=2
                 , dropout_probability=.5
                 , batch_size=64
                 , hidden_size=None
                 , sequence_length=16
                 , max_norm=2
                 , max_init_param=0.01
                 , device="gpu"
                 , sequence_step_size=None
                 , lstm_configuration="default"
                 , model_type='lstm'
                 , **kwargs
                 ):

        super().__init__()
        self.dictionary_size = dictionary_size
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size if hidden_size is None else hidden_size
        self.hidden_states = None
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.max_norm = max_norm
        self.max_init_param = max_init_param
        self.lstm_configuration = lstm_configuration

        if sequence_step_size is None:
            self.sequence_step_size = sequence_length
        else:
            self.sequence_step_size = sequence_step_size

        if device == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # if only one layer, force dropout probability to 1 (none)
        if num_layers == 1 and dropout_probability > 0:
            dropout_probability = 1

        self.dropout = nn.Dropout(dropout_probability)

        # a default embedding layer
        self.embedding = nn.Embedding(dictionary_size, embedding_size)

        # initialize parameters and weights
        for param in self.parameters():
            nn.init.uniform_(param, -max_init_param, max_init_param)

        # if provided, override embedding layer with pre-trained
        if embedding_layer is not None:
            self.embedding = embedding_layer

        if model_type == 'lstm':
            self.lstm = LSTM(embedding_size=self.embedding_size
                             , hidden_size=self.hidden_size
                             , num_layers=num_layers
                             , dropout_probability=dropout_probability
                             , lstm_configuration=lstm_configuration)
        else:
            #TODO: we can do other types like transformer here
            self.transformer = 'call transformer class here'

    def init_hidden(self):
        # initialize hidden states
        self.hidden_states = (torch.zeros(self.num_layers, self.batch_size, self.embedding_size, device=self.device),
                              torch.zeros(self.num_layers, self.batch_size, self.embedding_size, device=self.device))

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x, self.hidden_states = self.lstm(x, self.hidden_states)
        output = torch.tensordot(x, self.embedding.weight, dims=([2], [1]))

        return output


class LSTM(nn.Module):
    def __init__(self
                 , embedding_size
                 , hidden_size
                 , num_layers
                 , dropout_probability
                 , lstm_configuration
                 ):
        """Initialization for LSTM model.

        :param: embedding_size: integer, required. Number of features in the embedding space.
        :param num_layers: integer, required. Number of LSTM layers (for stacked-LSTM).
        :param: dropout_probability: float, default to 0.5. Probability for dropping nn dropout.
        :param: lstm_configuration: the configuration of the lstm. Possible configurations:

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

        self.lstm = nn.LSTM(input_size=embedding_size
                            , hidden_size=hidden_size
                            , num_layers=num_layers
                            , dropout=dropout_probability
                            )

        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x, hidden_states=None):
        if self.configuration == 0:
            x = self.dropout(x)
            x, hidden_states = self.lstm(x, hidden_states)
        return x, hidden_states
