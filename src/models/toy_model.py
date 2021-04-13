import torch
import torch.nn as nn
from torch.autograd import Variable



class ToyNN(nn.Module):
    """
    A Toy Model to go along with the toy Gutenberg dataset to test word embeddings.
    source: https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
    """
    def __init__(self, hidden_size, num_layers, embeddings):
        super(self).__init__()
        self.embedding = embeddings['emb_layer']
        num_embeddings = embeddings['num_embeddings']
        embedding_dim = embeddings['embedding_dim']
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)

    def forward(self, inp, hidden):
        return self.gru(self.embedding(inp), hidden)

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
