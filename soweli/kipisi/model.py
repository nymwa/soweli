import torch
import torch.nn as nn
from soweli.ponart.embedding import TransformerEmbedding
from soweli.ponart.encoder import TransformerEncoder
from soweli.ponart.ponart import Ponart

class KipisiClassifier(nn.Module):
    def __init__(self,
            d_vocab,
            d_model = 512,
            nhead = 8,
            d_feedforward = 2048,
            num_layers = 6,
            attention_dropout = 0.2,
            dropout = 0.3):
        super().__init__()
        self.embedding = TransformerEmbedding(d_vocab, d_model, dropout)
        self.encoder = TransformerEncoder(d_model, nhead, d_feedforward, num_layers, attention_dropout, dropout)
        self.fc = nn.Linear(d_model, 4)
        self.dropout = nn.Dropout(dropout)

    def load_checkpoint(self, checkpoint_path):
        ponart = Ponart(self.embedding.token_embedding.num_embeddings)
        ponart.load_state_dict(torch.load(checkpoint_path, map_location = 'cpu'))
        self.embedding = ponart.embedding
        self.encoder = ponart.encoder

    def forward(self, x, padding_mask = None):
        if padding_mask is None:
            padding_mask = (x == 0).T
        x = self.embedding(x)
        x = self.encoder(x, padding_mask = padding_mask)
        x = self.dropout(x)
        x = self.fc(x)
        return x

