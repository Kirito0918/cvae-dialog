import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_vocab,
                 embedding_size,
                 pad_id=0,
                 dropout=0.1):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_vocab, embedding_size, padding_idx=pad_id)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):  # [batch, seq]
        return self.dropout(self.embedding(x))  # [batch, seq, embedding_size]
