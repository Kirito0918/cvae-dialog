import torch.nn as nn

class WordEmbedding(nn.Module):

    def __init__(self, vocab_num, embed_size, embed=None, pad_id=0):

        self.embedding = nn.Embedding(vocab_num, embed_size, padding_idx=pad_id)

        if embed is not None:  # 初始化权值
            self.embedding.weight = embed

    def forward(self, input):  # [batch, len]

        embeded = self.embedding(input)
        return embeded
