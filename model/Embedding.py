import torch.nn as nn

class WordEmbedding(nn.Module):

    def __init__(self, num_vocab,  # 词汇表大小
                 embed_size,  # 词嵌入维度
                 pad_id=0):  # pad的id

        self.embedding = nn.Embedding(num_vocab, embed_size, padding_idx=pad_id)

    def forward(self, input):  # [batch, len]

        embeded = self.embedding(input)
        return embeded
