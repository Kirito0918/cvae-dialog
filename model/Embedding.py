import torch.nn as nn

# 嵌入层
class WordEmbedding(nn.Module):

    def __init__(self, num_vocab,  # 词汇表大小
                 embedding_size,  # 词嵌入维度
                 pad_id=0):  # pad的id
        super(WordEmbedding, self).__init__()

        self.embedding = nn.Embedding(num_vocab, embedding_size, padding_idx=pad_id)


    def forward(self, input):  # [batch, seq]

        embeded = self.embedding(input)

        return embeded  # [batch, seq, embed_size]
