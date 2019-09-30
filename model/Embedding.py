import torch.nn as nn

class WordEmbedding(nn.Module):

    def __init__(self, vocab_num,  # 词汇表大小
                 embed_size,  # 词嵌入维度
                 embed=None,  # 预训练词向量
                 pad_id=0):  # pad的id

        self.embedding = nn.Embedding(vocab_num, embed_size, padding_idx=pad_id)

        if embed is not None:  # 初始化权值
            self.embedding.weight = embed

    def forward(self, input):  # [batch, len]

        embeded = self.embedding(input)
        return embeded
