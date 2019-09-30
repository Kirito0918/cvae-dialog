import torch
import torch.nn as nn
from Embedding import WordEmbedding
from Encoder import SentenseEncoder
from PriorNet import PriorNet
from RecognizeNet import RecognizeNet
from Decoder import Decoder

class Model(nn.Module):

    def __init__(self, config):

        self.config = config

        # 定义嵌入层
        self.embedding = WordEmbedding(config.num_vocab,  # 词汇表大小
                                       config.embed_size,  # 嵌入层维度
                                       config.pad_id)  # padid

        # post编码器
        self.post_encoder = SentenseEncoder(config.post_encoder_cell_type,  # rnn类型
                                            config.embed_size,  # 输入维度
                                            config.post_encoder_output_size,  # 输出维度
                                            config.post_encoder_num_layer,  # rnn层数
                                            config.post_encoder_bidirection,  # 是否双向
                                            config.dropout)  # dropout概率

        # response编码器
        self.response_encoder = SentenseEncoder(config.post_encoder_cell_type,  # rnn类型
                                                config.embed_size,  # 输入维度
                                                config.post_encoder_output_size,  # 输出维度
                                                config.post_encoder_num_layer,  # rnn层数
                                                config.post_encoder_bidirection,  # 是否双向
                                                config.dropout)  # dropout概率

        # 先验网络
        self.prior_net = PriorNet(config.post_encoder_output_size,  # post输入维度
                                  config.dim_latent,  # 潜变量维度
                                  config.prior_dims)  # 隐藏层维度

        # 识别网络
        self.recognize_net = RecognizeNet(config.post_encoder_output_size,  # post输入维度
                                          config.response_encoder_output_size,  # response输入维度
                                          config.dim_latent,  # 潜变量维度
                                          config.recognize_dims)  # 隐藏层维度

        self.

        # 解码器
        self.decoder = Decoder(config.decoder_cell_type,  # rnn类型
                               config.embed_size,  # 输入维度
                               config.decoder_output_size,  # 输出维度
                               config.decoder_num_layer,  # rnn层数
                               config.dropout)  # dropout概率)

        # 输出层
        self.projector = nn.Sequential(
            nn.Linear('projector/linear', config.decoder_output_size, config.num_vocab),
            nn.Softmax()
        )



    def forward(self, input,  # input
                inference=False):

        id_post = input['post']  # [batch, seq]
        len_post = input['len_post']  # [batch]
        id_response = input['response']
        len_response = input['len_response']
        batch_size = id_post.size()[0]

        if inference:  # 测试
            pass

        else:  # 训练

            embed_post = self.embedding(id_post)  # [batch, seq, embed_size]
            embed_response = self.embedding(id_response)  # [batch, seq, embed_size]

            # state = [layers, batch, dim]
            _, post_state = self.post_encoder(embed_post.transpose(0, 1))
            _, response_state = self.response_encoder(embed_response.transponse(0, 1))

            x = post_state[-1, :, :]  # [batch, dim]
            y = response_state[-1, :, :]  # [batch, dim]

            # p(z|x)
            _mu, _logvar = self.prior_net(x)  # [batch, latent]

            # p(z|x,y)
            mu, logvar = self.recognize_net(x, y)  # [batch, latent]

            # 采样
            nz = torch.randn((batch_size, self.config.dim_latent))  # [batch, latent]

            # 重参数化
            z = mu + torch.exp(0.5*logvar)  # [batch, latent]

