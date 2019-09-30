"""
    计算后验概率p(z|x,y)的网络；x，y为解码器最后一步的输出
"""

import torch
import torch.nn as nn

class RecognizeNet(nn.Module):

    def __init__(self, dim_x,  # post编码维度
                 dim_y,  # response编码维度
                 dim_latent,  # 潜变量维度
                 dims):  # 隐藏层维度
        assert len(dims) >= 1  # 至少两层感知机8

        dims = [dim_x+dim_y] + dims + [dim_latent*2]
        dims_input = dims[:-1]
        dims_output = dims[1:]

        self.dim_latent = dim_latent
        self.mlp = nn.Sequential()

        # 多层感知机中间层
        for idx, (input, output) in enumerate(zip(dims_input[:-1], dims_output[:-1])):

            self.mlp.add_module('RecognizeNet/Linear%d' % idx, nn.Linear(input, output))
            self.mlp.add_module('RecognizeNet/Tanh%d' % idx, nn.Tanh())

        # 多层感知机输出层
        self.mlp.add_module('RecognizeNet/Linear%d' % idx, nn.Linear(input, output))



    def forward(self, input_x, # [batch, dim_x]
                input_y):  # [batch, dim_y]

        input = torch.cat([input_x, input_y], 1)  # [batch, dim_x+dim_y]

        predict = self.mlp(input)  # [batch, dim_latent*2]

        mu, logvar = torch.split(predict, [self.dim_latent]*2, dim=1)  # [batch, dim_latent]

        return mu, logvar