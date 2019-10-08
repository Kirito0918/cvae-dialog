"""
    计算先验概率p(z|x)的网络，x为解码器最后一步的输出
"""

import torch
import torch.nn as nn


class PriorNet(nn.Module):

    def __init__(self, x_size,  # post编码维度
                 latent_size,  # 潜变量维度
                 dims):  # 隐藏层维度
        super(PriorNet, self).__init__()

        assert len(dims) >= 1  # 至少两层感知机8

        dims = [x_size] + dims + [latent_size*2]
        dims_input = dims[:-1]
        dims_output = dims[1:]

        self.latent_size = latent_size
        self.mlp = nn.Sequential()

        # 多层感知机
        for idx, (input, output) in enumerate(zip(dims_input[:-1], dims_output[:-1])):

            self.mlp.add_module('linear%d' % idx, nn.Linear(input, output))  # 线性层
            self.mlp.add_module('activate%d' % idx, nn.Tanh())  # 激活层

        # 多层感知机输出层
        self.mlp.add_module('output', nn.Linear(dims_input[-1], dims_output[-1]))



    def forward(self, input):  # [batch, x_size]

        predict = self.mlp(input)  # [batch, latent_size*2]

        mu, logvar = torch.split(predict, [self.latent_size]*2, dim=1)  # [batch, latent_size]

        return mu, logvar
