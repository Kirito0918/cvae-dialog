import torch.nn as nn


class PrepareState(nn.Module):
    r""" 准备解码器的初始状态，使用潜变量和编码器输入进行初始化 """
    def __init__(self, input_size,  # 用于初始化状态的向量维度
                 decoder_cell_type,  # 解码器类型
                 decoder_output_size,  # 解码器隐藏层大小
                 decoder_num_layers):  # 解码器层数
        super(PrepareState, self).__init__()
        assert decoder_cell_type in ['GRU', 'LSTM']

        self.decoder_cell_type = decoder_cell_type
        self.num_layers = decoder_num_layers
        self.linear = nn.Linear(input_size, decoder_output_size)

    def forward(self, x):  # [batch, dim]
        if self.num_layers > 1:
            states = self.linear(x).unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, batch, output_size]
        else:
            states = self.linear(x).unsqueeze(0)
        if self.decoder_cell_type == 'LSTM':
            return states, states  # (h, c)
        else:
            return states
