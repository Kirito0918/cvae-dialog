import torch
import torch.nn as nn

# 准备解码器的初始状态，使用潜变量和编码器输入进行初始化
class PrepareState(nn.Module):

    def __init__(self, dim_input,  # 用于初始化状态的向量维度
                 decoder_cell_type,  # 解码器类型
                 decoder_output_size,  # 解码器隐藏层大小
                 decoder_num_layer):  # 解码器层数
        super(PrepareState, self).__init__()

        assert decoder_cell_type in ['GRU', 'LSTM']

        self.decoder_cell_type = decoder_cell_type
        self.num_layer = decoder_num_layer
        self.dim_out = decoder_output_size
        self.linear = nn.Linear(dim_input, decoder_output_size)


    def forward(self, input):  # [batch, dim]

        if self.num_layer > 1:

            states = []

            for _ in range(self.num_layer):

                state = self.linear(input).reshape((1, -1, self.dim_out))  # [1, batch, dim_out]
                states.append(state)

            states = torch.cat(states, 0)  # [num_layer, batch, dim_out]

        else:

            states = self.linear(input).reshape((1, -1, self.dim_out))

        if self.decoder_cell_type == 'LSTM':

            return (states, states)  # (h, c)

        else:

            return states








