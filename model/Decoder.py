import torch.nn as nn


class Decoder(nn.Module):
    r""" 解码器 """
    def __init__(self, cell_type,  # rnn类型
                 input_size,  # 输入维度
                 output_size,  # 输出维度
                 num_layer,  # rnn层数
                 dropout=0.1):  # dropout
        super(Decoder, self).__init__()
        assert cell_type in ['GRU', 'LSTM']  # 限定rnn类型

        self.cell_type = cell_type
        self.rnn_cell = getattr(nn, cell_type)(
            input_size=input_size,
            hidden_size=output_size,
            num_layers=num_layer,
            dropout=dropout)

    def forward(self, x,  # 输入 [seq, batch, dim] 或者单步输入 [1, batch, dim]
                state):  # 初始状态 [layers*directions, batch, dim]
        # output: [seq, batch, dim*directions] 每个时间步的输出
        # final_state: [layers*directions, batch, dim] 每一层的最终状态
        output, final_state = self.rnn_cell(x, state)
        return output, final_state
