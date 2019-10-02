import torch.nn as nn

# 解码器
class Decoder(nn.Module):

    def __init__(self, cell_type,  # rnn类型
                 input_size,  # 输入维度
                 output_size,  # 输出维度
                 num_layer,  # rnn层数
                 dropout=0):  # dropout
        super(Decoder, self).__init__()

        assert cell_type in ['GRU', 'LSTM']  # 限定rnn类型

        self.cell_type = cell_type
        self.rnncell = getattr(nn, cell_type)(  # rnncell
            input_size=input_size,
            hidden_size=output_size,
            num_layers=num_layer,
            dropout=dropout)


    def forward(self, input,  # 输入 [seq, batch, dim] 或者单步输入 [1, batch, dim]
                state):  # 初始状态 [layers*directions, batch, dim]

        # output = [seq, batch, dim*directions]  每个时间步的输出
        # final_state = [layers*directions, batch, dim]  # 每一层的最终状态
        output, final_state = self.rnncell(input, state)

        return output, final_state