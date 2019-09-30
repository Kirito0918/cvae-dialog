import torch
import torch.nn as nn

class Decoder(nn.Module):

    def __init__(self, cell_type,  # rnn类型
                 input_size,  # 输入维度
                 output_size,  # 输出维度
                 num_layer,  # rnn层数
                 bidirection=False,  # 是否双向
                 dropout=0):  # dropout

        assert cell_type in ['gru', 'lstm']  # 限定rnn类型
        if bidirection:  # 如果双向
            assert output_size % 2 == 0

        cell_size = output_size // 2  # rnn维度

        self.bidirection = bidirection
        self.cell_type = cell_type
        self.rnncell = getattr(nn, cell_type)(  # rnncell
            input_size=input_size,
            hidden_size=cell_size,
            num_layers=num_layer,
            bidirectional=bidirection,
            dropout=dropout)


    def forward(self, input,  # 输入 [seq, batch, dim] 或者单步输入 [1, batch, dim]
                state):  # 初始状态 [layers*directions, batch, dim]

        # output = [seq, batch, dim*directions]  每个时间步的输出
        # final_state = [layers*directions, batch, dim]  # 每一层的最终状态
        output, final_state = self.rnncell(input, state)

        if self.bidirection:  # 如果是双向的，对双向进行拼接作为每层的最终状态

            if self.cell_type == 'gru':
                final_state_forward = final_state[0::2]  # [layers, batch, dim]
                final_state_back = final_state[1::2]  # [layers, batch, dim]
                final_state = torch.cat([final_state_forward, final_state_back], 2)  # [layers, batch, dim*2]

            else:
                final_state_h, final_state_c = final_state
                final_state_h = torch.cat([final_state_h[0::2], final_state_h[1::2]], 2)
                final_state_c = torch.cat([final_state_c[0::2], final_state_c[1::2]], 2)
                final_state = (final_state_h, final_state_c)

        return output, final_state