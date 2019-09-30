import torch
import torch.nn as nn

class PrepareState(nn.Module):

    def __init__(self, dim_input,
                 decoder_output_size,
                 decoder_num_layer):

        self.num_layer = decoder_num_layer
        self.dim_out = decoder_num_layer
        self.linear = nn.Linear(dim_input, decoder_output_size)


    def forward(self, input):  # [batch, dim]

        if self.num_layer > 1:

            states = []

            for _ in range(self.num_layer):

                state = self.linear(input).reshape((1, -1, self.dim_out))  # [1, batch, dim_out]
                states.append(state)

            states = torch.cat(states, 0)  # [num_layer, batch, dim_out]

            return states

        else:

            state = self.linear(input).reshape((1, -1, self.dim_out))

            return state








