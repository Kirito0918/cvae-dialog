import torch
import torch.nn as nn
from Embedding import WordEmbedding
from Encoder import SentenseEncoder
from PriorNet import PriorNet
from RecognizeNet import RecognizeNet
from Decoder import Decoder
from PrepareState import PrepareState

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

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

        # 初始化解码器状态
        self.prepare_state = PrepareState(config.post_encoder_output_size+config.dim_latent,
                                          config.decoder_output_size,
                                          config.decoder_num_layer)

        # 解码器
        self.decoder = Decoder(config.decoder_cell_type,  # rnn类型
                               config.embed_size,  # 输入维度
                               config.decoder_output_size,  # 输出维度
                               config.decoder_num_layer,  # rnn层数
                               config.dropout)  # dropout概率)

        # 输出层
        self.projector = nn.Sequential(
            nn.Linear(config.decoder_output_size, config.num_vocab),
            nn.Softmax()
        )



    def forward(self, input,  # input
                inference=False,
                max_len=60):

        id_post = input['post']  # [batch, seq]
        len_post = input['len_post']  # [batch]
        id_response = input['response']  # [batch, seq]
        len_response = input['len_response']  # [batch, seq]

        batch_size = id_post.size()[0]

        if inference:  # 测试

            embed_post = self.embedding(id_post)  # [batch, seq, embed_size]

            # state = [layers, batch, dim]
            _, post_state = self.post_encoder(embed_post.transpose(0, 1), len_post)

            x = post_state[-1, :, :]  # [batch, dim]

            # p(z|x)
            _mu, _logvar = self.prior_net(x)  # [batch, latent]

            # 采样
            nz = torch.randn((batch_size, self.config.dim_latent))  # [batch, latent]

            # 重参数化
            z = _mu + torch.exp(0.5 * _logvar) * nz  # [batch, latent]

            first_state = self.prepare_state(torch.cat([z, x], 1))  # [num_layer, batch, dim_out]

            outputs = []
            done = torch.BoolTensor([0]*batch_size)  # 句子解码完成标志

            for idx in range(max_len):

                if idx == 0:
                    state = first_state  # 解码器初始状态
                    id_input = torch.ones((1, batch_size))*self.config.start_id
                    input = self.embedding(id_input)  # (1, batch, embed_size)

                # output: [1, batch, dim_out]
                # state: [num_layer, batch, dim_out]
                output, state = self.decoder(input, state)

                outputs.append(output)

                vocab_prob = self.projector(output)  # [1, batch, num_vocab]
                next_input_id = torch.argmax(vocab_prob, 2)  # [1, batch]

                _done = next_input_id.squeeze(0) == self.config.end_id  # [batch]
                done = done | _done

                if done.sum() == batch_size:  # 如果全部解码完成则提前停止

                    break
                else:

                    input = self.embedding(next_input_id)  # [1, batch, embed_size]

            outputs = torch.cat(outputs, 0).transpose(0, 1)

            output_vocab = self.projector(outputs)

            return output_vocab, _mu, _logvar, None, None

        else:  # 训练

            embed_post = self.embedding(id_post)  # [batch, seq, embed_size]
            embed_response = self.embedding(id_response)  # [batch, seq, embed_size]

            decoder_input = embed_response[:, :-1, :].transpose(0, 1)  # [seq-1, batch, embed_size]
            len_decoder = decoder_input.size()[0]  # seq-1
            decoder_input = decoder_input.split([1] * len_decoder, 0)  # seq-1个[1, batch, embed_size]

            # state = [layers, batch, dim]
            _, post_state = self.post_encoder(embed_post.transpose(0, 1), len_post)
            _, response_state = self.response_encoder(embed_response.transponse(0, 1), len_response)

            x = post_state[-1, :, :]  # [batch, dim]
            y = response_state[-1, :, :]  # [batch, dim]

            # p(z|x)
            _mu, _logvar = self.prior_net(x)  # [batch, latent]

            # p(z|x,y)
            mu, logvar = self.recognize_net(x, y)  # [batch, latent]

            # 采样
            nz = torch.randn((batch_size, self.config.dim_latent))  # [batch, latent]

            # 重参数化
            z = mu + torch.exp(0.5*logvar) * nz  # [batch, latent]

            first_state = self.prepare_state(torch.cat([z, x], 1))  # [num_layer, batch, dim_out]

            outputs = []

            for idx in range(len_decoder):

                state = first_state  # 解码器初始状态

                input = decoder_input[idx]  # 当前时间步输入 [1, batch, embed_size]

                # output: [1, batch, dim_out]
                # state: [num_layer, batch, dim_out]
                output, state = self.decoder(input, state)

                outputs.append(output)

            outputs = torch.cat(outputs, 0).transpose(0, 1)  # [batch, seq-1, dim_out]

            output_vocab = self.projector(outputs)  # [batch, seq-1, num_vocab]

            return output_vocab, _mu, _logvar, mu, logvar

    # 统计参数
    def print_parameters(self):

        def statistic_param(params):
            total_num = 0  # 参数总数
            for param in params:
                num = 1
                if param.requires_grad:
                    size = param.size()
                    for dim in size:
                        num *= dim
                total_num += num
            return total_num

        print("嵌入层参数个数: %d" % statistic_param(self.embedding.parameters()))
        print("post编码器参数个数: %d" % statistic_param(self.post_encoder.parameters()))
        print("response编码器参数个数: %d" % statistic_param(self.response_encoder.parameters()))
        print("先验网络参数个数: %d" % statistic_param(self.prior_net.parameters()))
        print("识别网络参数个数: %d" % statistic_param(self.recognize_net.parameters()))
        print("解码器初始状态参数个数: %d" % statistic_param(self.prepare_state.parameters()))
        print("解码器参数个数: %d" % statistic_param(self.decoder.parameters()))
        print("输出层参数个数: %d" % statistic_param(self.projector.parameters()))
        print("参数总数: %d" % statistic_param(self.parameters()))








