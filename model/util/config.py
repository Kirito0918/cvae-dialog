"""
    模型参数的类
"""

class Config(object):

    # 这些不需要改，由数据的预处理决定了，如果改预处理再改这里
    pad_id = 0
    start_id = 1
    end_id = 2
    unk_id = 3

    # 词汇表大小，根据预处理截取的词汇表设置
    num_vocab = 39000

    # 嵌入层参数，如果载入预训练的词向量，就由词向量的维度决定
    embed_size = 300

    # post编码器参数
    post_encoder_cell_type = 'LSTM'  # in ['GRU', 'LSTM']
    post_encoder_output_size = 300  # 隐藏层大小
    post_encoder_num_layer = 2  # 层数
    post_encoder_bidirection = True  # 是否是双向rnn

    # response编码器参数
    response_encoder_cell_type = 'LSTM'  # in ['GRU', 'LSTM']
    response_encoder_output_size = 300
    response_encoder_num_layer = 2
    response_encoder_bidirection = True

    # 潜变量参数
    dim_latent = 200  # 潜变量的维度

    # 先验网络参数
    dims_prior = [200]  # 先验网络多层感知机中间层的隐藏单元数，像[dim1, dim2,...,dimn]这样传入

    # 识别网络参数
    dims_recognize = [250]  # 识别网络多层感知机中间层的隐藏单元数，像[dim1, dim2,...,dimn]这样传入

    # 解码器参数
    decoder_cell_type = 'LSTM'  # in ['GRU', 'LSTM']
    decoder_output_size = 300  # 隐藏层大小
    decoder_num_layer = 2  # 层数

    # 优化参数
    batch_size = 32
    method = 'adam'  # in ['sgd', 'adam']
    lr = 0.0001  # 初始学习率
    lr_decay = 1.0  # 学习率衰减，每过1个epoch衰减的百分比
    weight_decay = 0  # 权值decay
    max_grad_norm = 5
    kl_step = 10000  # 更新多少次参数之后kl项权值达到1
    dropout = 0  # 这里只有编解码器设置了dropout



