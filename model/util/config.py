class Config(object):

    # 这些不需要改
    pad_id = 0
    start_id = 1
    end_id = 2

    # 词汇表大小，根据预处理设置
    num_vocab = 39000

    # 嵌入层参数
    embed_size = 300

    # post编码器参数
    post_encoder_cell_type = 'LSTM'  # in ['GRU', 'LSTM']
    post_encoder_output_size = 300
    post_encoder_num_layer = 2
    post_encoder_bidirection = True

    # response编码器参数
    response_encoder_cell_type = 'LSTM'  # in ['GRU', 'LSTM']
    response_encoder_output_size = 300
    response_encoder_num_layer = 2
    response_encoder_bidirection = True

    # 潜变量参数
    dim_latent = 100

    # 先验网络参数
    prior_dims = [200]

    # 识别网络参数
    recognize_dims = [200]

    # 解码器参数
    decoder_cell_type = 'LSTM'  # in ['GRU', 'LSTM']
    decoder_output_size = 300
    decoder_num_layer = 2

    # 优化参数
    dropout = 0

    #


