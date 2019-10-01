from model.util.config import Config
from model.model import Model
from model.Optim import Optim
from model.util.data_iterator import DataIterator
from model.util.sentence_process import SentenceProcess
import torch
import argparse
import json
import os
import numpy as np
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument('--trainset_path', dest='trainset_path', default='data/raw/trainset_cut300000.txt', type=str, help='训练集位置')
parser.add_argument('--validset_path', dest='validset_path', default='data/raw/validset.txt', type=str, help='验证集位置')
parser.add_argument('--testset_path', dest='testset_path', default='data/raw/testset.txt', type=str, help='测试集位置')
parser.add_argument('--embed_path', dest='embed_path', default='data/embed.txt', type=str, help='词向量位置')
parser.add_argument('--log_path', dest='log_path', default='log', type=str, help='记录模型位置')
parser.add_argument('--model_path', dest='model_path', default='log', type=str, help='载入模型位置')
parser.add_argument('--inference', dest='inference', default=False, type=bool, help='是否测试')
parser.add_argument('--seed', dest='seed', default=666, type=int, help='随机种子')
args = parser.parse_args()

config = Config()


# 确定随机种子，避免初始化对调参的影响
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


def main():

    trainset, validset, testset = [], [], []

    # 载入数据集
    if args.inference:
        with open(args.testset_path, 'r', encoding='utf8') as fr:
            for line in fr:
                testset.append(json.loads(line))
        print('载入测试集%d条' % len(testset))

    else:
        with open(args.trainset_path, 'r', encoding='utf8') as fr:
            for line in fr:
                trainset.append(json.loads(line))
        print('载入训练集%d条' % len(trainset))
        with open(args.validset_path, 'r', encoding='utf8') as fr:
            for line in fr:
                validset.append(json.loads(line))
        print('载入验证集%d条' % len(validset))

    # 载入词汇表，词向量
    vocab, embeds = [], []
    with open(args.embed_path, 'r', encoding='utf8') as fr:
        for line in fr:
            line = line.strip()
            word = line[: line.find(' ')]
            vec = line[line.find(' ') + 1:].split()
            embed = [float(v) for v in vec]
            assert len(embed) == config.embed_size
            vocab.append(word)
            embeds.append(embed)
    print('载入词汇表: %d个' % len(vocab))
    print('词向量维度: %d' % config.embed_size)

    sentence_processor = SentenceProcess(vocab, config.pad_id, config.start_id, config.end_id, config.unk_id)

    # 创建模型
    model = Model(config)

    epoch = 0
    global_step = 0

    # save_dir = os.path.join(args.log_path, str(int(time.time())))  # 模型保存位置
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)

    # 载入模型
    if os.path.isfile(args.model_path):
        epoch, global_step = model.load_model(args.model_path)
        print('载入模型完成')
    else:
        model.embedding.embedding.weight = torch.nn.Parameter(torch.FloatTensor(embeds))
        print('初始化模型完成')

    model.print_parameters()

    # 定义优化器
    optim = Optim(config.method, config.lr, config.lr_decay, config.weight_decay, config.max_grad_norm)
    optim.set_parameters(model.parameters())
    optim.update_lr(epoch)

    # 测试
    if args.inference:

        pass

    else:  #训练

        while True:

            random.shuffle(trainset)
            data_it = DataIterator(trainset, config.batch_size)

            for data in data_it.get_batch_data():

                data = prepare_batch_data(data, sentence_processor)
                train(model, data, optim)




                # 训练流程
                global_step += 1

            epoch += 1


def train(model, data, optim):

    input_data = {'post': torch.LongTensor(data['post']),
                  'len_post': torch.LongTensor(data['len_post']),
                  'response': torch.LongTensor(data['response']),
                  'len_response': torch.LongTensor(data['len_response'])}

    output_vocab, _mu, _logvar, mu, logvar = model(input_data)

    print(output_vocab)



def prepare_batch_data(data, sp):

    str_post, str_response = [], []

    for item in data:
        str_post.append(item['post'])
        str_response.append(item['response'])

    len_post = [len(post)+2 for post in str_post]
    len_response = [len(response)+2 for response in str_response]

    max_post_len = max(len_post)
    max_response_len = max(len_response)

    id_post, id_response = [], []

    for post in str_post:
        ids = sp.word2index(post, max_post_len)
        id_post.append(ids)

    for response in str_response:
        ids = sp.word2index(response, max_response_len)
        id_response.append(ids)

    return {'post': id_post,  # [batch, maxlen]
            'len_post': len_post,  # [batch]
            'response': id_response,
            'len_response': len_response}

if __name__ == '__main__':
    main()

