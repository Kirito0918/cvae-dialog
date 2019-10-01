from model.util.config import Config
from model.model import Model
from model.Optim import Optim
from model.util.data_iterator import DataIterator
from model.util.sentence_process import SentenceProcess
import torch
import torch.nn.functional as F
import argparse
import json
import os
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument('--trainset_path', dest='trainset_path', default='data/raw/trainset_cut300000.txt', type=str, help='训练集位置')
parser.add_argument('--validset_path', dest='validset_path', default='data/raw/validset.txt', type=str, help='验证集位置')
parser.add_argument('--testset_path', dest='testset_path', default='data/raw/testset.txt', type=str, help='测试集位置')
parser.add_argument('--embed_path', dest='embed_path', default='data/embed.txt', type=str, help='词向量位置')
parser.add_argument('--log_path', dest='log_path', default='log', type=str, help='记录模型位置')
parser.add_argument('--model_path', dest='model_path', default='log/run/.model', type=str, help='载入模型位置')
parser.add_argument('--inference', dest='inference', default=False, type=bool, help='是否测试')
parser.add_argument('--seed', dest='seed', default=666, type=int, help='随机种子')
parser.add_argument('--gpu', dest='gpu', default=False, type=bool, help='是否使用gpu')

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

    log_dir = os.path.join(args.log_path, 'run' + str(int(time.time())))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    epoch = 0
    global_step = 0

    # 载入模型
    if os.path.isfile(args.model_path):
        epoch, global_step = model.load_model(args.model_path)
        print('载入模型完成')
    else:
        model.embedding.embedding.weight = torch.nn.Parameter(torch.FloatTensor(embeds))
        print('初始化模型完成')

    if args.gpu:
        model.to('cuda')

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
                loss, nll_loss, kld_loss, ppl = comput_loss(model, data, global_step)
                print(loss.item(), nll_loss.item(), kld_loss.item(), torch.exp(ppl).item())

                optim.optimizer.zero_grad()
                loss.backward()
                optim.step()
                global_step += 1
                break

            epoch += 1
            optim.update_lr(epoch)

            # 保存模型
            log_file = os.path.join(log_dir, '%03d%012d.model' % (epoch, global_step))
            model.save_model(epoch, global_step, log_file)



def comput_loss(model, data, global_step):

    len_label = [l-1 for l in data['len_response']]  #
    mask = get_mask(len_label)  # [batch, len_decoder]

    if args.gpu:
        input_data = {'post': torch.LongTensor(data['post']).cuda(),
                      'len_post': torch.LongTensor(data['len_post']).cuda(),
                      'response': torch.LongTensor(data['response']).cuda(),
                      'len_response': torch.LongTensor(data['len_response']).cuda()}
        label = torch.LongTensor(data['response'])[:, 1:].cuda()  # 去掉start_id为标签
        mask = torch.FloatTensor(mask).cuda()

    else:
        input_data = {'post': torch.LongTensor(data['post']),
                      'len_post': torch.LongTensor(data['len_post']),
                      'response': torch.LongTensor(data['response']),
                      'len_response': torch.LongTensor(data['len_response'])}
        label = torch.LongTensor(data['response'])[:, 1:]
        mask = torch.FloatTensor(mask)

    # [batch, len_decoder, num_vocab] 对每个单词的softmax概率
    output_vocab, _mu, _logvar, mu, logvar = model(input_data, gpu=args.gpu)
    len_decoder = output_vocab.size()[1]

    output_vocab = output_vocab.reshape(-1, config.num_vocab)  # [batch*len_decoder, num_vocab]
    label = label.reshape(-1)  # [batch*len_decoder]
    mask = mask.reshape(-1)  # [batch*len_decoder]
    total_token = mask.sum()

    _nll_loss = F.nll_loss(output_vocab.log(), label, reduction='none')  # [batch*len_decoder]
    _nll_loss = _nll_loss * mask  # [batch*len_decoder]

    # print(_nll_loss)

    nll_loss = _nll_loss.reshape(-1, len_decoder).sum(1)  # 用于训练 [batch]
    ppl = _nll_loss.sum() / (total_token + 1e-4)  # 用于计算ppl

    kld_loss = gaussian_kld(mu, logvar, _mu, _logvar)  # kl散度损失 [batch]

    kld_weight = min(1.0 * global_step / config.kl_step, 1)  # kl退火

    loss = nll_loss + kld_weight * kld_loss

    return loss.mean(), nll_loss.mean(), kld_loss.mean(), ppl


def get_mask(len_label):
    mask = []
    max_len = max(len_label)  # len_decoder
    for l in len_label:
        m = [1] * l + [0] * (max_len-l)
        mask.append(m)
    return mask  # [batch, len_decoder]


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):  # [batch, latent]

    kld = 0.5 * torch.sum(prior_logvar - recog_logvar - 1
                          + recog_logvar.exp() / prior_logvar.exp()
                          + (prior_mu-recog_mu).pow(2) / prior_logvar.exp(), 1)

    return kld  # [batch]


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

