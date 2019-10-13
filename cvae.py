from model.util.config import Config
from model.model import Model
from model.Optim import Optim
from model.util.sentence_processor import SentenceProcessor
from model.util.data_processor import DataProcessor
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
import argparse
import json
import os
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--trainset_path', dest='trainset_path', default='data/raw/trainset.txt', type=str, help='训练集位置')
parser.add_argument('--validset_path', dest='validset_path', default='data/raw/validset.txt', type=str, help='验证集位置')
parser.add_argument('--testset_path', dest='testset_path', default='data/raw/testset.txt', type=str, help='测试集位置')
parser.add_argument('--embed_path', dest='embed_path', default='data/embed.txt', type=str, help='词向量位置')
parser.add_argument('--result_path', dest='result_path', default='result', type=str, help='测试结果位置')
parser.add_argument('--print_per_step', dest='print_per_step', default=100, type=int, help='每更新多少次参数summary学习情况')
parser.add_argument('--log_per_step', dest='log_per_step', default=30000, type=int, help='每更新多少次参数保存模型')
parser.add_argument('--log_path', dest='log_path', default='log', type=str, help='记录模型位置')
parser.add_argument('--inference', dest='inference', default=True, type=bool, help='是否测试')  #
parser.add_argument('--max_len', dest='max_len', default=60, type=int, help='测试时最大解码步数')
parser.add_argument('--model_path', dest='model_path', default='log/run1570549901/017000001891000.model', type=str, help='载入模型位置')  #
parser.add_argument('--seed', dest='seed', default=666, type=int, help='随机种子')  #
parser.add_argument('--gpu', dest='gpu', default=True, type=bool, help='是否使用gpu')  #
parser.add_argument('--max_epoch', dest='max_epoch', default=20, type=int, help='最大训练epoch')

args = parser.parse_args()  # 程序运行参数

config = Config()  # 模型配置

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


def main():

    # 载入数据集
    trainset, validset, testset = [], [], []
    if args.inference:  # 测试时只载入测试集
        with open(args.testset_path, 'r', encoding='utf8') as fr:
            for line in fr:
                testset.append(json.loads(line))
        print('载入测试集%d条' % len(testset))
    else:  # 训练时载入训练集和验证集
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
            assert len(embed) == config.embedding_size  # 检测词向量维度
            vocab.append(word)
            embeds.append(embed)
    print('载入词汇表: %d个' % len(vocab))
    print('词向量维度: %d' % config.embedding_size)

    # 通过词汇表构建一个word2index和index2word的工具
    sentence_processor = SentenceProcessor(vocab, config.pad_id, config.start_id, config.end_id, config.unk_id)

    # 创建模型
    model = Model(config)
    epoch = 0  # 训练集迭代次数
    global_step = 0  # 参数更新次数

    # 载入模型
    if os.path.isfile(args.model_path):  # 如果载入模型的位置存在则载入模型
        epoch, global_step = model.load_model(args.model_path)
        print('载入模型完成')
        # 记录模型的文件夹
        log_dir = os.path.split(args.model_path)[0]
    elif args.inference:  # 如果载入模型的位置不存在，但是又要测试，这是没有意义的
        print('请测试一个训练过的模型!')
        return
    else:  # 如果载入模型的位置不存在，重新开始训练，则载入预训练的词向量
        model.embedding.embedding.weight = torch.nn.Parameter(torch.FloatTensor(embeds))
        print('初始化模型完成')
        # 记录模型的文件夹
        log_dir = os.path.join(args.log_path, 'run' + str(int(time.time())))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    if args.gpu:
        model.to('cuda')  # 将模型参数转到gpu

    model.print_parameters()  # 输出模型参数个数

    # 定义优化器参数
    optim = Optim(config.method, config.lr, config.lr_decay, config.weight_decay, config.max_grad_norm)
    optim.set_parameters(model.parameters())  # 给优化器设置参数
    optim.update_lr(epoch)  # 每个epoch更新学习率

    # 训练
    if not args.inference:

        summary_writer = SummaryWriter(os.path.join(log_dir, 'summary'))  # 创建tensorboard记录的文件夹

        dp_train = DataProcessor(trainset, config.batch_size, sentence_processor)  # 数据的迭代器
        dp_valid = DataProcessor(validset, config.batch_size, sentence_processor, shuffle=False)

        while epoch < args.max_epoch:  # 最大训练轮数

            model.train()  # 切换到训练模式

            for data in dp_train.get_batch_data():

                start_time = time.time()

                feed_data = prepare_feed_data(data)
                loss, nll_loss, kld_loss, ppl, kld_weight = train(model, feed_data, global_step)

                optim.optimizer.zero_grad()  # 清空梯度
                loss.mean().backward()  # 反向传播
                optim.step()  # 更新参数

                use_time = time.time() - start_time

                # summary当前情况
                if global_step % args.print_per_step == 0:
                    print('epoch: %d, global_step: %d, lr: %g, nll_loss: %g, kld_loss: %g, kld_weight: %g, ppl: %g, time: %gs'
                          % (epoch, global_step, optim.lr, nll_loss.mean().item(), kld_loss.mean().item(),
                             kld_weight, ppl.mean().exp().item(), use_time))
                    summary_writer.add_scalar('train_nll', nll_loss.mean().item(), global_step)
                    summary_writer.add_scalar('train_kld', kld_loss.mean().item(), global_step)
                    summary_writer.add_scalar('train_weight', kld_weight, global_step)
                    summary_writer.add_scalar('train_ppl', ppl.mean().exp().item(), global_step)
                    summary_writer.flush()  # 将缓冲区写入文件

                global_step += 1  # 参数更新次数+1

                if global_step % args.log_per_step == 0:  # 保存模型

                    # 验证集上计算困惑度
                    model.eval()
                    nll_loss, kld_loss, ppl = valid(model, dp_valid, global_step)
                    model.train()
                    print('在验证集上的nll损失为: %g, kld损失为: %g, 困惑度为: %g' % (
                        nll_loss, kld_loss, np.exp(ppl)))
                    summary_writer.add_scalar('valid_nll', nll_loss, global_step)
                    summary_writer.add_scalar('valid_kld', kld_loss, global_step)
                    summary_writer.add_scalar('valid_ppl', np.exp(ppl), global_step)
                    summary_writer.flush()  # 将缓冲区写入文件

                    log_file = os.path.join(log_dir, '%03d%012d.model' % (epoch, global_step))
                    model.save_model(epoch, global_step, log_file)

            epoch += 1  # 数据集迭代次数+1
            optim.update_lr(epoch)  # 调整学习率

            # 保存模型
            log_file = os.path.join(log_dir, '%03d%012d.model' % (epoch, global_step))
            model.save_model(epoch, global_step, log_file)

            # 验证集上计算困惑度
            model.eval()
            nll_loss, kld_loss, ppl = valid(model, dp_valid, global_step)
            print('在验证集上的nll损失为: %g, kld损失为: %g, 困惑度为: %g' % (nll_loss, kld_loss, np.exp(ppl)))

        summary_writer.close()

    else:  # 测试

        if not os.path.exists(args.result_path):  # 创建结果文件夹
            os.makedirs(args.result_path)

        result_file = os.path.join(args.result_path, '%03d%012d.txt' % (epoch, global_step))  # 命名结果文件
        fw = open(result_file, 'w', encoding='utf8')

        dp_test = DataProcessor(testset, config.batch_size, sentence_processor, shuffle=False)

        model.eval()  # 切换到测试模式，会停用dropout等等

        nll_loss, kld_loss, ppl = valid(model, dp_test, global_step)  # 评估困惑度
        print('在测试集上的nll损失为: %g, kld损失为: %g, 困惑度为: %g' % (nll_loss, kld_loss, np.exp(ppl)))

        len_results = []  # 统计生成结果的总长度

        for data in dp_test.get_batch_data():

            posts = data['str_posts']
            responses = data['str_responses']

            feed_data = prepare_feed_data(data, inference=True)
            results = test(model, feed_data)  # 使用模型计算结果 [batch, len_decoder]

            for idx, result in enumerate(results):
                new_data = {}
                new_data['post'] = posts[idx]
                new_data['response'] = responses[idx]
                new_data['result'] = sentence_processor.index2word(result)  # 将输出的句子转回单词的形式
                len_results.append(len(new_data['result']))
                fw.write(json.dumps(new_data) + '\n')

        fw.close()
        print('生成句子平均长度: %d' % (1.0 * sum(len_results) / len(len_results)))


def prepare_feed_data(data, inference=False):

    def get_mask(length):
        """
            构建一个用于解码器损失的mask，因为超过本身句子长度的token是没必要计算损失的
        """
        masks = []
        max_len = max(length)  # len_decoder
        for l in length:
            mask = [1] * l + [0] * (max_len - l)  # 要计算损失的部分为1，其余部分为0
            masks.append(mask)
        return masks  # [batch, len_decoder]

    len_labels = [l-1 for l in data['len_responses']]  # 用于构建mask，因为标签是没有start_id的，所以长度-1
    masks = get_mask(len_labels)  # decoder损失的蒙版，因为不是每个token都要计算损失 [batch, len_decoder]
    batch_size = len(masks)

    if not inference:  # 训练时的输入

        feed_data = {'posts': torch.LongTensor(data['posts']),  # [batch, len_encoder]
                     'len_posts': torch.LongTensor(data['len_posts']),  # [batch]
                     'responses': torch.LongTensor(data['responses']),  # [batch, len_decoder]
                     'len_responses': torch.LongTensor(data['len_responses']),  # [batch]
                     'sampled_latents': torch.randn((batch_size, config.latent_size)),  # [batch, latent_size]
                     'masks': torch.FloatTensor(masks)}  # [batch, len_decoder]

    else:  # 测试时的输入

        feed_data = {'posts': torch.LongTensor(data['posts']),
                     'len_posts': torch.LongTensor(data['len_posts']),
                     'sampled_latents': torch.randn((batch_size, config.latent_size))}

    if args.gpu:  # 将数据转移到gpu上
        for key, value in feed_data.items():
            feed_data[key] = value.cuda()

    return feed_data


def comput_loss(outputs, labels, masks, global_step):

    def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):  # [batch, latent]
        """
            两个高斯分布之间的kl散度公式
        """
        kld = 0.5 * torch.sum(prior_logvar - recog_logvar - 1
                              + recog_logvar.exp() / prior_logvar.exp()
                              + (prior_mu - recog_mu).pow(2) / prior_logvar.exp(), 1)

        return kld  # [batch]

    # output_vocab: [batch, len_decoder, num_vocab] 对每个单词的softmax概率
    output_vocab, _mu, _logvar, mu, logvar = outputs  # 先验的均值、log方差，后验的均值、log方差

    token_per_batch = masks.sum(1)  # 每个样本要计算损失的token数 [batch]
    len_decoder = masks.size()[1]  # 解码长度

    output_vocab = output_vocab.reshape(-1, config.num_vocab)  # [batch*len_decoder, num_vocab]
    labels = labels.reshape(-1)  # [batch*len_decoder]
    masks = masks.reshape(-1)  # [batch*len_decoder]

    # nll_loss需要自己求log，它只是把label指定下标的损失取负并拿出来，reduction='none'代表只是拿出来，而不需要求和或者求均值
    _nll_loss = F.nll_loss((output_vocab + 1e-12).log(), labels, reduction='none')  # 每个token的-log似然 [batch*len_decoder]
    _nll_loss = _nll_loss * masks  # 忽略掉不需要计算损失的token [batch*len_decoder]

    nll_loss = _nll_loss.reshape(-1, len_decoder).sum(1)  # 每个batch的nll损失 [batch]

    ppl = nll_loss / (token_per_batch + 1e-12)  # ppl的计算需要平均到每个有效的token上 [batch]

    # kl散度损失 [batch]
    kld_loss = gaussian_kld(mu, logvar, _mu, _logvar)

    # kl退火
    # kld_weight = min(1.0 * global_step / config.kl_step, 1)  # 一次性退火
    kld_weight = min(1.0 * (global_step % (2*config.kl_step)) / config.kl_step, 1)  # 周期性退火

    # 损失
    loss = nll_loss + kld_weight * kld_loss

    return loss, nll_loss, kld_loss, ppl, kld_weight


def train(model, feed_data, global_step):
    output_vocab, _mu, _logvar, mu, logvar = model(feed_data)  # 前向传播
    outputs = (output_vocab, _mu, _logvar, mu, logvar)
    labels = feed_data['responses'][:, 1:]  # 去掉start_id
    masks = feed_data['masks']
    loss, nll_loss, kld_loss, ppl, kld_weight = comput_loss(outputs, labels, masks, global_step)  # 计算损失
    return loss, nll_loss, kld_loss, ppl, kld_weight


def valid(model, data_processor, global_step):

    nll_losses, kld_losses, ppls = [], [], []

    for data in data_processor.get_batch_data():

        feed_data = prepare_feed_data(data)
        output_vocab, _mu, _logvar, mu, logvar = model(feed_data)

        outputs = (output_vocab, _mu, _logvar, mu, logvar)
        labels = feed_data['responses'][:, 1:]  # 去掉start_id
        masks = feed_data['masks']

        _, nll_loss, kld_loss, ppl, kld_weight = comput_loss(outputs, labels, masks, global_step)

        nll_losses.extend(nll_loss.cpu().detach().numpy().tolist())
        kld_losses.extend(kld_loss.cpu().detach().numpy().tolist())
        ppls.extend(ppl.cpu().detach().numpy().tolist())

    nll_losses = np.array(nll_losses)
    kld_losses = np.array(kld_losses) * kld_weight
    ppls = np.array(ppls)

    return nll_losses.mean(), kld_losses.mean(), ppls.mean()


def test(model, feed_data):
    output_vocab, _, _, _, _ = model(feed_data, inference=True, max_len=args.max_len)
    return output_vocab.argmax(2).cpu().detach().numpy().tolist()


if __name__ == '__main__':
    main()

