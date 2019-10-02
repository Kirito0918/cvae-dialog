"""
    构建词汇表并提取出需要的预训练的词向量
"""

from word_frequency_statistics import statistics
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', dest='train_path', default='../data/raw/trainset.txt', type=str, help='训练集位置')
parser.add_argument('--valid_path', dest='valid_path', default='../data/raw/validset.txt', type=str, help='验证集位置')
parser.add_argument('--test_path', dest='test_path', default='../data/raw/testset.txt', type=str, help='测试集位置')
parser.add_argument('--glove_path', dest='glove_path', default='../data/raw/glove.840B.300d.txt', type=str, help='预训练的词向量位置')
parser.add_argument('--glove_dim', dest='glove_dim', default=300, type=int, help='预训练的词向量维度')
parser.add_argument('--output_path', dest='output_path', default='../data/embed.txt', type=str, help='输出结果位置')
parser.add_argument('--num_vocabulary', dest='num_vocabulary', default=39000, type=int, help='选取词汇表大小')
parser.add_argument('--pad_token', dest='pad_token', default='<pad>', type=str, help='pad的记法')
parser.add_argument('--start_token', dest='start_token', default='<s>', type=str, help='start的记法')
parser.add_argument('--end_token', dest='end_token', default='</s>', type=str, help='end的记法')
parser.add_argument('--unk_token', dest='unk_token', default='<unk>', type=str, help='unk的记法')
args = parser.parse_args()


def build_vocabulary(trainp, vp, testp, vn=30000):
    """
        构建词汇表

    参数:
        trainp: 训练集位置
        vp: 验证集位置
        testp: 测试集位置
        vn: 截取词汇表大小

    返回:
        最终词汇表的列表
    """
    vob_head = [args.pad_token] + [args.start_token] + [args.end_token] + [args.unk_token]
    trainset_vob = vob_head + statistics(trainp)  # 训练集词汇表
    validset_vob = vob_head + statistics(vp)  # 验证集词汇表
    testset_vob = vob_head + statistics(testp)  # 测试集词汇表

    trainset_vob_len = len(trainset_vob)  # 训练集词汇表大小
    validset_vob_len = len(validset_vob)  # 验证集词汇表大小
    testset_vob_len = len(testset_vob)  # 测试集词汇表大小

    num_vocab = min(vn, len(trainset_vob))  # 截取词汇表大小
    vocab = trainset_vob[:num_vocab]  # 最终词汇表

    cover_valid = 100.0 * len(set(vocab) & set(validset_vob)) / validset_vob_len
    cover_test = 100.0 * len(set(vocab) & set(testset_vob)) / testset_vob_len

    # print('训练集词汇表:', trainset_vob)
    print('训练集词汇表大小: %d' % trainset_vob_len, end=', ')
    # print('验证集集词汇表:', validset_vob)
    print('验证集词汇表大小: %d' % validset_vob_len, end=', ')
    # print('测试集词汇表:', testset_vob)
    print('测试集词汇表大小: %d' % testset_vob_len, end=', ')
    print('截取词汇表大小: %d' % len(vocab), end=', ')
    print('词汇表覆盖验证集%.2f%%的词汇' % cover_valid, end=', ')
    print('词汇表覆盖测试集%.2f%%的词汇' % cover_test)
    print('最终词汇表:', vocab)

    return vocab


def abstract_embed(vocab, gp, op):
    """
        根据词汇表从预训练的词向量中选取需要的

    参数:
        vocab: 词汇表
        gp: 预训练的词向量位置
        op: 输出的位置
    """
    vectors = {}

    # 载入预训练的词向量
    with open(gp, 'r', encoding='utf8') as fr:
        for line in fr:
            line = line.strip()
            word = line[: line.find(' ')]
            vector = line[line.find(' ')+1:]
            vectors[word] = vector

    # 提取词汇表需要的词向量
    not_in_vectors = 0
    embeds = {}
    for word in vocab:
        if word in vectors:
            embeds[word] = vectors[word]
        else:
            not_in_vectors += 1
            embeds[word] = ' '.join(['0'] * args.glove_dim)


    # 保存结果
    with open(op, 'w', encoding='utf8') as fw:
        for key, value in embeds.items():
            fw.write(key + ' ' + value + '\n')

    print('词汇表%.2f%%的词向量是预训练过的' % (100.0 - 100.0 * not_in_vectors / len(vocab)))
    print('词嵌入输出位置%s' % os.path.abspath(op))


if __name__ == '__main__':
    vocab = build_vocabulary(args.train_path, args.valid_path, args.test_path, args.num_vocabulary)
    abstract_embed(vocab, args.glove_path, args.output_path)

