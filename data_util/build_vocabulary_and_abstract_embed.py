"""
    构建词汇表并提取出需要的预训练的词向量
"""

from word_frequency_statistics import statistics
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', dest='train_path', default='../data/raw/trainset_cut300000.txt', type=str, help='训练集位置')
parser.add_argument('--valid_path', dest='valid_path', default='../data/raw/validset.txt', type=str, help='验证集位置')
parser.add_argument('--test_path', dest='test_path', default='../data/raw/testset.txt', type=str, help='测试集位置')
parser.add_argument('--num_vocabulary', dest='num_vocabulary', default=30000, type=int, help='选取词汇表大小')
parser.add_argument('--pad_token', dest='pad_token', default='<pad>', type=str, help='pad的记法')
parser.add_argument('--start_token', dest='start_token', default='<s>', type=str, help='start的记法')
parser.add_argument('--end_token', dest='end_token', default='</s>', type=str, help='end的记法')
args = parser.parse_args()

def build_vocabulary(trainp, vp, testp):
    vob_head = [args.pad_token] + [args.start_token] + [args.end_token]
    trainset_vob = vob_head + statistics(trainp)  # 训练集词汇表
    validset_vob = vob_head + statistics(vp)  # 验证集词汇表
    testset_vob = vob_head + statistics(testp)  # 测试集词汇表

    trainset_vob_len = len(trainset_vob)  # 训练集词汇表大小
    validset_vob_len = len(validset_vob)  # 验证集词汇表大小
    testset_vob_len = len(testset_vob)  # 测试集词汇表大小

    vocab_num = max(args.num_vocabulary, len(trainset_vob))  # 截取词汇表大小
    vocab = trainset_vob[:vocab_num]  # 最终词汇表

    cover_valid = 100.0 * len(set(vocab) & set(validset_vob)) / validset_vob_len
    cover_test = 100.0 * len(set(vocab) & set(testset_vob)) / testset_vob_len

    # print('训练集词汇表:', trainset_vob)
    print('训练集词汇表大小: %d' % trainset_vob_len)
    # print('验证集集词汇表:', validset_vob)
    print('验证集词汇表大小: %d' % validset_vob_len)
    # print('测试集词汇表:', testset_vob)
    print('测试集词汇表大小: %d' % testset_vob_len)
    print('截取词汇表大小: %d' % len(vocab))
    print('词汇表覆盖验证集%f\%的词汇' % cover_valid)
    print('词汇表覆盖测试集%f\%的词汇' % cover_test)
    print('最终词汇表:', vocab)

    return vocab


if __name__ == '__main__':
    build_vocabulary(args.train_path, args.valid_path, args.test_path)






