from collections import defaultdict
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', dest='file_path', default='../data/raw/trainset.txt', type=str, help='输入需要统计词频的数据集')
args = parser.parse_args()


def statistics(fp):
    r""" 对数据集进行统计
    参数:
        fp: 数据集的位置
    返回:
        包含的词汇列表，按词频降序
     """
    vocab = defaultdict(int)

    with open(fp, 'r', encoding='utf8') as fr:
        data_num = 0  # 统计样本总数
        post_len = 0  # 用于统计post平均长度
        response_len = 0  # 用于统计response平均长度

        for line in fr:
            data_num += 1
            data = json.loads(line)

            post = data['post']
            response = data['response']

            post_len += len(post)
            response_len += len(response)

            for word in post:
                vocab[word] += 1
            for word in response:
                vocab[word] += 1

    vocab = dict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))  # 词频降序排列

    print(f'数据集位置: {os.path.abspath(fp)}')
    print(f'样本总数: {data_num}', end=', ')
    print('post平均长度: {:.2f}'.format(post_len / data_num), end=', ')
    print('response平均长度: {:.2f}'.format(response_len / data_num), end=', ')
    print(f'包含词汇总数: {len(vocab)}')

    return list(vocab.keys())


if __name__ == '__main__':
    statistics(args.file_path)
