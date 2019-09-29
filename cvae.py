from model.util.config import Config
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--trainset_path', dest='trainset_path', default='data/raw/trainset_cut300000.txt', type=str, help='训练集位置')
parser.add_argument('--validset_path', dest='validset_path', default='data/raw/validset.txt', type=str, help='验证集位置')
parser.add_argument('--testset_path', dest='testset_path', default='data/raw/testset.txt', type=str, help='测试集位置')
parser.add_argument('--embed_path', dest='embed_path', default='data/embed.txt', type=str, help='词向量位置')
parser.add_argument('--embed_dim', dest='embed_dim', default=300, type=int, help='词向量维度')
parser.add_argument('--log_path', dest='log_path', default='log', type=str, help='记录模型位置')
parser.add_argument('--model_path', dest='model_path', default='log', type=str, help='载入模型位置')
parser.add_argument('--inference', dest='inference', default=False, type=bool, help='是否测试')
args = parser.parse_args()

args = dict(args)
config = Config()

def main():

    trainset, validset, testset = [], [], []

    # 载入数据集
    if args['inference']:
        with open(args['testset_path'], 'r', encoding='utf8') as fr:
            for line in fr:
                testset.append(json.loads(line))
        print('载入测试集%d条' % len(testset))

    else:
        with open(args['trainset_path'], 'r', encoding='utf8') as fr:
            for line in fr:
                trainset.append(json.loads(line))
        print('载入训练集%d条' % len(trainset))
        with open(args['validset_path'], 'r', encoding='utf8') as fr:
            for line in fr:
                validset.append(json.loads(line))
        print('载入验证集%d条' % len(validset))

    # 创建模型

    # 载入模型

    # 训练

    # 评估

    # 测试


if __name__ == '__main__':
    main()

