用于对话的cave

数据集链接：https://share.weiyun.com/5Evg5AN

word_frequency_statistics.py 可以用来统计数据集的词频

流程
1、build_vocabulary_and_abstract_embed.py 统计词频，截取词汇表

2、cvae.py进行训练
    model\util\config.py更改模型参数

把data_util和model文件右键Mark Directory as Source Root

查看日志

tensorboard --logdir log目录/run文件

运行模块
python=3.6.9
pytorch=1.2.0
tensorboardx=1.8 运行需要tensorflow

