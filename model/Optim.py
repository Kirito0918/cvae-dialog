import torch
import torch.nn as nn

class Optim(object):

    def __init__(self, method,  # 优化方法
                 lr,  # 学习率
                 lr_decay=1.0,  # 学习率衰减
                 weight_decay=0.0,  # 权值decay
                 max_grad_norm=5):  # 梯度裁剪

        assert method in ['sgd', 'adam']

        self.method = method
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm


    # 设置需要优化的参数
    def set_parameters(self, parameters):

        self.params = [p for p in parameters if p.requires_grad]  # 所有需要梯度的参数

        if self.method == 'sgd':
            self.optimizer = torch.optim.SGD(self.params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.method == 'adam':
            self.optimizer = torch.optim.Adam(self.params, lr=self.lr, weight_decay=self.weight_decay)


    # 更新学习率
    def update_lr(self, epoch):

        self.lr = self.lr * self.lr_decay ** epoch  # 计算新的学习率

        for param in self.optimizer.param_groups:
            param['lr'] = self.lr


    # 更新参数
    def step(self):

        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()











