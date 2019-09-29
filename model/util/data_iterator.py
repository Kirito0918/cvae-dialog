"""
    将data传入，每次返回batch_size个样本
"""

class DataIterator(object):

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.data_num = len(data)
        self.st = 0
        self.ed = 0

    def get_batch_data(self):

        while(self.st < self.data_num):

            self.ed = self.st + self.batch_size

            if self.ed < self.data_num:
                data = self.data[self.st: self.ed]
            else:
                data = self.data[self.st:]

            yield data

            self.st = self.ed
