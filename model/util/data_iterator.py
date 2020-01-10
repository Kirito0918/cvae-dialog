
class DataIterator(object):
    r""" 将data传入，每次返回batch_size个样本 """
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.num_data = len(data)
        self.st = 0
        self.ed = 0

    def get_batch_data(self):
        while self.st < self.num_data:
            self.ed = self.st + self.batch_size
            if self.ed < self.num_data:
                data = self.data[self.st: self.ed]
            else:
                data = self.data[self.st:]
            yield data
            self.st = self.ed
