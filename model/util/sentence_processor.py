
class SentenceProcessor(object):
    r""" 实现了句子的word2index，index2word和pad """
    def __init__(self, vocab, pad_id, start_id, end_id, unk_id):
        self.vocab = vocab  # index to vocab的词汇表
        self.pad_id = pad_id
        self.start_id = start_id
        self.end_id = end_id
        self.unk_id = unk_id
        self.v2i = dict(zip(vocab, range(len(vocab))))  # vocab to index的字典

    def word2index(self, sentence):
        r""" 将一句句子转化成它的id表示 """
        id_sentence = []
        for word in sentence:
            if word in self.v2i:  # 单词在词汇表中
                idx = self.v2i[word]
            else:  # 单词不在词汇表中
                idx = self.unk_id
            id_sentence.append(idx)
        len_sentence = len(id_sentence)
        return id_sentence, len_sentence

    def index2word(self, id_sentence):
        r""" 将句子的id转化成单词表示 """
        sentence = []
        for idx in id_sentence:
            if idx == self.end_id:
                break
            else:
                sentence.append(self.vocab[idx])
        return sentence

    def pad_sentence(self, sentence, length):
        r""" 将句子pad到length长度 """
        assert len(sentence) + 2 <= length
        sentence = [self.start_id] + sentence + [self.end_id]
        for _ in range(length - len(sentence)):
            sentence.append(self.pad_id)
        return sentence
