
class SentenceProcess(object):

    def __init__(self, vocab, pad_id, start_id, end_id, unk_id):

        self.vocab = vocab
        self.pad_id = pad_id
        self.start_id = start_id
        self.end_id = end_id
        self.unk_id = unk_id
        self.v2i = dict(zip(vocab, range(len(vocab))))

    # 将句子转化成id表示并补齐到length长度
    def word2index(self, sentence, length):

        assert length >= (len(sentence)+2)

        id_sentence = [self.start_id]

        for word in sentence:

            if word in self.v2i:
                idx = self.v2i[word]
            else:
                idx = self.unk_id

            id_sentence.append(idx)

        id_sentence.append(self.end_id)

        len_sentence = len(id_sentence)

        for _ in range(length - len_sentence):
            id_sentence.append(self.pad_id)

        return id_sentence

        # 将句子转化成id表示并补齐到length长度

    def index2word(self, id_sentence):

        sentence = []

        for idx in id_sentence:

            if idx == self.end_id:
                break
            else:
                sentence.append(self.vocab[idx])

        return sentence














