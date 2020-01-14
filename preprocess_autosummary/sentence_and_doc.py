# 词、句、章都有自己的属性
class Word:
    def __init__(self, text, vector):
        self.text = text
        self.vector = vector


class Sentence:
    def __init__(self, word_list):
        self.word_list = word_list

    def len(self) -> int:
        return len(self.word_list)


class Document(Sentence):
    def __init__(self, sentence_list):
        self.sentence_list = sentence_list
        word_list = []
        for sent in self.sentence_list:
            word_list += sent.word_list
        super(Document, self).__init__(word_list)

    def len(self) -> int:
        return 1