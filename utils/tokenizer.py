import nltk
import msgpack
import pymorphy2


class Tokenizer:
    __slots__ = ['_tokenizer', '_unpacker', '_analyzer', '_file', '_dictionary']

    def __init__(self, filename=u"samples/search_items.msgpack"):
        self._file = open(filename, 'rb')
        self._unpacker = msgpack.Unpacker(file_like=self._file)
        self._tokenizer = nltk.tokenize.WordPunctTokenizer()
        self._analyzer = pymorphy2.MorphAnalyzer()
        self._dictionary = dict()

    def reopen(self, filename=u"samples/search_items.msgpack"):
        self._file.close()
        self._file = open(filename, 'rb')
        self._unpacker = msgpack.Unpacker(file_like=self._file)

    def normalize_token(self, token):
        result = self._dictionary.get(token)
        if result is None:
            result = self._analyzer.parse(token)[0].normal_form
            self._dictionary[token] = result
        return result

    def generator_from_msgpack(self):
        for unpacked in self._unpacker:
            yield unpacked

    def tokenize(self, text):
        result = []
        text = text.lower()
        for w in self._tokenizer.tokenize(text):
            if len(w) <= 3 and not w.isalnum():
                continue
            result.append(self.normalize_token(w))
        return result

    def token_generator(self, epochs=1, print_delta=200):
        counter1, counter2 = 0, 1
        for epoch in range(epochs):
            for elem in self.generator_from_msgpack():
                tokens = self.tokenize(elem['content'])
                counter1 += 1
                if counter1 == print_delta:
                    print('generating {} '.format(counter2 * print_delta))
                    counter1, counter2 = 0, counter2 + 1
                if tokens:
                    yield [token for token in tokens]
