import nltk
import msgpack
import pymorphy2


class Tokenizer:
    __slots__ = ['_tokenizer', '_unpacker', '_analyzer', '_file', '_dictionary', '_filename']

    def __init__(self, *args):
        if len(args) <= 1:
            filename = u"samples/search_items.msgpack" if not args else args[0]
            self._file = open(filename, 'rb')
            self._unpacker = msgpack.Unpacker(file_like=self._file)
            self._tokenizer = nltk.tokenize.WordPunctTokenizer()
            self._analyzer = pymorphy2.MorphAnalyzer()
            self._dictionary = dict()
            self._filename = filename
        else:
            self._tokenizer, self._analyzer, self._dictionary = args

    def reopen(self):
        self._file.close()
        self._file = open(self._filename, 'rb')
        self._unpacker = msgpack.Unpacker(file_like=self._file)

    def __reduce__(self):
        return (self.__class__, (self._tokenizer, self._analyzer, self. _dictionary))

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

    def token_generator(self, epochs=1, print_delta=2000):
        counter1, counter2 = 0, 1
        for epoch in range(epochs):
            self.reopen()
            for elem in self.generator_from_msgpack():
                tokens = self.tokenize(elem['content'])
                counter1 += 1
                if counter1 == print_delta:
                    print('generating {} '.format(counter2 * print_delta))
                    counter1, counter2 = 0, counter2 + 1
                if tokens:
                    yield tokens
