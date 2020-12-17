import nltk
import msgpack
import pymorphy2


class Tokenizer:
    __slots__ = ['_tokenizer', '_unpacker', '_analyzer']

    def __init__(self, filename=u"samples/search_items.msgpack"):
        self._tokenizer = nltk.tokenize.WordPunctTokenizer()
        self._unpacker = msgpack.Unpacker(file_like=filename)
        self._analyzer = pymorphy2.MorphAnalyzer()

    def normalize_token(self, token):
        return self._analyzer.parse(token)[0].normal_form

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

    def token_generator(self, epochs=1):
        for epoch in range(epochs):
            for elem in self.generator_from_msgpack():
                tokens = self.tokenize(elem['content'])
                if tokens:
                    yield self.normalize_token(tokens)
