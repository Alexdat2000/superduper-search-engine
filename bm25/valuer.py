import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class Valuer:
    def self_init(self):
        pass

    def __init__(self, *args):
        if len(args) == 1:
            self.k1 = 1.6
            self.b = 0.75

            self._tokenizer = args[0]
            self._vectorizer = CountVectorizer()
            self._tokenizer.reopen()
            self._id_by_ind = [doc['item_id'] for doc in self._tokenizer.generator_from_msgpack()]
            self._tokenizer.reopen()
            self._corpus = [doc['content'] for doc in self._tokenizer.generator_from_msgpack()]
            self._corpus_vectors = self._vectorizer.fit_transform(self._corpus)
            self._document_len = self._corpus_vectors.sum(axis=1)
            self._avgdl = self._document_len.mean()
            self._idf = dict()
            self._N = len(self._corpus)
            print(len(self._corpus), len(self._id_by_ind))
            self.fit()
        else:
            self._tokenizer, self._vectorizer, self._id_by_ind, self._corpus_vectors, \
            self._document_len, self._avgdl, self._idf, self._N, self.k1, self.b = args

    def __reduce__(self):
        return (self.__class__, (self._tokenizer, self._vectorizer, self._id_by_ind, self._corpus_vectors,
                                 self._document_len, self._avgdl, self._idf, self._N, self.k1, self.b)
                )

    def fit(self):
        is_in_text = (self._corpus_vectors > 0).sum(axis=0)
        features = self._vectorizer.get_feature_names()
        N = self._N
        for i in range(len(features)):
            self._idf[features[i]] = np.log((N - is_in_text[0, i] + 0.5) / (is_in_text[0, i] + 0.5) + 1)

    def score(self, query: str):
        scores = np.zeros((self._N, 1), dtype=np.float64)

        k1 = self.k1
        b = self.b
        avgdl = self._avgdl
        for token in self._tokenizer.tokenize(query):
            if token not in self._vectorizer.vocabulary_:
                continue
            token_ind = self._vectorizer.vocabulary_[token]
            tf = self._corpus_vectors[..., token_ind]
            scores += self._idf[token] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self._document_len / avgdl))

        scores = np.squeeze(np.asarray(scores.ravel().T))
        best_inds = np.argpartition(scores, -10)[-10:]
        best_inds = best_inds[np.argsort(scores[best_inds])][::-1]
        return [self._id_by_ind[i] for i in best_inds]
