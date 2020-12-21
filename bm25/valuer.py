import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class Valuer:
    def self_init(self):
        pass

    def __init__(self, tokenizer):
        self.k1 = 1.6
        self.b = 0.75

        self._tokenizer = tokenizer
        self._vectorizer = CountVectorizer()
        self._tokenizer.reopen()
        self._id_by_ind = [doc['item_id'] for doc in self._tokenizer.generator_from_msgpack()]
        self._tokenizer.reopen()
        self._corpus = [doc['content'] for doc in self._tokenizer.generator_from_msgpack()]
        self._corpus_vectors = self._vectorizer.fit_transform(self._corpus)
        self._document_len = self._corpus_vectors.sum(axis=1)
        self._avgdl = self._document_len.mean()
        self._idf = dict()
        self.N = len(self._corpus)
        print(len(self._corpus), len(self._id_by_ind))
        self.fit()

    def fit(self):
        is_in_text = (self._corpus_vectors > 0).sum(axis=0)
        features = self._vectorizer.get_feature_names()
        N = self.N
        for i in range(len(features)):
            self._idf[features[i]] = np.log((N - is_in_text[0, i] + 0.5) / (is_in_text[0, i] + 0.5) + 1)

    def score_document(self, query: str, doc_ind: int):
        score = 0
        k1 = self.k1
        b = self.b
        avgdl = self._avgdl
        for token in self._tokenizer.tokenize(query):
            if token not in self._vectorizer.vocabulary_:
                continue
            token_ind = self._vectorizer.vocabulary_[token]
            tf = self._corpus_vectors[doc_ind, token_ind]
            score += self._idf[token] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self._document_len[doc_ind] / avgdl))
        return score

    def score(self, query: str):
        scores = np.empty(self.N, dtype=np.int64)
        for i in range(len(self._corpus)):
            scores[i] = self.score_document(query, i)
        best_inds = np.argpartition(scores, -10)[-10:]
        return [self._id_by_ind[i] for i in best_inds]