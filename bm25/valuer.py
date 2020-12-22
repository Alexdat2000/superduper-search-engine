import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

TOP_DOCUMENTS = 100


class Valuer:
    def self_init(self):
        pass

    def dummy(self, doc):
        return doc

    def __init__(self, *args):
        if len(args) == 1:
            self.k1 = 2.0
            self.b = 0.75

            self._tokenizer = args[0]
            self._ind_by_id = dict()
            self._id_by_ind = list()
            self._tokenizer.reopen()
            self._corpus = list()

            for i, doc in enumerate(self._tokenizer.generator_from_msgpack()):
                self._ind_by_id[doc['item_id']] = i
                self._id_by_ind.append(doc['item_id'])
                self._corpus.append(doc['content'])

            self._vectorizer = CountVectorizer(tokenizer=self._tokenizer.tokenize, preprocessor=self.dummy, max_df=0.95, min_df=3)
            self._corpus_vectors = self._vectorizer.fit_transform(self._corpus)
            self._document_len = self._corpus_vectors.sum(axis=1)
            self._avgdl = self._document_len.mean()
            self._idf = dict()
            self._N = len(self._corpus)
            print(len(self._corpus), len(self._id_by_ind))
            self.fit()
        else:
            self._tokenizer, self._vectorizer, self._id_by_ind, self._corpus_vectors, \
            self._document_len, self._avgdl, self._idf, self._N, self.k1, self.b, self._ind_by_id = args

    def __reduce__(self):
        return (self.__class__, (self._tokenizer, self._vectorizer, self._id_by_ind, self._corpus_vectors,
                                 self._document_len, self._avgdl, self._idf, self._N, self.k1, self.b, self._ind_by_id)
                )

    def fit(self):
        is_in_text = (self._corpus_vectors > 0).sum(axis=0)
        features = self._vectorizer.get_feature_names()
        N = self._N
        for i in range(len(features)):
            self._idf[features[i]] = np.log((N - is_in_text[0, i] + 0.5) / (is_in_text[0, i] + 0.5) + 1)

    def score(self, query: str, doc_ids=Ellipsis):
        if doc_ids is not Ellipsis:
            doc_inds = [self._ind_by_id[i] for i in doc_ids]
            scores = np.zeros((len(doc_inds), 1), dtype=np.float64)
        else:
            doc_inds = Ellipsis
            scores = np.zeros((self._N, 1), dtype=np.float64)

        k1 = self.k1
        b = self.b
        avgdl = self._avgdl
        for token in self._tokenizer.tokenize(query):
            if token not in self._vectorizer.vocabulary_:
                continue
            token_ind = self._vectorizer.vocabulary_[token]
            tf = self._corpus_vectors[doc_inds, token_ind]
            scores += self._idf[token] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self._document_len[doc_inds] / avgdl))

        scores = np.squeeze(np.asarray(scores.ravel().T))
        if doc_ids is not Ellipsis:
            return scores
        else:
            best_inds = np.argpartition(scores, -TOP_DOCUMENTS)[-TOP_DOCUMENTS:]
            best_inds = best_inds[np.argsort(scores[best_inds])][::-1]
            return [self._id_by_ind[i] for i in best_inds]
