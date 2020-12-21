import gensim
import hnswlib
import numpy as np
import pickle

class Word2Vec:
    def __init__(self, tokenizer, idf):
        self._model = gensim.models.Word2Vec(size=96, min_count=5, max_vocab_size=100000, workers=4)
        self._hnsw = 0  # how can i specify the type?
        self._document_vectors = np.ndarray(shape=(1, 96))
        self._word_to_index = dict()
        self._index_to_word = dict()
        self._ids = []
        self._idf = idf
        self._tokenizer = tokenizer
        self._vectors = np.array([])

    def build_model(self):
        self._model.build_vocab(self._tokenizer.token_generator(), progress_per=1000)

    def train(self, epochs):
        print("training model ({} epochs)...".format(epochs))
        self._tokenizer.reopen()
        items_count = sum([1 for _ in self._tokenizer.generator_from_msgpack()])
        self._model.train(self._tokenizer.token_generator(epochs),
                          total_examples=items_count,
                          epochs=epochs, report_delay=10.)
        self._model.init_sims(replace=True)

    def build_hnsw(self):
        self.build_knn_index(self._document_vectors, "hnsw", space='cosine')

    def build_knn_index(self, embeddings, output_path, space='ip'):
        assert space in ('ip', 'cosine')
        count = embeddings.shape[0]
        dimension = embeddings.shape[1]

        self._hnsw = hnswlib.Index(space=space, dim=dimension)  # possible options are l2, cosine or ip
        self._hnsw.init_index(max_elements=count, ef_construction=300, M=32)

        batch_size = 10000
        data_labels = np.arange(count)
        for batch_start in range(0, count, batch_size):
            self._hnsw.add_items(embeddings[batch_start: batch_start + batch_size, :],
                                 data_labels[batch_start: batch_start + batch_size])

        self._hnsw.set_ef(300)  # ef should always be > k
        self._hnsw.save_index(output_path)

    def load_knn_index(self, dimension, file_path, space='ip'):
        assert space in ('ip', 'cosine')

        self._hnsw = hnswlib.Index(space=space, dim=dimension)  # possible options are l2, cosine or ip

        self._hnsw.load_index(file_path)
        self._hnsw.set_ef(300)

    def build_and_train(self, epochs=1):
        print("building model...")
        self.build_model()
        self.train(epochs)
        self._word_to_index = {w: data.index for w, data in self._model.wv.vocab.items()}
        self._index_to_word = {data.index: w for w, data in self._model.wv.vocab.items()}
        self._vectors = self._model.wv.vectors
        print('preparing hnsw...')
        self.init_document_vectors()
        self.build_hnsw()
        # save_all_data()

    def init_document_vectors(self):
        self._tokenizer.reopen()
        document_list = []
        for document in self._tokenizer.generator_from_msgpack():
            tokens = self._tokenizer.tokenize(document['content'])
            self._ids.append(document['item_id'])
            if tokens:
                document_list.append(sum(
                    self._vectors[self._word_to_index[token]] * (self._idf[token] if self._idf.get(token) else 0.01)
                    if self._word_to_index.get(token) else np.zeros(96) for token in tokens))
            else:
                document_list.append(np.zeros(96))
        self._document_vectors = np.vstack(document_list)

    def save(self, file_path='word2vec/'):
        with open(file_path + 'vectors', 'wb') as vec_file:
            pickle.dump(self._vectors, vec_file)
        self._hnsw.save_index(file_path + 'hnsw')
        with open(file_path + 'ids', 'wb') as ids_file:
            pickle.dump(self._ids, ids_file)
        with open(file_path + 'word_to_index', 'wb') as w2i_file:
            pickle.dump(self._word_to_index, w2i_file)
        with open(file_path + 'index_to_word', 'wb') as i2w_file:
            pickle.dump(self._index_to_word, i2w_file)

    def load(self, file_path='word2vec/'):
        with open(file_path + 'vectors', 'rb') as vec_file:
            self._vectors = pickle.load(vec_file)
        self.load_knn_index(96, file_path + 'hnsw', 'cosine')
        with open(file_path + 'ids', 'rb') as ids_file:
            self._ids = pickle.load(ids_file)
        with open(file_path + 'word_to_index', 'rb') as w2i_file:
            self._word_to_index = pickle.load(w2i_file)
        with open(file_path + 'index_to_word', 'rb') as i2w_file:
            self._index_to_word = pickle.load(i2w_file)

    def evaluate(self, text, k):
        tokens = self._tokenizer.tokenize(text)
        query_vector = sum(
            self._vectors[self._word_to_index[token]] * (self._idf[token] if self._idf.get(token) else 0.01)
            if self._word_to_index.get(token) else np.zeros(96) for token in tokens)
        indices, scores = self._hnsw.knn_query(query_vector, k=k)
        indices = indices.ravel()
        scores = scores.ravel()
        ids = [self._ids[x.astype(int)] for x in indices]
        return ids, scores