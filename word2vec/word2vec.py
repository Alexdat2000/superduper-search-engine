import gensim
import hnswlib
import numpy as np
import tqdm


class Word2Vec:
    def __init__(self):
        self._model = gensim.models.Word2Vec(size=96, min_count=5, max_vocab_size=1000000)
        self._hnsw = 0  # how can i specify the type?
        self._document_vectors = np.ndarray(shape=(1, 96))
        self._word_to_index = dict()
        self._index_to_word = dict()

    def build_model(self, tokenizer):
        self._model.build_vocab(tokenizer.token_generator(), progress_per=1000)

    def train(self, tokenizer, epochs):
        print("training model ({} epochs)...".format(epochs))
        items_count = sum([1 for _ in tokenizer.generator_from_msgpack()])
        self._model.train(tokenizer.token_generator(epochs),
                          total_examples=items_count * epochs,
                          epochs=1, report_delay=10.)

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
        for batch_start in tqdm.tqdm_notebook(range(0, count, batch_size)):
            self._hnsw.add_items(embeddings[batch_start: batch_start + batch_size, :],
                        data_labels[batch_start: batch_start + batch_size])

        self._hnsw.set_ef(300)  # ef should always be > k
        self._hnsw.save_index(output_path)

    def load_knn_index(self, dimension, file_path, space='ip'):
        assert space in ('ip', 'cosine')

        self._hnsw = hnswlib.Index(space=space, dim=dimension)  # possible options are l2, cosine or ip

        self._hnsw.load_index(file_path)
        self._hnsw.set_ef(300)

    def build_and_train(self, tokenizer, epochs=1):
        print("building model...")
        self.build_model(tokenizer)
        self.train(tokenizer, epochs)
        self._word_to_index = {w: data.index for w, data in self._model.wv.vocab.items()}
        self._index_to_word = {data.index: w for w, data in self._model.wv.vocab.items()}
        print('preparing hnsw...')
        self.init_document_vectors(tokenizer.generator_from_msgpack())
        self.build_hnsw()
        # save_all_data()

    def init_document_vectors(self, documents):
        document_list = [sum(self._word_to_index[word] for word in document) for document in documents]
        self._document_vectors = np.array(document_list)

    #def save_all_data():
    # save document_vectors, model, hnsw, word_to_index, index_to_word

    def evaluate(self, text, tokenizer, k):
        query_vector = sum(self._word_to_index[word] for word in tokenizer.tokenize(text))
        indices, scores = self._hnsw.knn_query(query_vector, k=k)
        return indices, scores
