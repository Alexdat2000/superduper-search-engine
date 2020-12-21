from word2vec.word2vec import Word2Vec
from utils.tokenizer import Tokenizer
import bm25.valuer


def train():
    word2vec = Word2Vec(tokenizer=t, idf=v._idf)
    word2vec.build_and_train(10)
    word2vec.save('word2vec/')


def run():
    # test for word2vec
    word2vec = Word2Vec(tokenizer=t, idf=v._idf)
    word2vec.load('word2vec')
    return word2vec.evaluate('котик', 10)

if __name__ == '__main__':
    global v, t
    print('test')
    t = Tokenizer(u'samples/search_items_sample.msgpack')
    v = bm25.valuer.Valuer(t)
    train()