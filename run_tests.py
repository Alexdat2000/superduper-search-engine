from word2vec.word2vec import Word2Vec
from utils.tokenizer import Tokenizer


def run():
    # test for word2vec
    tokenizer = Tokenizer(u"samples/search_items_sample.msgpack")
    word2vec = Word2Vec()
    #word2vec.build_and_train(tokenizer)
    model_path = 'https://drive.google.com/file/d/1Gsa99WsmFFMj_RKG14LBzhve-rsva-LE/view?usp=sharing'
    word2vec.load_from_url(model_path)
    #print(word2vec.evaluate('котик', tokenizer, 10))
