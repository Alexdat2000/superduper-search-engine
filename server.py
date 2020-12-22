from flask import Flask, render_template, send_from_directory
from flask import request, redirect, make_response
import bm25.valuer
import utils.tokenizer
from word2vec.word2vec import Word2Vec
import numpy as np

id_to_urls = __import__("pickle").load(open("articles.dump", "rb"))

app = Flask(__name__, subdomain_matching=True)
t = utils.tokenizer.Tokenizer('samples/search_items_sample.msgpack')
v = bm25.valuer.Valuer(t)
w2v = Word2Vec(t, v._idf)
w2v.load('word2vec/', False)
w2v.build_hnsw()

def get_and_merge_results(query='котик'):
    res_len, best_len = 100, 10
    w2v_res = w2v.evaluate(query, res_len)
    bm25_res = v.score(query)
    id_set = set()
    for x in w2v_res[0]:
        id_set.add(x)
    for x in bm25_res:
        id_set.add(x)

    ids = [x for x in id_set]
    scores = np.array(w2v.get_scores(query, ids))
    # scores += np.array(v.get_scores(query, ids))
    return [ids[x] for x in np.argsort(scores)[-best_len:]]


@app.route("/")
def main_page():
    return render_template("main.html")


@app.route("/search-request", methods=["GET"])
def get_results():
    # res = w2v.evaluate(request.args['q'], 10)[0]  # TODO
    # res = v.score(request.args['q'])
    res = get_and_merge_results(request.args['q'])
    res.reverse()

    answer = []
    for id in res:
        if id not in id_to_urls:
            print(f"Error: id {id} not in pickle")
        else:
            answer.append((id, id_to_urls[id][0], id_to_urls[id][1]))

    return render_template(
        "results.html",
        results=[{
            "title": id_to_urls[i][0],
            "url": "https://zen.yandex.ru" + id_to_urls[i][1],
            "preview": id_to_urls[i][2] + "..."
        }
            for i in res
        ]
    )


@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')


if __name__ == '__main__':
    # if "LOCAL" not in __import__("os").environ:
    #     import run_tests
    #
    #     run_tests.run()

    # w2v = Word2Vec(t, v._idf)
    # w2v.load('word2vec/')

    app.run()
