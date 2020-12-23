from flask import Flask, render_template, send_from_directory
from flask import request, redirect, make_response
import bm25.valuer
import utils.tokenizer
from word2vec.word2vec import Word2Vec
import numpy as np
from simple_image_download import simple_image_download
import pickle

id_to_urls = __import__("pickle").load(open("articles.dump", "rb"))

app = Flask(__name__, subdomain_matching=True)
t = utils.tokenizer.Tokenizer('samples/search_items.msgpack')
v = pickle.load(open("valuer.dump", "rb"))
w2v = Word2Vec(t, v._idf)
w2v.load('word2vec/', True)


def get_and_merge_results(query='котик'):
    res_len, best_len = 100, 10
    w2v_res = w2v.evaluate(query, res_len)
    bm25_res = v.score(query)
    print(w2v_res)
    print(bm25_res)

    id_set = set()
    for x in w2v_res[0]:
        id_set.add(x)
    for x in bm25_res:
        id_set.add(x)

    ids = [x for x in id_set]
    if len(ids) == 0:
        return []

    scores = np.array(w2v.get_scores(query, ids))
    bm25_scores = np.array(v.score(query, ids))
    max_score = np.max(bm25_res)
    if max_score != 0:
        bm25_scores /= max_score
    scores += bm25_scores * 30
    print(scores)
    return [ids[x] for x in np.argsort(scores)[-best_len:]]


@app.route("/")
def main_page():
    return render_template("main.html")


@app.route("/search-request", methods=["GET"])
def get_results():
    response = simple_image_download
    url = response().urls(request.args['q'] + " imagesize:1280x720", 1)[0]

    res = get_and_merge_results(request.args['q'])
    res.reverse()

    if not res:
        return render_template("no_res.html",
                               re=request.args["q"]
                               )

    else:
        return render_template(
            "results.html",
            background=url,
            results=[{
                "title": id_to_urls[i][0],
                "url": "https://zen.yandex.ru" + id_to_urls[i][1],
                "preview": id_to_urls[i][2] + "..."
            }
                for i in res
            ],
            re=request.args["q"]
        )


@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')


if __name__ == '__main__':
    app.run()
