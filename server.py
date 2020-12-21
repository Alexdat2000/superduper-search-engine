from flask import Flask, render_template, send_from_directory
from flask import request, redirect, make_response
import bm25.valuer
import utils.tokenizer
# from word2vec.word2vec import Word2Vec

id_to_urls = __import__("pickle").load(open("articles.dump", "rb"))

app = Flask(__name__, subdomain_matching=True)
v = None
t = None


@app.route("/")
def main_page():
    return render_template("main.html")


@app.route("/search-request", methods=["GET"])
def get_results():
    res = v.score(request.args['q'])  # TODO

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
    global v, t, w2v
    # if "LOCAL" not in __import__("os").environ:
    #     import run_tests
    #
    #     run_tests.run()

    t = utils.tokenizer.Tokenizer('samples/search_items_sample.msgpack')
    v = bm25.valuer.Valuer(t)
    # w2v = Word2Vec(t, v._idf)
    # w2v.load('word2vec/')

    app.run()
