from flask import Flask, render_template, send_from_directory
from flask import request, redirect, make_response
import bm25.valuer
import utils.tokenizer
import pickle
from simple_image_download import simple_image_download

# from word2vec.word2vec import Word2Vec

id_to_urls = __import__("pickle").load(open("articles.dump", "rb"))

app = Flask(__name__, subdomain_matching=True)
v = pickle.load(open("valuer.dump", "rb"))


@app.route("/")
def main_page():
    return render_template("main.html")


@app.route("/search-request", methods=["GET"])
def get_results():
    response = simple_image_download

    url = response().urls(request.args['q'] + " imagesize:1280x720", 1)[0]

    res = v.score(request.args['q'])  # TODO

    answer = []
    for id in res:
        if id not in id_to_urls:
            print(f"Error: id {id} not in pickle")
        else:
            answer.append((id, id_to_urls[id][0], id_to_urls[id][1]))

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
    # if "LOCAL" not in __import__("os").environ:
    #     import run_tests
    #
    #     run_tests.run()

    # w2v = Word2Vec(t, v._idf)
    # w2v.load('word2vec/')

    app.run()
