from flask import Flask, render_template, send_from_directory
from flask import request, redirect, make_response
from word2vec.word2vec import Word2Vec
from utils.tokenizer import Tokenizer


app = Flask(__name__, subdomain_matching=True)


@app.route("/")
def main_page():
    return render_template("main.html")


@app.route("/search-request", methods=["GET"])
def get_results():
    # обработать запрос request.args["q"]
    return render_template("results.html")


@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')


if __name__ == '__main__':
    app.run()
    # test for word2vec
    # tokenizer = Tokenizer(u"samples/search_items_sample.msgpack")
    # word2vec = Word2Vec()
    # word2vec.build_and_train(tokenizer, tokenizer.token_generator())

