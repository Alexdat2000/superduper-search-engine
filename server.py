from flask import Flask, render_template, send_from_directory
from flask import request, redirect, make_response
import run_tests

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
    run_tests.run()
    app.run()
