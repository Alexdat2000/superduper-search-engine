from flask import Flask, render_template
from flask import request, redirect, make_response


app = Flask(__name__, subdomain_matching=True)


@app.route("/")
def main_page():
    return render_template("main.html")


if __name__ == '__main__':
    app.run()
