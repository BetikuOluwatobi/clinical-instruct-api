from flask import Flask, request, render_template, url_for
from markupsafe import escape

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/instruct")
def generate():
    text = request.args.get("text")
    text = escape(text)
    return f"{text}"