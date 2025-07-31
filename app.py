from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def index():
    return "<p>Hello, World!</p>"

@app.route("/instruct")
def generate():
    text = request.args.get("text")
    return text