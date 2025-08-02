from flask import Flask, request, render_template, url_for
from markupsafe import escape
from utils import text_to_token_ids, token_ids_to_text, generate, load_weights 
from model import GPT
import tiktoken

app = Flask(__name__)
tokenizer = tiktoken.get_encoding('gpt2')

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/instruct")
def generate():
    text = request.args.get("text")
    tokens = text_to_token_ids(escape(text), tokenizer)
    output = token_ids_to_text(tokens, tokenizer)
    return f"{output}"