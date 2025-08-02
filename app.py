from flask import Flask, request, render_template, url_for
from markupsafe import escape
from utils import text_to_token_ids, token_ids_to_text, generate, load_weights, SETTINGS 
from model import GPT
import os, tiktoken

app = Flask(__name__)
tokenizer = tiktoken.get_encoding('gpt2')
model = GPT(SETTINGS)
model = load_weights(model, name="custom_124M")

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/instruct")
def instruct():
    text = request.args.get("text")
    token_ids = generate(model=model, idx=text_to_token_ids(text, tokenizer), 
                         max_num_tokens=35, context_length=SETTINGS['n_ctx'], 
                         temperature=.5, top_k=5, eos=50256)
    output = token_ids_to_text(token_ids, tokenizer)
    return f"{output}"