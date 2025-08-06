from flask import Flask, request, render_template, url_for, abort, jsonify
from utils import text_to_token_ids, token_ids_to_text, generate, load_weights, SETTINGS 
from model import GPT
import os, tiktoken

app = Flask(__name__)
tokenizer = tiktoken.get_encoding('gpt2')
model = GPT(SETTINGS)
model = load_weights(model, name="custom_124M")



class InvalidAPIUsage(Exception):
    status_code = 400

    def __init__(self, param, status_code=None, payload=None):
        super().__init__()
        self.message = f"{param} must be an integer or a string representing an integer."
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

@app.errorhandler(InvalidAPIUsage)
def invalid_api_usage(e):
    return jsonify(e.to_dict()), e.status_code

def handle_bad_request(value, name):
    if isinstance(value, str):
        if value.isdigit(): # Check if string contains only digits
            value = int(value)
        else:
            raise InvalidAPIUsage(param=name, status_code=400)
    return value

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/instruct")
def instruct():
    text = request.args.get("text", None)
    if text is None:
        abort(400, description="'text' parameter is required.")
    max_num_tokens = handle_bad_request(request.args.get("max_num_tokens", 35), name="max_num_tokens")
    temperature = handle_bad_request(request.args.get("temperature", 1), name="temperature")
    top_k = handle_bad_request(request.args.get("top_k", 5), name="top_k")

    token_ids = generate(model=model, idx=text_to_token_ids(text, tokenizer), 
                         max_num_tokens=max_num_tokens, context_length=SETTINGS['n_ctx'], 
                         temperature=temperature, top_k=top_k, eos=50256)
    output = token_ids_to_text(token_ids, tokenizer)
    resp = {'response': output}
    return jsonify(resp)