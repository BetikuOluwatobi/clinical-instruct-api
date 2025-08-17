from flask import Flask, request, render_template, url_for, abort, jsonify
from utils import format_input, text_to_token_ids, token_ids_to_text, generate, load_weights, CONFIG 
from model import GPT
import os, tiktoken, torch

app = Flask(__name__)
torch.random.manual_seed(96)
tokenizer = tiktoken.get_encoding('gpt2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SETTINGS = CONFIG[1]
model = GPT(SETTINGS)
model.to(device)
name = "finetuned_355M_nurse_competency_ddx_snomed"
model = load_weights(model, name=name, device=device)



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
    prompt = request.args.get("prompt", None)
    if prompt is None:
        abort(400, description="'prompt' parameter is required.")
    competency = request.args.get("nursing_competency", None)
    max_num_tokens = handle_bad_request(request.args.get("max_num_tokens", 35), name="max_num_tokens")
    temperature = handle_bad_request(request.args.get("temperature", 1.), name="temperature")
    top_k = handle_bad_request(request.args.get("top_k", 5), name="top_k")

    token_ids = generate(model=model, idx=text_to_token_ids(format_input(prompt, competency), tokenizer), 
                         max_num_tokens=max_num_tokens, context_length=SETTINGS['n_ctx'], 
                         temperature=temperature, top_k=top_k, eos=50256)
    output = token_ids_to_text(token_ids, tokenizer)
    resp = {'response': output}
    return jsonify(resp)