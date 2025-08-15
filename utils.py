import numpy as np
import os, torch


CONFIG = {0:{'droprate': 0.0, 'qkv_bias': True, 'n_vocab': 50257, 'n_ctx': 1024, 
             'n_embd': 768, 'n_head': 12, 'n_layer': 12},
          1: {'droprate': 0.0,'qkv_bias': True,'n_vocab': 50257,'n_ctx': 1024,
              'n_embd': 1024,'n_head': 16,'n_layer': 24}
}


def format_input(prompt, competency):
    output = (
        f"Below is an instruction that describes a clinical situation with a specific question, paired with the nursing competency and an optional SNOMED CT diagnostic codes relevant to the scenario to provide further context. "
        f"Write ONLY a clinician response that appropriately completes the request.\n\n"
        f"### Instruction:\n{prompt}"
    )
    output += (f"\n\n### Nursing Competency:\n{competency}" if competency else "")
    return output

def text_to_token_ids(text, tokenizer):
    token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(token_ids).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    text = tokenizer.decode(list(token_ids.squeeze()))
    return text

def generate(model, idx, max_num_tokens, context_length, temperature, top_k, eos=50256):
    model.eval()
    for _ in range(max_num_tokens):
        idx_cond = idx[:, -context_length:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        if temperature > 0:
            logits = logits / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
        if idx_next == eos:
            break
            
        idx = torch.cat([idx, idx_next], dim=-1)
    return idx

def load_weights(model, name, device, base_dir="static/weights"):
    PATH = os.path.join(base_dir, f"{name}_model_weights.pth")
    model_state_dict = torch.load(PATH, weights_only=True, map_location=device)
    model.load_state_dict(model_state_dict)
    return model