import numpy as np
import torch


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

def load_weights(model, path="static"):
    model_state_dict = torch.load(path, weights_only=True)
    model.load_state_dict(model_state_dict)
    return model