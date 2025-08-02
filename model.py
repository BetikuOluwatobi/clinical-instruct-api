import torch

class LayerNorm(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.eps = 10e-6
        self.scale = torch.nn.Parameter(torch.ones(dim))
        self.shift = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, inp):
        mean = inp.mean(dim=-1, keepdim=True)
        var = inp.var(dim=-1, keepdim=True, unbiased=False)
        output = (inp - mean) / torch.sqrt(var + self.eps)
        return output * self.scale + self.shift

class FeedForward(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.GELU(approximate='tanh'),
            torch.nn.Linear(dim * 4, dim)
        )

    def forward(self, inp):
        return self.layer(inp)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim_in, dim_out, n_heads, context_length, 
                 dropout, qkv_bias=False):
        super().__init__()
        self.dim_out = dim_out
        self.W_query = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.n_heads = n_heads
        assert dim_out % n_heads == 0, f"Embedding dimension: {dim_in} must be divisible by n_heads: {n_heads}"

        self.head_dim = dim_out // n_heads
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.output_proj = torch.nn.Linear(dim_out, dim_out)

    def forward(self, inp):
        batch, seq_len, emb_dim = inp.shape
        query = self.W_query(inp)
        key = self.W_key(inp)
        value = self.W_value(inp)

        query = query.view(batch, seq_len, self.n_heads, self.head_dim)
        key = key.view(batch, seq_len, self.n_heads, self.head_dim)
        value = value.view(batch, seq_len, self.n_heads, self.head_dim)

        query = query.transpose(1, 2) # batch, self.n_heads, seq_len, self.head_dim
        key = key.transpose(1, 2) # batch, self.n_heads, seq_len, self.head_dim
        value = value.transpose(1, 2) # batch, self.n_heads, seq_len, self.head_dim

        attn_scores = torch.matmul(query, key.transpose(2,3))
        attn_scores = attn_scores.masked_fill_(self.mask[:seq_len, :seq_len], -torch.inf)
        attn_weights = torch.nn.functional.softmax(attn_scores / key.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vectors = (attn_weights @ value).transpose(1, 2)
        context_vectors = context_vectors.contiguous().view(batch, seq_len, self.dim_out)

        return self.output_proj(context_vectors)

class TransformerBlocks(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNorm(dim=cfg['n_embd'])
        self.ff = FeedForward(dim=cfg['n_embd'])
        self.mha = MultiHeadAttention(dim_in=cfg['n_embd'], dim_out=cfg['n_embd'], n_heads=cfg['n_head'], context_length=cfg['n_ctx'],
                                      dropout=cfg['droprate'], qkv_bias=cfg['qkv_bias'])
        self.dropout = torch.nn.Dropout(cfg['droprate'])
        self.norm2 = LayerNorm(dim=cfg['n_embd'])

    def forward(self, inp):
        shortcut = inp
        inp = self.norm1(inp)
        inp = self.mha(inp)
        inp = self.dropout(inp)
        inp = shortcut + inp

        shortcut = inp
        inp = self.norm2(inp)
        output = self.ff(inp)
        inp = self.dropout(inp)
        return shortcut + output

class GPT(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(cfg["n_vocab"], cfg["n_embd"])
        self.pos_emb = torch.nn.Embedding(cfg["n_ctx"], cfg["n_embd"])

        self.final_norm = LayerNorm(dim=cfg['n_embd'])
        self.trfs_blocks = torch.nn.Sequential(*[
            TransformerBlocks(cfg) for _ in range(cfg['n_layer'])
        ])
        self.dropout = torch.nn.Dropout(cfg['droprate'])
        self.output_proj = torch.nn.Linear(cfg['n_embd'], cfg['n_vocab'], bias=False)

    def forward(self, inp):
        batch, seq_len = inp.shape
        tok_emb = self.tok_emb(inp)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=inp.device))
        inp = tok_emb + pos_emb
        inp = self.dropout(inp)
        inp = self.trfs_blocks(inp)
        inp = self.final_norm(inp)
        output = self.output_proj(inp)
        return output        