import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_slopes_power_of_2(n_heads: int, device=None, dtype=None):
    start = 2 ** (-(2 ** -(math.log2(n_heads) - 3)))
    ratio = start
    slopes = [start * ratio**i for i in range(n_heads)]
    return torch.tensor(slopes, device=device, dtype=dtype)


def get_alibi_slopes(n_heads: int, device=None, dtype=None):
    if math.log2(n_heads).is_integer():
        return _get_slopes_power_of_2(n_heads, device=device, dtype=dtype)
    closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
    slopes = _get_slopes_power_of_2(closest_power_of_2, device=device, dtype=dtype)
    extra = torch.tensor(
        _get_slopes_power_of_2(
            2 * closest_power_of_2, device=device, dtype=dtype
        ).tolist()[0::2],
        device=device,
        dtype=dtype,
    )
    return torch.cat([slopes, extra[: n_heads - closest_power_of_2]], dim=0)


def build_alibi_bias(
    slopes: torch.Tensor, q_len: int, k_len: int, device=None, dtype=None
):
    q_pos = torch.arange(q_len, device=device, dtype=dtype)  # [q_len]
    k_pos = torch.arange(k_len, device=device, dtype=dtype)  # [k_len]
    pos_diff = k_pos[None, :] - q_pos[:, None]  # [q_len, k_len]
    bias = slopes[:, None, None] * pos_diff[None, :, :]  # [h, q_len, k_len]
    return bias.unsqueeze(0)  # [1, h, q_len, k_len]


class AlibiMultiheadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

        # Slopes are fixed constants; register as buffer for device/dtype tracking
        slopes = get_alibi_slopes(n_heads, device=None, dtype=torch.float32)
        self.register_buffer("alibi_slopes", slopes, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.size()

        q = (
            self.q_proj(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        )  # [b, h, q, d]
        k = (
            self.k_proj(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        )  # [b, h, k, d]
        v = (
            self.v_proj(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        )  # [b, h, k, d]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.d_head
        )  # [b, h, q, k]

        # ALiBi bias
        alibi = build_alibi_bias(
            self.alibi_slopes.to(device=attn_scores.device, dtype=attn_scores.dtype),
            q_len=seq_len,
            k_len=seq_len,
            device=attn_scores.device,
            dtype=attn_scores.dtype,
        )  # [1, h, q, k]
        attn_scores = attn_scores + alibi

        # Causal mask (upper triangle masked)
        causal_mask = torch.ones(
            (seq_len, seq_len), device=attn_scores.device, dtype=torch.bool
        ).tril()
        attn_scores = attn_scores.masked_fill(
            ~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        y = torch.matmul(attn_weights, v)  # [b, h, q, d]
        y = (
            y.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        )  # [b, q, d_model]
        y = self.o_proj(y)
        return self.resid_dropout(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = d_model * hidden_mult
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class AlibiBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = AlibiMultiheadAttention(
            d_model, n_heads, attn_dropout=attn_dropout, resid_dropout=resid_dropout
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, hidden_mult=4, dropout=mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class AlibiTransformer(nn.Module):
    def __init__(
        self,
        d_vocab: int,
        d_model: int,
        n_heads: int,
        layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model

        self.embedding = nn.Embedding(d_vocab, d_model)
        self.blocks = nn.ModuleList(
            [
                AlibiBlock(
                    d_model,
                    n_heads,
                    attn_dropout=dropout,
                    resid_dropout=dropout,
                    mlp_dropout=dropout,
                )
                for _ in range(layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, d_vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embedding(x)  # [b, t, d]
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.lm_head(h)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
