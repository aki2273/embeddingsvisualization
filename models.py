import torch
from torch import nn


def _get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class FiLM(nn.Module):
    def __init__(self, fin_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(fin_dim, hidden_dim * 2)

    def forward(self, x, fin):
        gamma, beta = self.proj(fin).chunk(2, dim=1)
        return x * (1 + gamma) + beta


class MLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, activation, layernorm, residual):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if layernorm else nn.Identity()
        self.act = _get_activation(activation)
        self.drop = nn.Dropout(dropout)
        self.use_residual = residual
        if residual:
            self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        else:
            self.skip = None

    def forward(self, x, film=None, x_fin=None):
        y = self.linear(x)
        y = self.norm(y)
        if film is not None and x_fin is not None:
            y = film(y, x_fin)
        y = self.act(y)
        y = self.drop(y)
        if self.use_residual:
            y = y + self.skip(x)
        return y


class MLPClassifier(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_classes,
        hidden_dims,
        dropout=0.2,
        activation="gelu",
        layernorm=True,
        residual=False,
        fusion="concat",
        fin_dim=None,
    ):
        super().__init__()
        self.fusion = fusion
        self.fin_dim = fin_dim

        if fusion not in {"concat", "film"}:
            raise ValueError("fusion must be 'concat' or 'film'")
        if fusion == "film" and fin_dim is None:
            raise ValueError("fin_dim is required for FiLM fusion")

        input_dim = embed_dim + (fin_dim or 0) if fusion == "concat" else embed_dim
        dims = [input_dim] + list(hidden_dims)

        self.blocks = nn.ModuleList()
        self.films = nn.ModuleList() if fusion == "film" else None
        for i in range(len(hidden_dims)):
            self.blocks.append(
                MLPBlock(
                    dims[i],
                    dims[i + 1],
                    dropout=dropout,
                    activation=activation,
                    layernorm=layernorm,
                    residual=residual,
                )
            )
            if self.films is not None:
                self.films.append(FiLM(fin_dim, dims[i + 1]))

        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x_embed, x_fin=None, return_features=False):
        if self.fusion == "concat":
            x = torch.cat([x_embed, x_fin], dim=1) if x_fin is not None else x_embed
        else:
            if x_fin is None:
                raise ValueError("x_fin is required for FiLM fusion")
            x = x_embed

        feats = {"input": x_embed}
        for i, block in enumerate(self.blocks):
            film = self.films[i] if self.films is not None else None
            x = block(x, film=film, x_fin=x_fin)
            feats[f"block{i + 1}"] = x
        feats["penultimate"] = x

        logits = self.head(x)
        if return_features:
            feats["logits"] = logits
            return logits, feats
        return logits
