from typing import List

import torch
import torch.nn as nn


class All(nn.Module):
    def forward(self, node_embeddings: List[torch.Tensor]):
        return torch.cat(node_embeddings, dim=-1)


class MLP(nn.Module):
    def __init__(self, emb_size: int, nlayers: int, output_size: int):
        super().__init__()
        match_emb_size = (nlayers + 1) * emb_size
        self.mlp = nn.Sequential(
            nn.Linear(match_emb_size, match_emb_size, bias=True),
            nn.LeakyReLU(),
            nn.Linear(match_emb_size, output_size, bias=True),
        )

    def forward(self, node_embeddings: List[torch.Tensor]):
        node_emb_cat = torch.cat(node_embeddings, dim=-1)
        return self.mlp(node_emb_cat)
