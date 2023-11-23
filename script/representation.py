import torch
from torch import nn


def main():
    num_input_features = 12
    num_config_features = 12
    emb_size = 32
    batch_size = 32

    input_emb = nn.Sequential(
        nn.Linear(num_input_features, 64), nn.ReLU(), nn.Linear(64, emb_size)
    )
    config_emb = nn.Sequential(
        nn.Linear(num_config_features, 64), nn.ReLU(), nn.Linear(64, emb_size)
    )

    optimizer = torch.optim.AdamW(
        list(input_emb.parameters()) + list(config_emb.parameters()), lr=0.001
    )

    optimizer.zero_grad()
    a, p, n = None, None, None
    loss = nn.functional.triplet_margin_loss(anchor=a, positive=p, negative=n)
    loss.backward()
    optimizer.step()
