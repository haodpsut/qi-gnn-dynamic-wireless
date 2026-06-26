"""Training / evaluation loop shared by all operators (seeded, reproducible)."""

import numpy as np
import torch

from .models import PropGNN, n_params
from .metrics import mae, far_node_aurc


def _tensors(sample, device, eig_key="L"):
    """eig_key selects which eigenbasis to feed: 'L' (Laplacian, default),
    'A' (adjacency), or 'S' (symmetric-normalized Laplacian) for Exp B2."""
    if eig_key == "A":
        sample.ensure_variant_eigs()
        wv, Uv = sample.w_A, sample.U_A
    elif eig_key == "S":
        sample.ensure_variant_eigs()
        wv, Uv = sample.w_S, sample.U_S
    else:
        wv, Uv = sample.w, sample.U
    X = torch.tensor(sample.X, dtype=torch.float32, device=device)
    w = torch.tensor(wv, dtype=torch.float32, device=device)
    U = torch.tensor(Uv, dtype=torch.float32, device=device)
    y = torch.tensor(sample.y, dtype=torch.float32, device=device)
    from .operators import gcn_propagator
    gphi = torch.tensor(gcn_propagator(sample.A), dtype=torch.float32, device=device)
    return X, w, U, y, gphi


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_eval(op_kind, train_samples, eval_samples, hidden=16, n_layers=3,
               epochs=200, lr=0.01, seed=0, device="cpu", verbose=False,
               eig_key="L", n_times=1):
    """Train one operator on train_samples, evaluate on eval_samples.

    eig_key picks the eigenbasis ('L'/'A'/'S'); n_times>1 enables the multi-time
    readout variant. Returns dict with mae, aurc, and param count.
    """
    set_seed(seed)
    model = PropGNN(in_dim=2, hidden=hidden, out_dim=1,
                    n_layers=n_layers, op_kind=op_kind, n_times=n_times).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    cache = [_tensors(s, device, eig_key=eig_key) for s in train_samples]
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = 0.0
        for (X, w, U, y, gphi) in cache:
            pred = model(X, w, U, gphi)
            loss = loss + loss_fn(pred, y)
        loss = loss / len(cache)
        loss.backward()
        opt.step()
        if verbose and ep % 50 == 0:
            print(f"  [{op_kind}] ep{ep} loss={loss.item():.4f}")

    model.eval()
    maes, aurcs = [], []
    with torch.no_grad():
        for s in eval_samples:
            X, w, U, y, gphi = _tensors(s, device, eig_key=eig_key)
            pred = model(X, w, U, gphi).cpu().numpy()
            maes.append(mae(pred, s.y))
            aurcs.append(far_node_aurc(pred, s.y, s.hops, s.ecc))
    return {
        "op": op_kind,
        "mae": float(np.mean(maes)),
        "aurc": float(np.nanmean(aurcs)),
        "params": n_params(model),
    }
