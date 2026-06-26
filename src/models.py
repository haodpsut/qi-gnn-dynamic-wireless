"""
A single GNN architecture, three pluggable propagation operators.

GCN / Heat / QI (quantum-inspired) differ ONLY in the operator Phi; they share
depth, hidden width, and weight shapes, so any performance gap is attributable to
the operator, not capacity. Heat and QI build Phi from a learnable scalar t_l
through the Laplacian eigenvalues, so training is plain backprop.
"""

import numpy as np
import torch
import torch.nn as nn


def build_propagator(op_kind, w, U, t, gcn_phi):
    """Return Phi (N x N torch tensor) for the chosen operator.

    op_kind: 'gcn' | 'heat' | 'qw'
    w, U:    Laplacian eigenpairs as torch tensors (U real)
    t:       learnable scalar (torch) for heat/qw, ignored for gcn
    gcn_phi: precomputed symmetric-normalized adjacency (torch), used for gcn
    """
    if op_kind == "gcn":
        return gcn_phi
    if op_kind == "heat":
        # exp(-t L) = U diag(exp(-t w)) U^T
        scale = torch.exp(-torch.relu(t) * w)            # (N,)
        return (U * scale) @ U.t()
    if op_kind == "qw":
        # |U diag(exp(-i t w)) U^T|^2  -> quantum-walk transition probability
        ang = torch.relu(t) * w                          # (N,)
        cos, sin = torch.cos(ang), torch.sin(ang)
        re = (U * cos) @ U.t()
        im = (U * (-sin)) @ U.t()
        return re * re + im * im
    raise ValueError(op_kind)


class PropGNN(nn.Module):
    """K-layer message passing with a pluggable propagation operator."""

    def __init__(self, in_dim=2, hidden=16, out_dim=1, n_layers=2, op_kind="gcn"):
        super().__init__()
        self.op_kind = op_kind
        self.n_layers = n_layers
        dims = [in_dim] + [hidden] * (n_layers - 1) + [hidden]
        self.lins = nn.ModuleList(
            [nn.Linear(dims[l], dims[l + 1]) for l in range(n_layers)]
        )
        self.readout = nn.Linear(hidden, out_dim)
        # one learnable propagation time per layer (heat / qw only)
        self.t = nn.Parameter(torch.ones(n_layers) * 0.5)
        self.act = nn.ReLU()

    def forward(self, X, w, U, gcn_phi):
        H = X
        for l in range(self.n_layers):
            Phi = build_propagator(self.op_kind, w, U, self.t[l], gcn_phi)
            H = self.act(self.lins[l](Phi @ H))
        return self.readout(H).squeeze(-1)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
