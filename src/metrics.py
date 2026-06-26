"""Cherry-pick-proof metrics: MAE, far-node AURC, and mean +/- std helpers."""

import numpy as np


def mae(y_pred, y_true, mask=None):
    y_pred = np.asarray(y_pred, float)
    y_true = np.asarray(y_true, float)
    if mask is not None:
        y_pred, y_true = y_pred[mask], y_true[mask]
    return float(np.mean(np.abs(y_pred - y_true)))


def far_node_aurc(y_pred, y_true, hops, ecc, n_bins=10):
    """Area under the error-vs-difficulty curve.

    Difficulty axis rho in [0,1]; bin b keeps nodes with hops >= rho*ecc and
    reports their MAE. Integrating over rho gives one scale-free number that is
    robust to picking a flattering far-node threshold.
    """
    y_pred = np.asarray(y_pred, float)
    y_true = np.asarray(y_true, float)
    hops = np.asarray(hops, float)
    rhos = np.linspace(0.0, 1.0, n_bins)
    errs = []
    for rho in rhos:
        m = hops >= rho * ecc
        if m.sum() == 0:
            errs.append(np.nan)
        else:
            errs.append(np.mean(np.abs(y_pred[m] - y_true[m])))
    errs = np.array(errs)
    valid = ~np.isnan(errs)
    return float(np.trapezoid(errs[valid], rhos[valid]))


def mean_std(values):
    v = np.asarray(values, float)
    return float(v.mean()), float(v.std())
