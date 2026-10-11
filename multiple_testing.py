"""Holm adjustment for a complete family of finite p-values."""

import numpy as np


def holm(values):
    values = np.asarray(values,dtype=float)
    assert values.ndim==1 and len(values)>0 and np.isfinite(values).all() and ((values>=0)&(values<=1)).all(), "Holm requires finite, valid p-values for the declared family"
    order = np.argsort(values)
    adjusted = np.minimum(1,np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    result = np.empty(len(values))
    result[order] = adjusted
    return result.tolist()
