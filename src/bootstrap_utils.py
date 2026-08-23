from __future__ import annotations
import numpy as np

def cluster_bootstrap_indices(patient_ids,seed: int=0):
    ids=np.asarray(patient_ids); unique=np.unique(ids); rng=np.random.default_rng(seed); sampled=rng.choice(unique,size=len(unique),replace=True)
    return [np.flatnonzero(ids==pid) for pid in sampled]

