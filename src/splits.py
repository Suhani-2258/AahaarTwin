"""Leakage-safe evaluation splits for AahaarTwin (Phase 1).

Two split strategies, both leakage-free by construction:

1. time_ordered_within_patient: for each patient with >= min_meals usable meals,
   the first `train_fraction` of meals in chronological order form the reference
   set and the rest form the test set. No shuffling; no future meal of a patient
   ever appears in that patient's reference set.

2. leave_one_patient_out: for population-model evaluation. Each patient in turn
   is the test patient; the model trains on all other patients. Test patient
   IDs never appear in the training data.
"""
import numpy as np
import pandas as pd


def time_ordered_within_patient(df, min_meals=10, train_fraction=0.5):
    """Return {pid: (train_index, test_index)} using integer positions into df."""
    df = df.sort_values(["pid", "ts"]).reset_index(drop=True)
    splits = {}
    for pid, gdf in df.groupby("pid"):
        if len(gdf) < min_meals:
            continue
        k = max(1, int(len(gdf) * train_fraction))
        idx = gdf.index.to_numpy()
        splits[int(pid)] = (idx[:k], idx[k:])
    return splits


def leave_one_patient_out(df):
    """Return list of (train_index, test_index) with whole patients held out."""
    df = df.sort_values(["pid", "ts"]).reset_index(drop=True)
    pids = sorted(df["pid"].unique().tolist())
    folds = []
    for pid in pids:
        te = df.index[df["pid"] == pid].to_numpy()
        tr = df.index[df["pid"] != pid].to_numpy()
        folds.append((tr, te))
    return folds


def assert_no_leakage_time_ordered(df, splits):
    """Every test meal timestamp must be strictly after every train meal timestamp."""
    df = df.sort_values(["pid", "ts"]).reset_index(drop=True)
    for pid, (tr, te) in splits.items():
        assert len(tr) > 0 and len(te) > 0, f"empty split for patient {pid}"
        t_tr_max = df.loc[tr, "ts"].max()
        t_te_min = df.loc[te, "ts"].min()
        assert t_te_min > t_tr_max, f"time leakage for patient {pid}"
        assert set(tr).isdisjoint(set(te)), f"index overlap for patient {pid}"


def assert_no_leakage_lopo(df, folds):
    """No patient may appear in both train and test of the same fold."""
    for tr, te in folds:
        p_tr = set(df.loc[tr, "pid"].unique())
        p_te = set(df.loc[te, "pid"].unique())
        assert p_tr.isdisjoint(p_te), "patient leakage in LOPO fold"
