"""Reproducible meal-event preprocessing pipeline for AahaarTwin (Phase 1).

Reads the heterogeneous CGMacros participant CSVs, normalizes schemas,
builds one meal-event table with pre-meal features and post-meal labels,
and writes it to data/processed/meals.csv.

Deterministic: no randomness; identical input -> identical output.
"""
import glob
import os

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)


def load_config(path=None):
    path = path or os.path.join(PROJ, "configs", "phase1.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def participant_files(raw_dir):
    return sorted(glob.glob(os.path.join(raw_dir, "CGMacros-*", "CGMacros-*.csv")))


def normalize_columns(df):
    """Strip whitespace; unify known variant spellings."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    rename = {
        "Image path": "image_path",
        "Amount Consumed": "amount_consumed",
        "Calories (Activity)": "activity_kcal",
        "Meal Type": "meal_type",
        "Dexcom GL": "dexcom_gl",
        "Libre GL": "libre_gl",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def normalize_meal_type(s, mapping):
    if pd.isna(s):
        return np.nan
    key = str(s).strip().lower()
    return mapping.get(key, key)


def build_meal_table(cfg):
    """Build the meal-event table. Returns (meals_df, report_dict)."""
    raw_dir = os.path.normpath(os.path.join(PROJ, cfg["data"]["raw_dir"]))
    pre = cfg["preprocessing"]
    mt_map = cfg["meal_types"]["mapping"]
    report = {}

    files = participant_files(raw_dir)
    report["participant_csvs_found"] = len(files)
    if not files:
        raise FileNotFoundError(f"no participant CSVs under {raw_dir}")

    schema_variants = {}
    rows = []
    for f in files:
        pid = int(os.path.basename(os.path.dirname(f)).split("-")[-1])
        df = pd.read_csv(f, low_memory=False)
        schema_variants.setdefault(tuple(sorted(df.columns)), []).append(pid)
        df = normalize_columns(df)
        df["ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.sort_values("ts").reset_index(drop=True)

        g = df["dexcom_gl"].astype(float) if "dexcom_gl" in df.columns else pd.Series(np.nan, index=df.index)
        hr = df["HR"].astype(float) if "HR" in df.columns else pd.Series(np.nan, index=df.index)
        act = (df["activity_kcal"].astype(float) if "activity_kcal" in df.columns
               else pd.Series(np.nan, index=df.index))

        mrows = df[df["meal_type"].notna()]
        for idx, mr in mrows.iterrows():
            i = df.index.get_loc(idx)
            t0 = df.at[idx, "ts"]

            pre_win = g.iloc[max(0, i - pre["pre_glucose_lookback_min"]): i + 1].dropna()
            g0 = float(pre_win.iloc[-1]) if len(pre_win) else np.nan

            slope_win = g.iloc[max(0, i - pre["pre_slope_window_min"]): i + 1].dropna()
            slope = (float(np.polyfit(np.arange(len(slope_win)), slope_win, 1)[0])
                            if len(slope_win) >= 12 else np.nan)

            ctx = pre["pre_context_window_min"]
            hr_win = hr.iloc[max(0, i - ctx): i + 1].dropna()
            act_win = act.iloc[max(0, i - ctx): i + 1].dropna()

            w = pre["iauc_window_min"]
            post = g.iloc[i: i + w + 1]
            cov = float(post.notna().mean())
            peak = float(post.max()) if cov >= pre["min_post_coverage_peak"] else np.nan
            if cov >= pre["min_post_coverage_iauc"]:
                gg = post.interpolate(limit_direction="both").to_numpy()
                iauc = float(np.trapz(np.maximum(gg - g0, 0), dx=1) / 60.0)
            else:
                iauc = np.nan

            rows.append(dict(
                pid=pid, ts=t0,
                meal_type=normalize_meal_type(mr["meal_type"], mt_map),
                calories=mr.get("Calories", np.nan),
                carbs=mr.get("Carbs", np.nan),
                protein=mr.get("Protein", np.nan),
                fat=mr.get("Fat", np.nan),
                fiber=mr.get("Fiber", np.nan),
                amount_consumed=(float(mr["amount_consumed"])
                                 if "amount_consumed" in mr.index and pd.notna(mr["amount_consumed"])
                                 else np.nan),
                g0=g0, slope_60min=slope,
                hr_mean_2h=float(hr_win.mean()) if len(hr_win) else np.nan,
                act_kcal_2h=float(act_win.sum()) if len(act_win) else np.nan,
                post_cov_2h=cov, iauc_2h=iauc, peak_2h=peak,
            ))

    meals = pd.DataFrame(rows).sort_values(["pid", "ts"]).reset_index(drop=True)
    report["meal_events_raw"] = len(meals)
    report["n_schema_variants"] = len(schema_variants)
    report["schema_variant_file_counts"] = {
        (", ".join(c for c in cols if c not in ("Timestamp",))[:140]): len(pids)
        for cols, pids in schema_variants.items()
    }

    # Deduplicate: drop meal rows < dedup_min_gap_min after the previous kept meal (same patient).
    gap = pre["dedup_min_gap_min"]
    keep = []
    last_kept = {}
    dropped = 0
    for _, r in meals.iterrows():
        p, t = r["pid"], r["ts"]
        if p in last_kept and (t - last_kept[p]).total_seconds() / 60 < gap:
            dropped += 1
            continue
        keep.append(r.name)
        last_kept[p] = t
    meals = meals.loc[keep].reset_index(drop=True)
    report["dedup_dropped_rows"] = dropped
    report["meal_events_after_dedup"] = len(meals)

    # Labels
    thr = 180.0
    meals["excursion_2h"] = (meals["peak_2h"] >= thr).astype("Int64")

    # Usable subset: valid iAUC + macros + pre-meal glucose
    usable = meals["iauc_2h"].notna() & meals["carbs"].notna() & meals["g0"].notna()
    meals["usable"] = usable
    report["usable_meals"] = int(usable.sum())
    upp = meals[usable].groupby("pid").size()
    report["usable_meals_per_patient"] = {
        "min": int(upp.min()), "median": float(upp.median()),
        "mean": float(upp.mean()), "max": int(upp.max()),
        "n_patients": int(upp.shape[0]),
        "patients_with_ge10": int((upp >= 10).sum()),
        "patients_with_ge20": int((upp >= 20).sum()),
    }
    exc = meals[usable & meals["peak_2h"].notna()]
    report["excursion_rate_peak_ge_180"] = float((exc["peak_2h"] >= thr).mean())
    report["missingness"] = {
        "g0": float(meals["g0"].isna().mean()),
        "hr_mean_2h": float(meals["hr_mean_2h"].isna().mean()),
        "act_kcal_2h": float(meals["act_kcal_2h"].isna().mean()),
        "iauc_2h": float(meals["iauc_2h"].isna().mean()),
        "peak_2h": float(meals["peak_2h"].isna().mean()),
        "carbs": float(meals["carbs"].isna().mean()),
    }
    report["iauc_2h_stats"] = {k: float(v) for k, v in
                               meals.loc[usable, "iauc_2h"].describe().items()}
    report["peak_2h_stats"] = {k: float(v) for k, v in
                               meals.loc[usable & meals["peak_2h"].notna(), "peak_2h"].describe().items()}
    report["carbs_stats"] = {k: float(v) for k, v in
                             meals.loc[usable, "carbs"].describe().items()}
    return meals, report


def main():
    cfg = load_config()
    meals, report = build_meal_table(cfg)
    out_dir = os.path.normpath(os.path.join(PROJ, cfg["data"]["processed_dir"]))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "meals.csv")
    meals.to_csv(out, index=False)
    print(f"wrote {out} ({len(meals)} rows)")
    for k, v in report.items():
        print(f"  {k}: {v if not isinstance(v, dict) else json_compact(v)}")
    return meals, report


def json_compact(d):
    import json as _json
    return _json.dumps(d, default=str)[:220]


if __name__ == "__main__":
    main()
