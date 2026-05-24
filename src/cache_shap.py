"""
Compute SHAP values for the RF model on a 5000-row sample of the test set
and pickle them to data/shap_values.pkl. Run once on a fast machine; the
rest of the team can load the pickle instead of recomputing.

Run: python src/cache_shap.py
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

from split import temporal_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "shap_values.pkl"
N_SAMPLES = 5000
SEED = 42

FCOLS = [
    "site_id", "hour", "day_of_week", "month",
    "is_holiday", "days_to_nearest_holiday", "is_in",
    "temp_c", "humidity", "dewpoint_c", "precip_mm", "rain_mm", "snowfall_cm",
    "windspeed_kmh", "winddir_deg", "windgust_kmh",
    "lat", "long", "road_bearing", "wind_impact",
]


def prep(df):
    out = df.dropna(subset=["cyclist_count"]).copy()
    out["road_bearing"] = out["road_bearing"].fillna(-1)
    out["wind_impact"] = out["wind_impact"].fillna(0)
    return out


def main():
    print("loading panel", flush=True)
    panel = pd.read_parquet(DATA / "panel_merged.parquet")
    train, _, test = temporal_split(panel)
    train_sites = set(train["site_id"].unique())
    test = test[test["site_id"].isin(train_sites)]
    tr, te = prep(train), prep(test)
    X_tr, y_tr = tr[FCOLS], tr["cyclist_count"]
    X_te = te[FCOLS]

    print("training RF", flush=True)
    t0 = time.time()
    model = RandomForestRegressor(
        n_estimators=100, max_depth=20, max_samples=0.3,
        random_state=SEED, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    print(f"computing SHAP values on n={N_SAMPLES} test sample", flush=True)
    sample = X_te.sample(N_SAMPLES, random_state=SEED)
    explainer = shap.TreeExplainer(model)
    t0 = time.time()
    shap_values = explainer.shap_values(sample)
    print(f"  done in {time.time()-t0:.1f}s "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)

    payload = {
        "sample": sample,
        "shap_values": shap_values,
        "expected_value": explainer.expected_value,
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "feature_columns": FCOLS,
    }
    with open(OUT, "wb") as f:
        pickle.dump(payload, f)
    print(f"saved {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
