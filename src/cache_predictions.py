"""
Train the RF predictive model and save its predictions on train/val/test
to data/predictions.parquet so the dashboard can load them without
retraining.

Run: python src/cache_predictions.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from split import temporal_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "predictions.parquet"
SEED = 42

FCOLS = [
    "site_id", "hour", "day_of_week", "month",
    "is_holiday", "days_to_nearest_holiday", "is_in",
    "temp_c", "humidity", "dewpoint_c", "precip_mm", "rain_mm", "snowfall_cm",
    "windspeed_kmh", "winddir_deg", "windgust_kmh",
    "lat", "long", "road_bearing", "wind_impact",
]

KEEP_COLS = [
    "site_id", "richting", "datetime", "naam", "lat", "long",
    "year", "hour", "day_of_week", "is_holiday", "is_in",
    "temp_c", "humidity", "precip_mm", "windspeed_kmh",
    "cyclist_count",
]


def prep(df):
    out = df.dropna(subset=["cyclist_count"]).copy()
    out["road_bearing"] = out["road_bearing"].fillna(-1)
    out["wind_impact"] = out["wind_impact"].fillna(0)
    return out


def main():
    print("loading panel", flush=True)
    panel = pd.read_parquet(DATA / "panel_merged.parquet")
    train, val, test = temporal_split(panel)
    train_sites = set(train["site_id"].unique())
    val = val[val["site_id"].isin(train_sites)]
    test = test[test["site_id"].isin(train_sites)]

    tr, va, te = prep(train), prep(val), prep(test)
    X_tr, y_tr = tr[FCOLS], tr["cyclist_count"]

    print("training RF", flush=True)
    t0 = time.time()
    model = RandomForestRegressor(
        n_estimators=100, max_depth=20, max_samples=0.3,
        random_state=SEED, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    print("predicting", flush=True)
    parts = []
    for split_name, df in [("train", tr), ("val", va), ("test", te)]:
        pred = np.clip(model.predict(df[FCOLS]), 0, None)
        out = df[KEEP_COLS].copy()
        out["predicted"] = pred
        out["residual"] = out["cyclist_count"] - pred
        out["split"] = split_name
        parts.append(out)
        print(f"  {split_name}: {len(out):,} rows", flush=True)

    full = pd.concat(parts, ignore_index=True)
    full.to_parquet(OUT, index=False)
    print(f"saved {OUT}  ({OUT.stat().st_size / 1e6:.2f} MB)", flush=True)
    print(f"  columns: {list(full.columns)}", flush=True)


if __name__ == "__main__":
    main()
