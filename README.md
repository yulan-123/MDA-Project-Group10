# MDA-Project-Group10

Analysis of cycling traffic in Flanders using AWV bicycle counting stations. The project predicts hourly cyclist counts with a Random Forest model, detects anomalies, and clusters stations by usage profile.

---

## Project Structure

```
MDA-Project-Group10/
├── src/                          # Data pipeline and preprocessing scripts
│   ├── 01_build_event_features.py
│   ├── 02_build_panel.py
│   ├── split.py
│   ├── download_data.py
│   ├── cache_predictions.py
│   └── cache_shap.py
├── Predictive model/             # EDA, model training, and prediction dashboard
│   ├── EDA_and_RFmodels.ipynb
│   └── dashboard_for_predictive_model.py
├── Cluster/                      # Clustering analysis and cluster dashboard
│   ├── clustering.ipynb
│   ├── dashboard_for_clusters.py
│   └── README.md                 # Full details for this folder
├── data/                         # Raw and processed data (not committed)
├── requirements.txt
└── .gitignore
```

---

## `src/` — Data Pipeline

Run these scripts in order before using the notebooks or dashboards.

### `download_data.py`

Downloads the processed data assets automatically on first run. Large data files
are not committed to GitHub; they are stored as GitHub release assets and saved
locally under `data/` or `Cluster/` when needed.

Run manually if needed: `python src/download_data.py`

### `01_build_event_features.py`

Generates a daily calendar of event features from 2019-08-01 to 2025-12-31 and saves it to `data/event_features.csv`.

Reads two input CSVs from `data/`:
- `belgian_holidays.csv` — public holidays with Dutch names, mapped to categories (`new_year`, `easter`, `christmas`, `national`).
- `school_holidays_flanders.csv` — school holiday periods with start/end dates.

Produces one row per calendar day with these columns:
| Column | Description |
|---|---|
| `date` | Calendar date |
| `is_public_holiday` | 1 if a Belgian public holiday |
| `is_school_holiday` | 1 if within a Flemish school holiday period |
| `holiday_type` | Category string (`none`, `national`, `easter`, `christmas`, etc.) |
| `days_to_nearest_holiday` | Integer days to the closest public holiday |

---

### `02_build_panel.py`

Merges all raw data sources into a single hourly panel and saves it to `data/panel_merged.parquet`. This is the main input for all downstream notebooks and dashboards.

Reads from `data/`:
- `sites.csv` — counting station metadata (id, coordinates, road name, municipality).
- `richtingen.csv` — direction per station (`IN` / `OUT`).
- `road_bearings.csv` — compass bearing of the road at each station. The `OUT` direction is computed as bearing + 180°.
- `data-YYYY-MM.csv` — monthly raw cyclist counts; only rows of type `FIETSERS` are kept and aggregated to hourly totals.
- `weather_multi_region.csv` — hourly weather for multiple Flemish regions. Each station is assigned to its nearest region using Euclidean distance on coordinates.
- `event_features.csv` — output of `01_build_event_features.py`.

Key derived columns added during the merge:
| Column | Description |
|---|---|
| `cyclist_count` | Hourly cyclist total for a (station, direction) pair |
| `hour`, `day_of_week`, `month`, `year` | Time components |
| `is_weekend` | 1 for Saturday/Sunday |
| `is_in` | 1 if direction is `IN` |
| `wind_impact` | cos(wind direction − road bearing): tailwind (+1) vs headwind (−1) |
| `is_holiday` | 1 if weekend, public holiday, or school holiday |

---

### `split.py`

Defines the temporal train/val/test split used by all models:
- **Train:** 2023
- **Val:** 2024
- **Test:** 2025

Years before 2023 are excluded because most stations were not yet operational (only ~25 stations before 2022, ~100 more came online during 2022). Exposes a `temporal_split(panel)` function that returns three filtered DataFrames.

---

### `cache_predictions.py`

Trains the Random Forest model and saves predictions for all three splits to `data/predictions.parquet` so the dashboard can load them without retraining.

- Downloads `panel_merged.parquet` automatically if it is missing.
- Uses the same 20 features and hyperparameters as the no-lag baseline model in the notebook (`n_estimators=100`, `max_depth=20`, `max_samples=0.3`).
- Excludes stations from val/test that were never seen in training.
- Output columns include actual `cyclist_count`, `predicted`, `residual`, and `split` label.

Run once: `python src/cache_predictions.py`

---

### `cache_shap.py`

Trains the same Random Forest and computes SHAP values on a 5000-row random sample of the test set. Saves the result to `data/shap_values.pkl`.

The pickle contains: the feature sample, SHAP values array, explainer expected value, and metadata (seed, feature names).

Run once: `python src/cache_shap.py`

---

## `Predictive model/` — EDA and Prediction Dashboard

### `EDA_and_RFmodels.ipynb`

Jupyter notebook covering the full modelling workflow:

1. **Data loading** — reads `panel_merged.parquet` and applies the `temporal_split`.
2. **Data quality** — checks missing values, zero counts, and IN/OUT balance.
3. **Station map** — scatter plot of all station coordinates.
4. **Temporal patterns** — average cyclist count by hour, weekday vs weekend comparison, and daily volume over the year.
5. **Top stations** — horizontal bar chart of the 15 busiest stations by total count.
6. **Weather correlations** — Spearman correlation of weather variables against cyclist count.
7. **Random Forest — 20 features (baseline)** — trains an RF regressor on 2023 data and evaluates on val/test. Reports MAE, RMSE, and R². Results: Train R²=0.892, Val R²=0.679, Test R²=0.649.
8. **Random Forest — 21 features (lag-24h)** — adds a 24-hour lagged count feature. Improves test R² to 0.779.
9. **SHAP analysis** — uses the 20-feature no-lag model as an interpretability benchmark and loads pre-computed SHAP values from `cache_shap.py`. It produces:
   - Global bar chart of mean |SHAP| per feature.
   - Beeswarm summary plot.
   - Dependence plots for `hour`, `temp_c`, and `humidity`.
   - Waterfall plots for individual normal, over-prediction, and under-prediction case studies.

---

### `dashboard_for_predictive_model.py`

Interactive Shiny for Python dashboard for exploring model predictions and anomalies.

**Features:**
- Station selector and date range picker in a sidebar.
- A dual-axis Plotly chart showing actual vs predicted hourly cyclist counts, with precipitation overlaid on a secondary axis.
- Anomaly alarms: data points where the absolute prediction error exceeds a user-defined threshold are flagged with red markers.

Loads `data/predictions.parquet` if available; falls back to `panel_merged.parquet` (without predictions).
Both files are downloaded automatically on first run if missing.

**To run:**
```bash
shiny run "Predictive model/dashboard_for_predictive_model.py"
```

---

## `Cluster/`

Contains the station clustering analysis (K-Means on 5 usage-profile features) and an interactive map dashboard. See [`Cluster/README.md`](Cluster/README.md) for full details.

---

