"""
Download processed data assets on first run.

Large data files are stored as GitHub release assets instead of being committed
to this repository. The notebooks, cache scripts, and dashboards call
``ensure_data`` before reading from the local data folder.
"""

from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://github.com/pihaoking/mda-project-group10-data/releases/download/v1.0"

ASSETS = {
    "panel_merged.parquet": ROOT / "data" / "panel_merged.parquet",
    "predictions.parquet": ROOT / "data" / "predictions.parquet",
    "shap_values.pkl": ROOT / "data" / "shap_values.pkl",
    "shap_values_stratified.pkl": ROOT / "data" / "shap_values_stratified.pkl",
    "station_profiles_clustered.csv": ROOT / "Cluster" / "station_profiles_clustered.csv",
}


def download_asset(name, force=False):
    if name not in ASSETS:
        raise KeyError(f"Unknown data asset: {name}")

    path = ASSETS[name]
    if path.exists() and path.stat().st_size > 0 and not force:
        print(f"{name} already exists")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{name}"
    tmp_path = path.with_suffix(path.suffix + ".part")

    print(f"Downloading {name}...")
    request = Request(url, headers={"User-Agent": "mda-project-group10"})
    with urlopen(request) as response, tmp_path.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    tmp_path.replace(path)
    print(f"Saved to {path}")
    return path


def ensure_data(names=None):
    selected = list(ASSETS) if names is None else list(names)
    for name in selected:
        download_asset(name)


if __name__ == "__main__":
    ensure_data()
