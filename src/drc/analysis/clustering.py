"""Cluster the fitted curves to ask whether constructions learn the same way.

If every construction climbs its dose-response curve with roughly the same
shape, the Hill parameters should huddle together and k-means won't find clean
groups. But if some constructions need a sharp threshold while others rise
gently, the parameter vectors should split — and a high silhouette at k=2 is the
quantitative tell. That's the "two routes to grammar" hypothesis (H3), tested
without us hand-picking the routes.

Each (construction, seed) becomes a feature vector ``[E0, log(E50), n, Emax]``.
We log ``E50`` because it spans orders of magnitude across constructions; left
raw it would swamp the z-scoring and the clustering would just rediscover "which
construction needed the most data." After standardising, we sweep k in {2,3,4}
and record silhouette scores so the decision rule can read them off directly.

CLI::

    python -m drc.analysis.clustering --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from drc.data.download import load_config, resolve_path

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)

# Features in fixed order. log(E50) tames the scale; the rest go in raw and the
# z-scoring handles the units.
FEATURE_NAMES = ("E0", "log_E50", "n", "Emax")

# k values to try. Two seeds per construction times four constructions gives us
# only a dozen points, so anything past 4 would be overfitting noise.
K_VALUES = (2, 3, 4)


def build_features(hill_fits: pd.DataFrame) -> pd.DataFrame:
    """Turn converged Hill fits into the clustering feature table.

    Drops non-converged rows up front — a failed fit has no meaningful
    parameters to cluster on, and silently feeding NaNs to k-means would either
    crash or, worse, get imputed away.
    """
    import numpy as np

    good = hill_fits[hill_fits["converged"].astype(bool)].copy()
    if good.empty:
        raise RuntimeError(
            "No converged Hill fits to cluster. Check results/hill_fits.csv."
        )

    good["log_E50"] = np.log(np.clip(good["E50"].to_numpy(dtype=float), 1e-6, None))
    feats = good[["construction", "seed", "E0", "log_E50", "n", "Emax"]].copy()
    return feats.reset_index(drop=True)


def zscore(feats: pd.DataFrame) -> np.ndarray:
    """Standardise each feature column to zero mean, unit variance."""

    X = feats[list(FEATURE_NAMES)].to_numpy(dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0  # a constant feature contributes nothing; don't divide by 0
    return (X - mu) / sd


def cluster(feats: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Run k-means for each k and attach labels plus silhouette scores.

    Returns the feature table with one ``cluster_k{K}`` column per k. The
    silhouette for each k is stored in every row of a ``silhouette_k{K}`` column
    (constant within k) so a single CSV carries both the assignments and the
    scores the decision rule needs.
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = zscore(feats)
    out = feats.copy()
    n_samples = X.shape[0]

    for k in K_VALUES:
        if n_samples <= k:
            logger.warning("Only %d points; skipping k=%d.", n_samples, k)
            out[f"cluster_k{k}"] = -1
            out[f"silhouette_k{k}"] = np.nan
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(X)
        out[f"cluster_k{k}"] = labels
        # Silhouette is undefined if every point lands in one cluster.
        if len(set(labels)) > 1:
            out[f"silhouette_k{k}"] = float(silhouette_score(X, labels))
        else:
            out[f"silhouette_k{k}"] = np.nan
        logger.info("k=%d silhouette=%.3f", k, out[f"silhouette_k{k}"].iloc[0])

    return out


def run(config_path: Path, seed: int = 0) -> Path:
    """Cluster the Hill fits and write ``results/cluster_assignments.csv``."""
    config = load_config(config_path)
    results_dir = resolve_path(config_path, config["paths"]["results"])
    hill_path = results_dir / "hill_fits.csv"
    if not hill_path.exists():
        raise FileNotFoundError(
            f"Hill fits not found at {hill_path}. Run drc.analysis.hill first."
        )

    import pandas as pd

    fits = pd.read_csv(hill_path)
    feats = build_features(fits)
    assignments = cluster(feats, seed=seed)

    out_path = results_dir / "cluster_assignments.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(out_path, index=False)
    logger.info("Wrote cluster assignments (%d rows) to %s", len(assignments), out_path)
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--seed", type=int, default=0, help="k-means RNG seed.")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    run(args.config, seed=args.seed)


if __name__ == "__main__":
    main()
