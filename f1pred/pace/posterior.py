"""Posterior samples on disk: one `.npz` of arrays plus a small JSON sidecar of metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SAMPLE_PREFIX = "s__"
RHAT_PREFIX = "r__"
ESS_PREFIX = "e__"
SUMMARY_COLUMNS = ["param", "mean", "sd", "lo", "hi", "r_hat", "ess"]
# Vector parameters worth naming one row at a time, and the lookup that names them.
LABELLED = {
    "skill": "driver_season_keys",
    "skill_season": "driver_season_keys",
    "car": "constructor_season_keys",
}
SCALARS = (
    "beta_grid",
    "kappa",
    "sigma_q",
    "tau_skill",
    "tau_season",
    "tau_car",
    "tau_car_reg",
    "tau_form",
    "tau_wet",
    "tau_track",
    "h0",
    "h_wet",
)


@dataclass
class Posterior:
    """Flattened draws (`samples[name]` has a leading draw axis) plus per-parameter diagnostics."""

    samples: dict[str, np.ndarray]
    design_meta: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    r_hat: dict[str, np.ndarray] = field(default_factory=dict)
    ess: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(next(iter(self.samples.values())).shape[0])

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {SAMPLE_PREFIX + k: np.asarray(v) for k, v in self.samples.items()}
        arrays |= {RHAT_PREFIX + k: np.asarray(v) for k, v in self.r_hat.items()}
        arrays |= {ESS_PREFIX + k: np.asarray(v) for k, v in self.ess.items()}
        np.savez_compressed(path, **arrays)
        path.with_suffix(".json").write_text(
            json.dumps({"design_meta": self.design_meta, "diagnostics": self.diagnostics}, indent=2)
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> Posterior:
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            samples = {
                k[len(SAMPLE_PREFIX) :]: z[k] for k in z.files if k.startswith(SAMPLE_PREFIX)
            }
            r_hat = {k[len(RHAT_PREFIX) :]: z[k] for k in z.files if k.startswith(RHAT_PREFIX)}
            ess = {k[len(ESS_PREFIX) :]: z[k] for k in z.files if k.startswith(ESS_PREFIX)}
        side = json.loads(path.with_suffix(".json").read_text())
        return cls(samples, side["design_meta"], side.get("diagnostics", {}), r_hat, ess)

    def labels(self, name: str) -> list[str]:
        lookup = LABELLED.get(name)
        if lookup is None:
            return [f"{name}[{i}]" for i in range(self.samples[name].shape[1])]
        return [f"{name}[{a},{b}]" for a, b in self.design_meta[lookup]]

    def summary(self, names: tuple[str, ...] | None = None) -> pd.DataFrame:
        """Mean, sd, 90% interval, r_hat and ESS for the parameters worth reading by eye."""
        wanted = names or (tuple(LABELLED) + SCALARS)
        rows = []
        for name in wanted:
            draws = self.samples.get(name)
            if draws is None:
                continue
            flat = draws.reshape(draws.shape[0], -1)
            labels = [name] if flat.shape[1] == 1 and name in SCALARS else self.labels(name)
            lo, hi = np.percentile(flat, [5, 95], axis=0)
            rh = np.ravel(self.r_hat.get(name, np.full(flat.shape[1], np.nan)))
            es = np.ravel(self.ess.get(name, np.full(flat.shape[1], np.nan)))
            for i, label in enumerate(labels):
                rows.append(
                    {
                        "param": label,
                        "mean": float(flat[:, i].mean()),
                        "sd": float(flat[:, i].std(ddof=1)),
                        "lo": float(lo[i]),
                        "hi": float(hi[i]),
                        "r_hat": float(rh[i]) if i < rh.size else float("nan"),
                        "ess": float(es[i]) if i < es.size else float("nan"),
                    }
                )
        return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

    def season_table(self, name: str, season: int) -> pd.DataFrame:
        """`summary()` rows for one season of a labelled parameter, best first."""
        keys = self.design_meta[LABELLED[name]]
        wanted = {f"{name}[{a},{b}]" for a, b in keys if int(b) == season}
        table = self.summary((name,))
        table = table[table["param"].isin(wanted)].copy()
        table["name"] = [p.split("[", 1)[1].split(",")[0] for p in table["param"]]
        return table.sort_values("mean", ascending=False).reset_index(drop=True)

    def seasons(self) -> list[int]:
        return sorted({int(s) for _, s in self.design_meta["constructor_season_keys"]})

    def driver_table(self, season: int) -> pd.DataFrame:
        """Career skill, this season's deviation and their sum, per driver, with 90% intervals.

        The total is combined draw by draw, so its interval accounts for the two terms trading
        off against each other rather than adding their widths.
        """
        keys = self.design_meta["driver_season_keys"]
        idx = [i for i, (_, s) in enumerate(keys) if int(s) == season]
        skill = self.samples["skill"][:, idx]
        deviation = self.samples["skill_season"][:, idx]
        total = skill + deviation
        rows = [
            {
                "driver_id": keys[i][0],
                "skill": float(skill[:, j].mean()),
                "season": float(deviation[:, j].mean()),
                "season_lo": float(np.percentile(deviation[:, j], 5)),
                "season_hi": float(np.percentile(deviation[:, j], 95)),
                "total": float(total[:, j].mean()),
                "lo": float(np.percentile(total[:, j], 5)),
                "hi": float(np.percentile(total[:, j], 95)),
            }
            for j, i in enumerate(idx)
        ]
        table = pd.DataFrame(rows)
        return table.sort_values("total", ascending=False).reset_index(drop=True)

    def car_table(self, season: int) -> pd.DataFrame:
        """Constructor season pace with a 90% interval, fastest first."""
        table = self.season_table("car", season).rename(columns={"name": "constructor_id"})
        return table[["constructor_id", "mean", "lo", "hi", "r_hat"]]
