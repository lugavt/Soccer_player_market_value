"""Compare a registered model with the champion; promote it if it is better.

    python -m src.models.promote                 # newest version vs champion (report only)
    python -m src.models.promote --version 4     # a specific version
    python -m src.models.promote --yes           # promote if it passes the gate
    python -m src.models.promote --export        # copy the champion's files to models/

Both models are re-scored now, on the same test seasons of the current data,
each with its own features — so the comparison is fair even when the data or
the feature set changed since the champion was trained.

The gate: lower test MAPE than the champion, and both intervals' coverage within
COVERAGE_TOLERANCE points of nominal (50% and 80%). Passing it is necessary, not
sufficient: --yes is still a person's decision, so read the per-league table.
"""

import argparse

import pandas as pd

from src.data.loader import load_panel
from src.features.engineer import build_features, temporal_split
from src.models import registry

COVERAGE_TOLERANCE = 5.0   # points either side of 50 / 80


def compare(candidate: str, champion: str | None) -> tuple[pd.DataFrame, list[str]]:
    """Side-by-side metrics table and the list of gate failures (empty = passes)."""
    df = build_features(load_panel())
    *_, (_, _, test_rows) = temporal_split(df)
    versions = [v for v in (champion, candidate) if v is not None]
    table = pd.DataFrame({
        f"v{v}{' (champion)' if v == champion else ''}": registry.evaluate(*registry.load_version(v), test_rows)
        for v in versions
    })

    cand = table.iloc[:, -1]
    failures = []
    for level in (50, 80):
        cov = cand[f"test_cov{level}"]
        if abs(cov - level) > COVERAGE_TOLERANCE:
            failures.append(f"{level}% interval covers {cov:.1f}% (allowed {level}±{COVERAGE_TOLERANCE:.0f})")
    if champion is not None and champion != candidate:
        champ = table.iloc[:, 0]
        if cand["test_mape"] >= champ["test_mape"]:
            failures.append(f"MAPE {cand['test_mape']:.1f}% is not below the champion's {champ['test_mape']:.1f}%")
    return table, failures


def print_table(table: pd.DataFrame) -> None:
    """Overall block first, then one block per league."""
    metrics = ["mape", "rmse_log", "r2", "cov50", "cov80", "n"]
    leagues = ["all"] + sorted({k.split(".", 1)[1] for k in table.index if "." in k})
    keys = [(lg, m) for lg in leagues for m in metrics]
    names = [f"test_{m}" if lg == "all" else f"test_{m}.{lg}" for lg, m in keys]
    shown = table.reindex(names).set_axis(pd.MultiIndex.from_tuples(keys, names=["league", "metric"]))
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print(shown.dropna(how="all").to_string())


def main(version: str | None = None, yes: bool = False, export: bool = False):
    if export:
        registry.export_champion()
        return

    champion = registry.champion_version()
    candidate = version or registry.latest_version()
    if candidate is None:
        raise SystemExit("no registered versions yet — run python -m src.models.train")
    if candidate == champion:
        print(f"version {candidate} is already the champion")

    print(f"candidate: version {candidate}   champion: {champion or 'none'}\n")
    table, failures = compare(candidate, champion)
    print_table(table)

    if failures:
        print("\nGATE: FAIL\n  - " + "\n  - ".join(failures))
    else:
        print("\nGATE: PASS")

    if yes and candidate != champion:
        if failures:
            raise SystemExit("not promoted: the gate failed")
        registry.set_champion(candidate)
        print(f"version {candidate} is now the champion")
        registry.export_champion()
    elif not failures and candidate != champion:
        print("re-run with --yes to promote it")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", help="registered version to compare (default: newest)")
    parser.add_argument("--yes", action="store_true", help="promote if the gate passes")
    parser.add_argument("--export", action="store_true", help="only copy the champion's files to models/")
    main(**vars(parser.parse_args()))
