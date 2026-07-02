"""Analyse NHTSA Standing General Order crash reports (Level-2 ADAS; Tesla files ~84% of
them) and map the crashes onto the same conflict taxonomy the model uses. Downloads the
public CSV on first run and writes figures to assets/insights/.

Caveats: Tesla redacts all narrative text, many fields are Unknown (percentages are over
known values), and there is no mileage denominator, so this measures crash composition,
not crash rate.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SGO_ADAS_URL = "https://static.nhtsa.gov/odi/ffdd/sgo-2021-01/SGO-2021-01_Incident_Reports_ADAS.csv"
DATA_DIR = Path.home() / "data" / "nhtsa"
OUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "insights"

# Map NHTSA "SV Pre-Crash Movement" to Foresee's conflict taxonomy (foresee.risk).
MOVEMENT_TO_CONFLICT = {
    "Proceeding Straight": "Closing on lead / fixed object",
    "Stopped": "Closing on lead / fixed object",
    "Decelerating": "Closing on lead / fixed object",
    "Lane / Road Departure": "Lane departure / swerve",
    "Changing Lanes": "Lane change / merge",
    "Merging": "Lane change / merge",
    "Making Left Turn": "Turn across path",
    "Making Right Turn": "Turn across path",
    "Making U-Turn": "Turn across path",
    "Negotiating a Curve": "Lane departure / swerve",
}

DARK, FG, ACC = "#0e1117", "#c8ccd4", "#1f9bff"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"[data] downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_tesla() -> pd.DataFrame:
    csv = _download(SGO_ADAS_URL, DATA_DIR / "adas.csv")
    df = pd.read_csv(csv, low_memory=False, encoding="latin-1")
    return df, df[df["Make"] == "TESLA"].copy()


def _known(series: pd.Series):
    """Drop Unknown/redacted/blank; return (value_counts, unknown_fraction)."""
    s = series.astype(str).str.strip()
    bad = s.str.upper().isin(["UNKNOWN", "NAN", ""]) | s.str.contains("REDACTED", case=False)
    vc = s[~bad].value_counts()
    return vc, float(bad.mean())


def _barh(vc, title, path, color=ACC, note=""):
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * len(vc) + 1)))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    y = np.arange(len(vc))[::-1]
    ax.barh(y, vc.values, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(vc.index, color=FG, fontsize=9)
    for yi, v in zip(y, vc.values):
        ax.text(v, yi, f" {v}", va="center", color=FG, fontsize=9)
    ax.set_title(title, color="white", fontsize=12)
    if note:
        ax.set_xlabel(note, color="#8893a5", fontsize=8)
    ax.tick_params(colors="#8893a5")
    for s in ax.spines.values():
        s.set_color("#333")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=DARK)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, t = load_tesla()
    n = len(t)
    share = 100 * n / len(df)
    print("\n=== NHTSA SGO Level-2 ADAS crash reports (current snapshot) ===")
    print(f"total reports: {len(df)} | Tesla: {n} ({share:.0f}% of all L2 ADAS crash reports)")

    engaged = (t["Engagement Status"].astype(str).str.contains("Engaged", case=False)).mean()
    print(f"driver-assist confirmed/alleged engaged at crash: {100*engaged:.0f}%")
    print("narratives redacted (CBI): "
          f"{100*t['Narrative'].astype(str).str.contains('REDACTED').mean():.0f}%  "
          "-> structured-field analysis only")

    # Manufacturer share figure.
    ent = df["Reporting Entity"].value_counts().head(6)
    _barh(ent, "Who reports the most Level-2 ADAS crashes?", OUT_DIR / "01_entity_share.png",
          color="#e6194B")

    # Tesla impact location -> striking vs struck.
    fa = {c.replace("SV Contact Area - ", ""): int((t[c] == "Y").sum())
          for c in t.columns if c.startswith("SV Contact Area")}
    fa.pop("Unknown", None)
    front = sum(v for k, v in fa.items() if k.startswith("Front"))
    rear = sum(v for k, v in fa.items() if k.startswith("Rear"))
    print(f"\nTesla impact location: FRONT-type {front}  vs  REAR-type {rear}  "
          f"-> Tesla was the *striking* vehicle in ~{100*front/(front+rear):.0f}% of known cases")
    vc = pd.Series(fa).sort_values(ascending=False)
    _barh(vc, "Where the Tesla was hit (impact location)", OUT_DIR / "02_impact_area.png",
          color="#f58231", note="Front-heavy = the car drove INTO something")

    # Pre-crash movement + conflict-taxonomy mapping.
    mv, mv_unk = _known(t["SV Pre-Crash Movement"])
    _barh(mv.head(8), f"Tesla pre-crash movement ({mv_unk*100:.0f}% unknown, excluded)",
          OUT_DIR / "03_precrash_movement.png", color="#3cb44b")

    t["_conflict"] = t["SV Pre-Crash Movement"].map(MOVEMENT_TO_CONFLICT)
    conf = t["_conflict"].value_counts()
    print("\n--- Real Tesla crashes mapped to Foresee's conflict taxonomy ---")
    for k, v in conf.items():
        print(f"  {k:<32} {v:>4}  ({100*v/conf.sum():.0f}% of classifiable crashes)")
    _barh(conf, "Real Tesla crashes by Foresee conflict type", OUT_DIR / "04_conflict_taxonomy.png",
          color=ACC, note="The conflict types Foresee predicts, seen in real crashes")

    # Crash partner + roadway + severity.
    cw, cw_unk = _known(t["Crash With"])
    _barh(cw.head(8), f"What Tesla crashed with ({cw_unk*100:.0f}% unknown, excluded)",
          OUT_DIR / "05_crash_with.png", color="#911eb4")
    rd, rd_unk = _known(t["Roadway Type"])
    _barh(rd.head(8), f"Roadway type ({rd_unk*100:.0f}% unknown, excluded)",
          OUT_DIR / "06_roadway.png", color="#42d4f4")

    sev, _ = _known(t["Highest Injury Severity Alleged"])
    fatal = int((t["Highest Injury Severity Alleged"] == "Fatality").sum())
    spd = pd.to_numeric(t["SV Precrash Speed (MPH)"], errors="coerce")
    print(f"\nseverity (known): fatalities={fatal} | "
          f"median pre-crash speed {spd.median():.0f} mph (p90 {spd.quantile(.9):.0f})")

    # Save a tidy CSV summary.
    summary = pd.concat([
        conf.rename("count").to_frame().assign(dimension="conflict_type"),
        mv.head(8).rename("count").to_frame().assign(dimension="precrash_movement"),
        cw.head(8).rename("count").to_frame().assign(dimension="crash_with"),
    ])
    summary.to_csv(OUT_DIR / "tesla_crash_summary.csv")
    print(f"\n[out] figures + summary written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
