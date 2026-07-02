# Demo bundle

This folder makes the dashboard runnable (locally and on Streamlit Community Cloud) without
downloading the full ~100 GB dataset or training a model:

- `checkpoint.pt` - the trained goal-anchored model (~11 MB).
- `data/val/` - 18 curated scenarios from the Argoverse 2 Motion Forecasting validation split,
  chosen to include high-risk conflicts, turn maneuvers, and ordinary driving.

The dashboard uses this bundle automatically whenever `FORESEE_DATA_ROOT` is not set.

## Data attribution

The scenarios under `data/` are a small sample of the
[Argoverse 2 Motion Forecasting Dataset](https://www.argoverse.org/av2.html)
(Argo AI, LLC), redistributed for non-commercial demonstration under the terms of
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
If you use the data itself, cite:

> Wilson et al., "Argoverse 2: Next Generation Datasets for Self-Driving Perception and
> Forecasting", NeurIPS Datasets and Benchmarks 2021.
