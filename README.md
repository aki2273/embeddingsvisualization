## Neural Baseline + Live Dashboard

This repo includes a GPU MLP training pipeline and a Streamlit dashboard that visualizes
loss curves, representation evolution (PCA + UMAP), linear probes, and loss landscapes.

### Data format (.npz)

Create a single `.npz` with the following keys (recommended), or point `--data` to the `data/` folder to load the provided `.pkl` files directly. When using the folder path, the processed cache is saved as `data/cached_dataset.npz`.

- `X_train_embed`, `X_val_embed` (float32)
- `y_train`, `y_val` (int64)
- `classes` (list of class names or ids)
- Optional: `financial_train`, `financial_val`, `X_excl`, `y_excl`, `financial_excl`

Example (from the notebook):

```python
np.savez(
    "data/industry_train.npz",
    X_train_embed=X_train_embed,
    X_val_embed=X_val_embed,
    y_train=y_train,
    y_val=y_val,
    financial_train=financial_train_scaled,
    financial_val=financial_val_scaled,
    classes=classes,
)
```

### Train

```bash
python train.py \
  --data data/cached_dataset.npz \
  --use-financial \
  --fusion concat \
  --epochs 50 \
  --viz-every 1 \
  --landscape-every 5 \
  --export-mp4
```

If your inputs are not standardized, add `--standardize`. Otherwise, the trainer auto-checks training stats and skips scaling when data already looks standardized; `--standardize` always forces scaling.

To avoid diagnostics bottlenecks, run training in offline mode and compute diagnostics later (baseline is computed in the diagnostics step):

```bash
python train.py --data data/cached_dataset.npz --diagnostics offline --save-epoch-every 1
python diagnostics.py --run-dir runs/<timestamp> --data data/cached_dataset.npz
```

Offline diagnostics also generate:

- PCA decision surfaces in embedding space
- interpolation curves between sample pairs
- neuron response curves (hidden unit activations vs PC1)

To cache per-epoch visualization inputs during training, use `--save-viz-every` (default: 1). This stores `logits_viz` and `penultimate_viz` for the fixed `viz_idx` subset and lets diagnostics reuse them without recomputing forward passes.

To load directly from `data/df_train.pkl` and `data/df_financials_train.pkl`:

```bash
python train.py --data data --use-financial
```

### Performance tuning

Defaults now favor throughput (larger batch/model, fewer diagnostics). Useful knobs:

- `--batch-size`, `--hidden-dims`
- `--num-workers`, `--cpu-jobs`, `--prefetch-factor`, `--torch-threads`
- `--compile/--no-compile`, `--tf32/--no-tf32`, `--cudnn-benchmark/--no-cudnn-benchmark`
- `--viz-every`, `--probe-every`, `--landscape-every`

### Dashboard

```bash
streamlit run dashboard.py -- --runs-dir runs
```

On WSL/headless shells, prevent the browser auto-open:

```bash
BROWSER=none streamlit run dashboard.py -- --runs-dir runs
```

Artifacts (metrics, projections, landscapes, snapshots, checkpoints) are saved under
`runs/<timestamp>/`.

### MLP representation grids (like the reference image)

You can get grid-style plots with an MLP by visualizing how representations evolve
across layers (not the raw 768-D input space).

Representation evolution (small multiples):
- Sample 2k to 5k points once (fixed indices).
- Capture activations at: input, block1, block2, penultimate, logits.
- Project each activation set to 2D with PCA or UMAP.
- Plot as a grid (one panel per layer), colored by class.

Scatter-matrix look:
- Use top PCs (PC1..PC6) of a layer and plot pairwise scatter panels.
- Or pick 8 to 16 neurons and plot neuron_i vs neuron_j.

Accuracy curve:
- Plot train/val accuracy over epochs (already in `metrics.csv`).

Decision surface in 2D:
- Train a simple classifier on the 2D PCA/UMAP projection.
- Plot decision regions plus points (PCA surface is already in diagnostics).

Implementation notes:
- Cache a fixed sample once and save per-epoch projections + labels.
- Use `return_features=True` (or hooks) to capture layer activations.
- Store artifacts under `runs/<run>/projections/epoch_###.npz` for the dashboard.

### Manim (optional)

Manim scenes live under `grok/`. Example:

```bash
manimgl grok/grokking_hacking_1.py GrokkingHackingOne
```

These scripts hardcode `data_dir` paths; update them to point at your exported assets.
