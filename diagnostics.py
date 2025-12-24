import argparse
import inspect
import json
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.pop("TORCH_LOGS", None)
import torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.deprecation")
warnings.filterwarnings("ignore", message=".*n_jobs value.*")

_cuml_import_error = None
try:
    from cuml import PCA as cuPCA
    from cuml import UMAP as cuUMAP
    from cuml import LogisticRegression as cuLogisticRegression
    _cuml_available = True
except Exception as exc:
    _cuml_import_error = exc
    _cuml_available = False
    cuPCA = None
    cuUMAP = None
    cuLogisticRegression = None
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from umap import UMAP

from models import MLPClassifier
from metrics import compute_metrics
from train import (
    load_npz,
    load_from_pickles,
    maybe_standardize,
    extract_features,
    run_linear_probes,
    compute_loss_landscape,
    make_directions,
    parse_hidden_dims,
    append_csv,
    compute_class_weights,
)

USE_CUML = _cuml_available and torch.cuda.is_available()
DEFAULT_REP_LAYERS = ("input", "block1", "block2", "penultimate", "logits")


def parse_args():
    parser = argparse.ArgumentParser(description="Run offline diagnostics from saved checkpoints.")
    parser.add_argument("--run-dir", default="", help="Path to a training run directory (default: latest)")
    parser.add_argument("--runs-dir", default="runs", help="Base runs directory for latest selection")
    parser.add_argument("--data", required=True, help="Path to .npz or data folder")
    parser.add_argument(
        "--use-financial",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override use_financial from config",
    )
    parser.add_argument("--epochs", default="", help="Comma-separated list of epochs to process")
    parser.add_argument("--overwrite", action="store_true", help="Clear existing diagnostics outputs")
    parser.add_argument("--no-landscape", action="store_true", help="Skip loss landscape computation")
    parser.add_argument("--no-surface", action="store_true", help="Skip PCA decision surface")
    parser.add_argument("--no-interp", action="store_true", help="Skip interpolation curves")
    parser.add_argument("--no-response", action="store_true", help="Skip neuron response curves")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline computation")
    parser.add_argument("--cpu-jobs", type=int, default=None, help="Override CPU jobs from config")
    parser.add_argument(
        "--probe-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run probe layers in parallel (threads)",
    )
    parser.add_argument(
        "--probe-parallel-jobs",
        type=int,
        default=None,
        help="Parallel jobs for probe layers (default: cpu-jobs)",
    )
    parser.add_argument("--surface-grid", type=int, default=160, help="Grid size for PCA surface")
    parser.add_argument("--surface-pad", type=float, default=0.05, help="Padding for PCA surface bounds")
    parser.add_argument("--interp-steps", type=int, default=100, help="Interpolation steps per pair")
    parser.add_argument("--interp-pairs-same", type=int, default=50, help="Same-class interpolation pairs")
    parser.add_argument("--interp-pairs-diff", type=int, default=50, help="Different-class interpolation pairs")
    parser.add_argument("--response-units", type=int, default=32, help="Number of units to track")
    parser.add_argument("--rep-every", type=int, default=5, help="Export representations every N epochs (0=disable)")
    parser.add_argument("--rep-n", type=int, default=15000, help="Total points for representation export")
    parser.add_argument("--rep-method", choices=["pca", "umap"], default="umap")
    parser.add_argument("--rep-pc-k", type=int, default=6, help="PCs to store for scatter-matrix view")
    parser.add_argument("--rep-umap-k", type=int, default=2, help="UMAP components to store for scatter-matrix view")
    return parser.parse_args()


def load_config(run_dir):
    config_path = Path(run_dir) / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def parse_epoch_list(value):
    if not value:
        return None
    return sorted({int(x.strip()) for x in value.split(",") if x.strip()})


def resolve_run_dir(run_dir_arg, runs_dir):
    if run_dir_arg:
        return Path(run_dir_arg)
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_path}")
    candidates = [p for p in runs_path.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No runs found under: {runs_path}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def select_interpolation_pairs(labels, n_same, n_diff, rng):
    pairs = []
    types = []
    labels = np.asarray(labels)
    classes = np.unique(labels)
    class_to_idx = {cls: np.where(labels == cls)[0] for cls in classes}

    for _ in range(n_same):
        valid = [cls for cls in classes if class_to_idx[cls].shape[0] >= 2]
        if not valid:
            break
        cls = rng.choice(valid)
        idxs = rng.choice(class_to_idx[cls], size=2, replace=False)
        pairs.append((int(idxs[0]), int(idxs[1])))
        types.append(0)

    for _ in range(n_diff):
        if classes.shape[0] < 2:
            break
        cls_a, cls_b = rng.choice(classes, size=2, replace=False)
        idx_a = rng.choice(class_to_idx[cls_a])
        idx_b = rng.choice(class_to_idx[cls_b])
        pairs.append((int(idx_a), int(idx_b)))
        types.append(1)

    return np.array(pairs, dtype=int), np.array(types, dtype=int)


def log_accel_status():
    if USE_CUML:
        print("[gpu] cuML enabled for PCA/UMAP/LogReg")
    elif _cuml_available:
        print("[gpu] cuML installed but CUDA not available; using CPU PCA/UMAP/LogReg")
    else:
        print("[gpu] cuML not installed; using CPU PCA/UMAP/LogReg")


def _filter_kwargs(cls, kwargs):
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return kwargs
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _to_numpy(array):
    if hasattr(array, "to_numpy"):
        return array.to_numpy()
    if hasattr(array, "get"):
        return array.get()
    return np.asarray(array)


def _normalize_state_dict(state):
    if not isinstance(state, dict):
        return state
    if any(key.startswith("_orig_mod.") for key in state):
        return {key.replace("_orig_mod.", "", 1): value for key, value in state.items()}
    return state


def _build_pca(random_state, n_components=2):
    pca_cls = cuPCA if USE_CUML else PCA
    pca_kwargs = {"n_components": n_components, "random_state": random_state}
    return pca_cls(**_filter_kwargs(pca_cls, pca_kwargs))


def _build_umap(random_state, n_neighbors, min_dist, n_jobs, n_components=2):
    umap_cls = cuUMAP if USE_CUML else UMAP
    umap_state = random_state if (USE_CUML or n_jobs == 1) else None
    umap_kwargs = {
        "n_components": int(n_components),
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "random_state": umap_state,
        "n_jobs": n_jobs,
    }
    return umap_cls(**_filter_kwargs(umap_cls, umap_kwargs))


def _build_logreg(**kwargs):
    logreg_cls = cuLogisticRegression if USE_CUML else LogisticRegression
    return logreg_cls(**_filter_kwargs(logreg_cls, kwargs))


def load_or_create_rep_sample(path, y_train, y_val, rep_n, seed):
    path = Path(path)
    if path.exists():
        data = np.load(path)
        return {k: data[k] for k in data.files}

    rng = np.random.default_rng(seed)
    n_train = min(rep_n // 2, y_train.shape[0])
    n_val = min(rep_n - n_train, y_val.shape[0])
    idx_train = rng.choice(y_train.shape[0], size=n_train, replace=False)
    idx_val = rng.choice(y_val.shape[0], size=n_val, replace=False)

    indices = np.concatenate([idx_train, idx_val])
    split = np.concatenate([np.zeros(n_train, dtype=np.int8), np.ones(n_val, dtype=np.int8)])
    labels = np.concatenate([y_train[idx_train], y_val[idx_val]]).astype(np.int64)

    payload = {
        "indices": indices.astype(np.int64),
        "indices_train": idx_train.astype(np.int64),
        "indices_val": idx_val.astype(np.int64),
        "split": split,
        "labels": labels,
        "rep_n": np.array(rep_n, dtype=np.int64),
        "seed": np.array(seed, dtype=np.int64),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    return payload


def compute_rep_projections(
    layer_feats, method, random_state, n_neighbors, min_dist, n_jobs, pc_k, umap_k
):
    projections = {}
    pcs = {}
    for name, feats in layer_feats.items():
        feats = feats.astype(np.float32)
        if method == "pca":
            pc_components = max(2, pc_k) if pc_k > 0 else 2
            pca = _build_pca(random_state, n_components=pc_components)
            pc_scores = _to_numpy(pca.fit_transform(feats))
            projections[f"{name}_pca"] = pc_scores[:, :2].astype(np.float32)
            if pc_k > 0:
                pcs[f"{name}_pc"] = pc_scores[:, :pc_k].astype(np.float32)
        else:
            umap_components = max(2, int(umap_k or 2))
            umap = _build_umap(
                random_state, n_neighbors, min_dist, n_jobs, n_components=umap_components
            ).fit_transform(feats)
            projections[f"{name}_umap"] = _to_numpy(umap).astype(np.float32)
            if pc_k > 0:
                pc_scores = _to_numpy(_build_pca(random_state, n_components=pc_k).fit_transform(feats))
                pcs[f"{name}_pc"] = pc_scores.astype(np.float32)
    return projections, pcs


def build_pca_surface(pca, proj, grid_size, pad):
    proj = _to_numpy(proj)
    x_min, x_max = proj[:, 0].min(), proj[:, 0].max()
    y_min, y_max = proj[:, 1].min(), proj[:, 1].max()
    x_pad = (x_max - x_min) * pad
    y_pad = (y_max - y_min) * pad
    x_lin = np.linspace(x_min - x_pad, x_max + x_pad, grid_size)
    y_lin = np.linspace(y_min - y_pad, y_max + y_pad, grid_size)
    grid_x, grid_y = np.meshgrid(x_lin, y_lin)
    grid_pc = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_embed = _to_numpy(pca.inverse_transform(grid_pc)).astype(np.float32)
    return grid_x, grid_y, grid_embed


def compute_projections_timed(layer_feats, random_state, n_neighbors, min_dist, n_jobs):
    projections = {}
    for name, feats in layer_feats.items():
        t0 = time.perf_counter()
        pca = _build_pca(random_state).fit_transform(feats)
        t1 = time.perf_counter()
        t2 = time.perf_counter()
        umap = _build_umap(random_state, n_neighbors, min_dist, n_jobs).fit_transform(feats)
        t3 = time.perf_counter()
        projections[f"{name}_pca"] = _to_numpy(pca)
        projections[f"{name}_umap"] = _to_numpy(umap)
        print(
            f"[proj] {name}: PCA {t1 - t0:.2f}s | "
            f"UMAP {t3 - t2:.2f}s | n={feats.shape[0]} d={feats.shape[1]}"
        )
    return projections


def start_progress_heartbeat(label, interval=10.0):
    stop_event = threading.Event()

    def _runner():
        start = time.monotonic()
        while not stop_event.wait(interval):
            elapsed = time.monotonic() - start
            print(f"{label} running... {elapsed:.0f}s elapsed")

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    return stop_event, thread


@torch.no_grad()
def predict_proba(model, x_embed, x_fin, device, use_amp, batch_size):
    model.eval()
    outputs = []
    for start in range(0, x_embed.shape[0], batch_size):
        end = start + batch_size
        xb = torch.as_tensor(x_embed[start:end], dtype=torch.float32, device=device)
        if x_fin is not None:
            xf = torch.as_tensor(x_fin[start:end], dtype=torch.float32, device=device)
        else:
            xf = None
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(xb, xf)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        outputs.append(proba)
    return np.concatenate(outputs, axis=0)


def interpolate_pairs(x_embed, x_fin, pairs, steps):
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    t_col = t[:, None]
    embed_list = []
    fin_list = []
    for idx_a, idx_b in pairs:
        z_a = x_embed[idx_a]
        z_b = x_embed[idx_b]
        embed_list.append((1.0 - t_col) * z_a + t_col * z_b)
        if x_fin is not None:
            f_a = x_fin[idx_a]
            f_b = x_fin[idx_b]
            fin_list.append((1.0 - t_col) * f_a + t_col * f_b)
    embeds = np.vstack(embed_list).astype(np.float32)
    fins = np.vstack(fin_list).astype(np.float32) if x_fin is not None else None
    return t, embeds, fins


def main():
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir, args.runs_dir)
    config = load_config(run_dir)

    if args.use_financial is None:
        use_financial = bool(config.get("use_financial", False))
    else:
        use_financial = args.use_financial

    seed = int(config.get("seed", 42))
    val_split = float(config.get("val_split", 0.2))
    viz_max_points = int(config.get("viz_max_points", 2000))
    probe_max_train = int(config.get("probe_max_train", 4000))
    probe_max_val = int(config.get("probe_max_val", 2000))
    probe_max_iter = int(config.get("probe_max_iter", 1000))
    probe_tol = float(config.get("probe_tol", 1e-2))
    probe_c = float(config.get("probe_c", 0.5))
    cpu_jobs = int(config.get("cpu_jobs", 1))
    if args.cpu_jobs is not None:
        cpu_jobs = args.cpu_jobs
    landscape_every = int(config.get("landscape_every", 5))
    landscape_points = int(config.get("landscape_points", 21))
    landscape_radius = float(config.get("landscape_radius", 1.0))
    batch_size = int(config.get("batch_size", 512))

    data_path = Path(args.data)
    if data_path.suffix.lower() == ".npz":
        payload = load_npz(data_path)
    else:
        payload = load_from_pickles(data_path, use_financial, val_split, seed)

    x_train_embed = payload["X_train_embed"].astype(np.float32)
    x_val_embed = payload["X_val_embed"].astype(np.float32)
    y_train = payload["y_train"].astype(np.int64)
    y_val = payload["y_val"].astype(np.int64)
    classes = payload.get("classes") or list(range(int(y_train.max()) + 1))
    num_classes = len(classes)

    x_train_fin = payload.get("financial_train")
    x_val_fin = payload.get("financial_val")
    if use_financial:
        if x_train_fin is None or x_val_fin is None:
            raise ValueError("financial_train/financial_val missing for offline diagnostics")
        x_train_fin = x_train_fin.astype(np.float32)
        x_val_fin = x_val_fin.astype(np.float32)
    else:
        x_train_fin = None
        x_val_fin = None

    x_train_embed, x_val_embed, _, _ = maybe_standardize(
        "embeddings",
        x_train_embed,
        x_val_embed,
        None,
        force=bool(config.get("standardize", False)),
    )
    if use_financial:
        x_train_fin, x_val_fin, _, _ = maybe_standardize(
            "financial",
            x_train_fin,
            x_val_fin,
            None,
            force=bool(config.get("standardize", False)),
        )

    rng = np.random.default_rng(seed)
    diag_indices_path = run_dir / "diag_indices.npz"
    if diag_indices_path.exists():
        diag_indices = np.load(diag_indices_path)
        viz_idx = diag_indices["viz_idx"]
        probe_train_idx = diag_indices["probe_train_idx"]
        probe_val_idx = diag_indices["probe_val_idx"]
        land_idx = diag_indices["land_idx"]
    else:
        viz_idx = rng.choice(x_val_embed.shape[0], size=min(viz_max_points, x_val_embed.shape[0]), replace=False)
        probe_train_idx = rng.choice(
            x_train_embed.shape[0], size=min(probe_max_train, x_train_embed.shape[0]), replace=False
        )
        probe_val_idx = rng.choice(
            x_val_embed.shape[0], size=min(probe_max_val, x_val_embed.shape[0]), replace=False
        )
        land_idx = rng.choice(x_val_embed.shape[0], size=min(256, x_val_embed.shape[0]), replace=False)

    projections_dir = run_dir / "projections"
    landscapes_dir = run_dir / "landscapes"
    surfaces_dir = run_dir / "surfaces"
    interp_dir = run_dir / "interpolations"
    response_dir = run_dir / "responses"
    rep_root = run_dir / "diagnostics"
    rep_dir = rep_root / "rep"
    rep_sample_path = rep_root / "rep_sample.npz"
    projections_dir.mkdir(parents=True, exist_ok=True)
    landscapes_dir.mkdir(parents=True, exist_ok=True)
    surfaces_dir.mkdir(parents=True, exist_ok=True)
    interp_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    probes_path = run_dir / "probes.csv"
    if args.overwrite:
        shutil.rmtree(projections_dir, ignore_errors=True)
        shutil.rmtree(landscapes_dir, ignore_errors=True)
        shutil.rmtree(surfaces_dir, ignore_errors=True)
        shutil.rmtree(interp_dir, ignore_errors=True)
        shutil.rmtree(response_dir, ignore_errors=True)
        shutil.rmtree(rep_dir, ignore_errors=True)
        projections_dir.mkdir(parents=True, exist_ok=True)
        landscapes_dir.mkdir(parents=True, exist_ok=True)
        surfaces_dir.mkdir(parents=True, exist_ok=True)
        interp_dir.mkdir(parents=True, exist_ok=True)
        response_dir.mkdir(parents=True, exist_ok=True)
        rep_dir.mkdir(parents=True, exist_ok=True)
        if probes_path.exists():
            probes_path.unlink()
        if rep_sample_path.exists():
            rep_sample_path.unlink()

    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError("No epoch checkpoints found. Train with --diagnostics offline.")

    epochs_filter = parse_epoch_list(args.epochs)
    if epochs_filter is not None:
        ckpts = [ckpt for ckpt in ckpts if int(ckpt.stem.split("_")[-1]) in epochs_filter]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    log_accel_status()

    model = MLPClassifier(
        embed_dim=x_train_embed.shape[1],
        num_classes=num_classes,
        hidden_dims=parse_hidden_dims(config.get("hidden_dims", "1024,512")),
        dropout=float(config.get("dropout", 0.2)),
        activation=config.get("activation", "gelu"),
        layernorm=True,
        residual=bool(config.get("residual", False)),
        fusion=config.get("fusion", "concat"),
        fin_dim=x_train_fin.shape[1] if use_financial else None,
    ).to(device)

    class_weights = compute_class_weights(y_train, num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    land_x = torch.as_tensor(x_val_embed[land_idx], dtype=torch.float32, device=device)
    land_y = torch.as_tensor(y_val[land_idx], dtype=torch.long, device=device)
    land_x_fin = (
        torch.as_tensor(x_val_fin[land_idx], dtype=torch.float32, device=device) if use_financial else None
    )
    land_batch = (land_x, land_x_fin)
    dir1 = make_directions([p for p in model.parameters() if p.requires_grad], seed=seed + 1)
    dir2 = make_directions([p for p in model.parameters() if p.requires_grad], seed=seed + 2)
    alphas = np.linspace(-landscape_radius, landscape_radius, landscape_points)
    betas = np.linspace(-landscape_radius, landscape_radius, landscape_points)

    ref_embed = x_val_embed[viz_idx]
    pca = _build_pca(seed).fit(ref_embed)
    ref_proj = _to_numpy(pca.transform(ref_embed))
    grid_x, grid_y, grid_embed = build_pca_surface(pca, ref_proj, args.surface_grid, args.surface_pad)
    ref_labels = y_val[viz_idx]
    fin_center = None
    if use_financial:
        fin_center = x_train_fin.mean(axis=0)

    pairs, pair_types = select_interpolation_pairs(
        y_val, args.interp_pairs_same, args.interp_pairs_diff, rng
    )
    pair_labels = np.stack([y_val[pairs[:, 0]], y_val[pairs[:, 1]]], axis=1) if pairs.size else np.zeros((0, 2), dtype=int)

    unit_indices = None
    rep_every = max(0, int(args.rep_every))
    rep_method = args.rep_method
    rep_pc_k = max(0, int(args.rep_pc_k))
    rep_umap_k = max(2, int(args.rep_umap_k))
    rep_sample = None
    rep_labels = None
    rep_split = None
    x_rep_embed = None
    x_rep_fin = None
    if rep_every > 0:
        rep_sample = load_or_create_rep_sample(
            rep_sample_path,
            y_train,
            y_val,
            rep_n=max(1, int(args.rep_n)),
            seed=seed,
        )
        idx_train = rep_sample.get("indices_train")
        idx_val = rep_sample.get("indices_val")
        if idx_train is None or idx_val is None:
            indices = rep_sample["indices"]
            split = rep_sample["split"]
            idx_train = indices[split == 0]
            idx_val = indices[split == 1]
        rep_labels = rep_sample["labels"]
        rep_split = rep_sample["split"]
        x_rep_embed = np.vstack([x_train_embed[idx_train], x_val_embed[idx_val]]).astype(np.float32)
        if use_financial and x_train_fin is not None and x_val_fin is not None:
            x_rep_fin = np.vstack([x_train_fin[idx_train], x_val_fin[idx_val]]).astype(np.float32)

    for ckpt_path in ckpts:
        epoch = int(ckpt_path.stem.split("_")[-1])
        checkpoint = torch.load(ckpt_path, map_location=device)
        state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        model.load_state_dict(_normalize_state_dict(state))

        print(f"[offline] Epoch {epoch}: projections")
        cache_path = run_dir / "viz_cache" / f"epoch_{epoch:03d}.npz"
        if cache_path.exists():
            cache = np.load(cache_path)
            feats_val = {
                "input": x_val_embed[viz_idx],
                "penultimate": cache["penultimate_viz"],
            }
            labels_viz = cache.get("labels", y_val[viz_idx])
        else:
            feats_val = extract_features(
                model,
                x_val_embed[viz_idx],
                x_val_fin[viz_idx] if use_financial else None,
                device,
                batch_size=batch_size,
                use_amp=use_amp,
            )
            labels_viz = y_val[viz_idx]
        projections = compute_projections_timed(
            feats_val, random_state=seed, n_neighbors=15, min_dist=0.1, n_jobs=cpu_jobs
        )
        projections["labels"] = labels_viz
        np.savez(projections_dir / f"epoch_{epoch:03d}.npz", **projections)

        if rep_every > 0 and epoch % rep_every == 0 and x_rep_embed is not None and rep_labels is not None:
            rep_path = rep_dir / f"epoch_{epoch:03d}.npz"
            if not rep_path.exists() or args.overwrite:
                print(f"[offline] Epoch {epoch}: representations ({rep_method})")
                rep_feats = extract_features(
                    model,
                    x_rep_embed,
                    x_rep_fin,
                    device,
                    batch_size=batch_size,
                    use_amp=use_amp,
                    feature_keys=DEFAULT_REP_LAYERS,
                )
                if rep_feats:
                    rep_proj, rep_pcs = compute_rep_projections(
                        rep_feats,
                        method=rep_method,
                        random_state=seed,
                        n_neighbors=15,
                        min_dist=0.1,
                        n_jobs=cpu_jobs,
                        pc_k=rep_pc_k,
                        umap_k=rep_umap_k,
                    )
                    payload = {
                        **rep_proj,
                        **rep_pcs,
                        "labels": rep_labels,
                        "split": rep_split,
                        "epoch": np.array(epoch, dtype=np.int64),
                        "method": np.array(rep_method),
                        "pc_k": np.array(rep_pc_k, dtype=np.int64),
                        "umap_k": np.array(rep_umap_k, dtype=np.int64),
                        "layers": np.array(list(rep_feats.keys())),
                    }
                    np.savez(rep_path, **payload)

        if not args.no_surface:
            print(f"[offline] Epoch {epoch}: PCA decision surface")
            grid_fin = None
            if use_financial:
                grid_fin = np.repeat(fin_center[None, :], grid_embed.shape[0], axis=0).astype(np.float32)
            proba = predict_proba(model, grid_embed, grid_fin, device, use_amp, batch_size=batch_size)
            proba_grid = proba.reshape(args.surface_grid, args.surface_grid, -1)
            pred_grid = proba_grid.argmax(axis=2)
            np.savez(
                surfaces_dir / f"epoch_{epoch:03d}.npz",
                grid_x=grid_x,
                grid_y=grid_y,
                proba=proba_grid,
                pred=pred_grid,
                ref_proj=ref_proj,
                ref_labels=ref_labels,
            )

        print(f"[offline] Epoch {epoch}: probes")
        feats_train = extract_features(
            model,
            x_train_embed[probe_train_idx],
            x_train_fin[probe_train_idx] if use_financial else None,
            device,
            batch_size=batch_size,
            use_amp=use_amp,
        )
        feats_val = extract_features(
            model,
            x_val_embed[probe_val_idx],
            x_val_fin[probe_val_idx] if use_financial else None,
            device,
            batch_size=batch_size,
            use_amp=use_amp,
        )
        probe_results = run_linear_probes(
            feats_train,
            y_train[probe_train_idx],
            feats_val,
            y_val[probe_val_idx],
            max_iter=probe_max_iter,
            c_value=probe_c,
            n_jobs=cpu_jobs,
            tol=probe_tol,
            parallel_layers=args.probe_parallel,
            parallel_jobs=args.probe_parallel_jobs,
        )
        for result in probe_results:
            result.update({"epoch": epoch})
            append_csv(probes_path, list(result.keys()), result)

        if not args.no_response and "penultimate" in feats_val:
            if unit_indices is None:
                penultimate = feats_val["penultimate"]
                variances = penultimate.var(axis=0)
                topk = np.argsort(variances)[-args.response_units :][::-1]
                unit_indices = topk.astype(int)
            activations = feats_val["penultimate"][:, unit_indices]
            np.savez(
                response_dir / f"epoch_{epoch:03d}.npz",
                pc1=ref_proj[:, 0],
                activations=activations,
                unit_indices=unit_indices,
                labels=labels_viz,
            )

        if not args.no_interp and pairs.size:
            print(f"[offline] Epoch {epoch}: interpolation curves")
            t, interp_embed, interp_fin = interpolate_pairs(x_val_embed, x_val_fin, pairs, args.interp_steps)
            proba_interp = predict_proba(model, interp_embed, interp_fin, device, use_amp, batch_size=batch_size)
            proba_interp = proba_interp.reshape(pairs.shape[0], args.interp_steps, -1)
            proba_a = np.zeros((pairs.shape[0], args.interp_steps), dtype=np.float32)
            proba_b = np.zeros((pairs.shape[0], args.interp_steps), dtype=np.float32)
            for i in range(pairs.shape[0]):
                label_a, label_b = pair_labels[i]
                proba_a[i] = proba_interp[i, :, label_a]
                proba_b[i] = proba_interp[i, :, label_b]
            np.savez(
                interp_dir / f"epoch_{epoch:03d}.npz",
                t=t,
                proba_a=proba_a,
                proba_b=proba_b,
                pair_a=pairs[:, 0],
                pair_b=pairs[:, 1],
                pair_labels=pair_labels,
                pair_types=pair_types,
            )

        if args.no_landscape:
            continue
        if epoch % landscape_every != 0:
            continue

        print(f"[offline] Epoch {epoch}: loss landscape")
        losses = compute_loss_landscape(model, criterion, land_batch, land_y, dir1, dir2, alphas, betas, use_amp)
        land_path = landscapes_dir / f"epoch_{epoch:03d}.npz"
        np.savez(land_path, alphas=alphas, betas=betas, loss=losses)

    results_path = run_dir / "results.csv"
    results = []
    if results_path.exists():
        results_df = pd.read_csv(results_path)
        results = results_df.to_dict("records")
    else:
        metrics_path = run_dir / "metrics.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            if not metrics_df.empty:
                best_row = metrics_df.sort_values("val_official_f1", ascending=False).iloc[0]
                results.append(
                    {
                        "name": "MLP",
                        "val_accuracy": float(best_row["val_accuracy"]),
                        "val_f1_macro": float(best_row["val_f1_macro"]),
                        "val_official_f1": float(best_row["val_official_f1"]),
                    }
                )

    results = [row for row in results if row.get("name") != "LogReg_baseline"]
    if not args.skip_baseline:
        print("[offline] Baseline: LogisticRegression")
        heartbeat, heartbeat_thread = start_progress_heartbeat(
            "[offline] Baseline: LogisticRegression",
            interval=10.0,
        )
        x_train_base = x_train_embed
        x_val_base = x_val_embed
        if use_financial and x_train_fin is not None and x_val_fin is not None:
            x_train_base = np.hstack([x_train_embed, x_train_fin])
            x_val_base = np.hstack([x_val_embed, x_val_fin])
        try:
            solver = "qn" if USE_CUML else "lbfgs"
            clf = _build_logreg(
                solver=solver,
                max_iter=2000,
                class_weight="balanced",
                n_jobs=cpu_jobs,
                verbose=1,
            )
            clf.fit(x_train_base, y_train)
        finally:
            heartbeat.set()
            heartbeat_thread.join(timeout=1.0)
        proba = _to_numpy(clf.predict_proba(x_val_base))
        base_metrics = compute_metrics(y_val, proba, classes)
        results.append(
            {
                "name": "LogReg_baseline",
                "val_accuracy": base_metrics["accuracy"],
                "val_f1_macro": base_metrics["f1_macro"],
                "val_official_f1": base_metrics["official_global_f1"],
            }
        )

    if results:
        results_df = pd.DataFrame(results).sort_values("val_official_f1", ascending=False)
        results_df.to_csv(results_path, index=False)
        print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
