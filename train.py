import argparse
import ast
import csv
import inspect
import json
import os
import platform
import random
import time
from pathlib import Path

import multiprocessing as mp
from queue import Empty
import queue

os.environ.pop("TORCH_LOGS", None)

import numpy as np
import pandas as pd
import torch
import importlib.util
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from umap import UMAP
from tqdm import tqdm

from metrics import compute_metrics
from models import MLPClassifier

USE_CUML = _cuml_available and torch.cuda.is_available()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and hard examples."""
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.weight, 
            label_smoothing=self.label_smoothing, reduction="none"
        )
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class EmbeddingDataset(Dataset):
    def __init__(self, x_embed, y, x_fin=None):
        self.x_embed = torch.as_tensor(x_embed, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.x_fin = torch.as_tensor(x_fin, dtype=torch.float32) if x_fin is not None else None

    def __len__(self):
        return self.x_embed.shape[0]

    def __getitem__(self, idx):
        if self.x_fin is None:
            return self.x_embed[idx], None, self.y[idx]
        return self.x_embed[idx], self.x_fin[idx], self.y[idx]


def collate_batch(batch):
    x_embed, x_fin, y = zip(*batch)
    x_embed = torch.stack(x_embed, dim=0)
    y = torch.stack(y, dim=0)
    if x_fin[0] is None:
        return x_embed, None, y
    return x_embed, torch.stack(x_fin, dim=0), y


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_accel_status():
    if USE_CUML:
        print("[gpu] cuML enabled for PCA/UMAP/LogReg")
    elif _cuml_available:
        print("[gpu] cuML installed but CUDA not available; using CPU PCA/UMAP/LogReg")
    else:
        print("[gpu] cuML not installed; using CPU PCA/UMAP/LogReg")


def resolve_cpu_count():
    return os.cpu_count() or 1


def configure_runtime(args, device):
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    if args.torch_interop_threads is not None:
        torch.set_num_interop_threads(args.torch_interop_threads)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(args.matmul_precision)


def triton_available():
    return importlib.util.find_spec("triton") is not None


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    required = ["X_train_embed", "X_val_embed", "y_train", "y_val"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing keys in npz: {missing}")

    payload = {k: data[k] for k in data.files}
    if "classes" in payload:
        payload["classes"] = payload["classes"].tolist()
    return payload


def build_financial_features(df_all):
    ratio_cols = ["net_profit_margin", "ebitda_margin", "asset_turnover"]
    df_all[ratio_cols] = df_all[ratio_cols].replace([np.inf, -np.inf, 0], np.nan)

    missing_flags = df_all[ratio_cols].isna().astype(int).add_prefix("missing_")
    df_all["country_code"] = df_all["country_code"].fillna("UNK").astype(str)

    country_medians = df_all.groupby("country_code")[ratio_cols].median()
    global_medians = df_all[ratio_cols].median()

    for col in ratio_cols:
        df_all[col] = df_all[col].fillna(df_all["country_code"].map(country_medians[col]))
        df_all[col] = df_all[col].fillna(global_medians[col])

    clip_bounds = df_all[ratio_cols].quantile([0.005, 0.995])
    for col in ratio_cols:
        lo, hi = clip_bounds.loc[0.005, col], clip_bounds.loc[0.995, col]
        df_all[col] = df_all[col].clip(lo, hi)

    country_center = df_all.groupby("country_code")[ratio_cols].transform("median")
    country_scale = df_all.groupby("country_code")[ratio_cols].transform("std").replace(0, np.nan)
    country_z = ((df_all[ratio_cols] - country_center) / country_scale).fillna(0.0)
    country_z = country_z.add_prefix("country_z_")

    country_rank = df_all.groupby("country_code")[ratio_cols].rank(pct=True).add_prefix("country_rank_")

    fin_feats = df_all[ratio_cols].copy()
    fin_feats["log1p_asset_turnover"] = np.log1p(np.clip(fin_feats["asset_turnover"], a_min=0, a_max=None))
    fin_feats["npm_x_turn"] = fin_feats["net_profit_margin"] * fin_feats["asset_turnover"]
    fin_feats["ebitda_x_turn"] = fin_feats["ebitda_margin"] * fin_feats["asset_turnover"]
    fin_feats["npm_x_ebitda"] = fin_feats["net_profit_margin"] * fin_feats["ebitda_margin"]

    country_dummies = pd.get_dummies(df_all["country_code"], prefix="country")
    financial_features = pd.concat(
        [
            fin_feats.reset_index(drop=True),
            country_dummies.reset_index(drop=True),
            missing_flags.reset_index(drop=True),
            country_z.reset_index(drop=True),
            country_rank.reset_index(drop=True),
        ],
        axis=1,
    )
    return financial_features.to_numpy(dtype=np.float32)


def load_from_pickles(path, use_financial, val_split, seed):
    data_path = Path(path)
    if data_path.is_dir():
        text_path = data_path / "df_train.pkl"
        fin_path = data_path / "df_financials_train.pkl"
        cache_path = data_path / "cached_dataset.npz"
    else:
        text_path = data_path
        fin_path = data_path.with_name("df_financials_train.pkl")
        cache_path = data_path.with_name("cached_dataset.npz")

    if cache_path.exists():
        print(f"Loading cached dataset from {cache_path}. Delete this file to re-process pickles.")
        return load_npz(cache_path)

    if not text_path.exists():
        raise FileNotFoundError(f"Missing text data: {text_path}")

    df_text = pd.read_pickle(text_path)
    if use_financial:
        if not fin_path.exists():
            raise FileNotFoundError(f"Missing financial data: {fin_path}")
        df_financials = pd.read_pickle(fin_path)
        df_all = df_text.merge(df_financials, on="id", how="inner", suffixes=("", "_fin"))
    else:
        df_all = df_text.copy()

    embeddings = np.vstack(df_all["business_description_embedding"].apply(ast.literal_eval).to_numpy()).astype(
        np.float32
    )
    labels = df_all["industry"].astype(str).to_numpy()
    encoder = LabelEncoder()
    y_all = encoder.fit_transform(labels)
    classes = encoder.classes_.tolist()

    indices = np.arange(len(df_all))
    idx_train, idx_val = train_test_split(
        indices,
        test_size=val_split,
        random_state=seed,
        stratify=y_all,
    )

    payload = {
        "X_train_embed": embeddings[idx_train],
        "X_val_embed": embeddings[idx_val],
        "y_train": y_all[idx_train],
        "y_val": y_all[idx_val],
        "classes": classes,
    }

    if use_financial:
        financial_features = build_financial_features(df_all)
        payload["financial_train"] = financial_features[idx_train]
        payload["financial_val"] = financial_features[idx_val]

    print(f"Saving processed dataset to {cache_path}...")
    np.savez(cache_path, **payload)
    return payload


def compute_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32)


def parse_hidden_dims(value):
    if not value:
        return []
    return [int(x) for x in value.split(",") if x.strip()]


def summarize_standardization(x):
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    avg_abs_mean = float(np.mean(np.abs(means)))
    avg_std = float(np.mean(stds))
    return avg_abs_mean, avg_std


def maybe_standardize(name, x_train, x_val, x_excl, force, mean_thresh=0.1, std_low=0.9, std_high=1.1):
    avg_abs_mean, avg_std = summarize_standardization(x_train)
    if force:
        print(
            f"Scaling check [{name}]: avg_abs_mean={avg_abs_mean:.4f}, avg_std={avg_std:.4f} "
            "-> forcing StandardScaler (--standardize)."
        )
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)
        if x_excl is not None:
            x_excl = scaler.transform(x_excl)
        return x_train, x_val, x_excl, True

    is_standardized = avg_abs_mean <= mean_thresh and std_low <= avg_std <= std_high
    if is_standardized:
        print(
            f"Scaling check [{name}]: avg_abs_mean={avg_abs_mean:.4f}, avg_std={avg_std:.4f} "
            "-> already standardized; skipping scaling."
        )
        return x_train, x_val, x_excl, False

    print(
        f"Scaling check [{name}]: avg_abs_mean={avg_abs_mean:.4f}, avg_std={avg_std:.4f} "
        "-> applying StandardScaler."
    )
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)
    if x_excl is not None:
        x_excl = scaler.transform(x_excl)
    return x_train, x_val, x_excl, True


def run_diagnostic_worker(queue, run_dir, seed, n_neighbors, min_dist, n_jobs, probe_kwargs, viz_max_points, probe_max_train, probe_max_val):
    print("Diagnostic worker started.")
    while True:
        try:
            task = queue.get(timeout=1.0)
        except Empty:
            continue
        
        if task is None:
            break
            
        cmd, payload = task
        if cmd == "viz":
            epoch, feats_val, y_val, class_names = payload
            print(f"[Worker] Processing UMAP for epoch {epoch}...")
            try:
                projections = compute_projections(
                    feats_val, random_state=seed, n_neighbors=n_neighbors, min_dist=min_dist, n_jobs=n_jobs
                )
                projections["labels"] = y_val
                proj_path = run_dir / "projections" / f"epoch_{epoch:03d}.npz"
                np.savez(proj_path, **projections)
            except Exception as e:
                print(f"[Worker] UMAP failed: {e}")

        elif cmd == "probe":
            epoch, feats_train, y_train, feats_val, y_val = payload
            print(f"[Worker] Processing Probes for epoch {epoch}...")
            try:
                probe_results = run_linear_probes(
                    feats_train, y_train, feats_val, y_val,
                    max_iter=probe_kwargs["max_iter"],
                    c_value=probe_kwargs["c_value"],
                    n_jobs=probe_kwargs["n_jobs"],
                    tol=probe_kwargs["tol"]
                )
                for result in probe_results:
                    result.update({"epoch": epoch})
                    append_csv(run_dir / "probes.csv", list(result.keys()), result)
            except Exception as e:
                print(f"[Worker] Probes failed: {e}")
                
    print("Diagnostic worker finished.")


def make_run_dir(base_dir):
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "projections").mkdir(parents=True, exist_ok=True)
    (run_dir / "landscapes").mkdir(parents=True, exist_ok=True)
    (run_dir / "frames").mkdir(parents=True, exist_ok=True)
    return run_dir


def append_csv(path, fieldnames, row):
    path = Path(path)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, use_amp, show_progress=True):
    model.train()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    iterator = tqdm(loader, desc="Training", leave=False) if show_progress else loader
    for x_embed, x_fin, y in iterator:
        x_embed = x_embed.to(device, non_blocking=use_amp)
        y = y.to(device, non_blocking=use_amp)
        if x_fin is not None:
            x_fin = x_fin.to(device, non_blocking=use_amp)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x_embed, x_fin)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x_embed.shape[0]
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_count += x_embed.shape[0]
    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def predict_proba(model, loader, criterion, device, use_amp, show_progress=True):
    model.eval()
    probs = []
    labels = []
    total_loss = 0.0
    total_count = 0
    iterator = tqdm(loader, desc="Inference", leave=False) if show_progress else loader
    for x_embed, x_fin, y in iterator:
        x_embed = x_embed.to(device, non_blocking=use_amp)
        y = y.to(device, non_blocking=use_amp)
        if x_fin is not None:
            x_fin = x_fin.to(device, non_blocking=use_amp)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x_embed, x_fin)
            loss = criterion(logits, y)
        proba = torch.softmax(logits, dim=1)
        probs.append(proba.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        total_loss += loss.item() * x_embed.shape[0]
        total_count += x_embed.shape[0]

    return (
        total_loss / max(total_count, 1),
        np.concatenate(probs, axis=0),
        np.concatenate(labels, axis=0),
    )


def gpu_batch_iterator(x_embed, x_fin, y, batch_size, shuffle=True):
    """Iterate over GPU tensors in batches without DataLoader overhead."""
    n = x_embed.shape[0]
    indices = torch.randperm(n, device=x_embed.device) if shuffle else torch.arange(n, device=x_embed.device)
    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        x_fin_batch = x_fin[idx] if x_fin is not None else None
        yield x_embed[idx], x_fin_batch, y[idx]


def train_one_epoch_gpu(model, x_embed, x_fin, y, optimizer, scaler, criterion, batch_size, use_amp):
    """GPU-optimized training - data already on GPU, no transfer overhead."""
    model.train()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    
    for x_batch, x_fin_batch, y_batch in gpu_batch_iterator(x_embed, x_fin, y, batch_size, shuffle=True):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=x_embed.device.type, enabled=use_amp):
            logits = model(x_batch, x_fin_batch)
            loss = criterion(logits, y_batch)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * x_batch.shape[0]
        total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
        total_count += x_batch.shape[0]
    
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


@torch.no_grad()
def predict_proba_gpu(model, x_embed, x_fin, y, criterion, batch_size, use_amp):
    """GPU-optimized inference - data already on GPU."""
    model.eval()
    probs = []
    total_loss = 0.0
    total_count = 0
    
    for x_batch, x_fin_batch, y_batch in gpu_batch_iterator(x_embed, x_fin, y, batch_size, shuffle=False):
        with torch.amp.autocast(device_type=x_embed.device.type, enabled=use_amp):
            logits = model(x_batch, x_fin_batch)
            loss = criterion(logits, y_batch)
        probs.append(torch.softmax(logits, dim=1))
        total_loss += loss.item() * x_batch.shape[0]
        total_count += x_batch.shape[0]
    
    all_probs = torch.cat(probs, dim=0).cpu().numpy()
    return total_loss / max(total_count, 1), all_probs, y.cpu().numpy()


@torch.no_grad()
def extract_features(model, x_embed, x_fin, device, batch_size, use_amp, feature_keys=None):
    model.eval()
    if feature_keys is None:
        feature_keys = ("input", "block1", "penultimate")
    feats = None
    for start in range(0, x_embed.shape[0], batch_size):
        end = start + batch_size
        xb = torch.as_tensor(x_embed[start:end], dtype=torch.float32).to(device, non_blocking=use_amp)
        if x_fin is not None:
            xf = torch.as_tensor(x_fin[start:end], dtype=torch.float32).to(device, non_blocking=use_amp)
        else:
            xf = None
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            _, feat = model(xb, xf, return_features=True)
        if feats is None:
            feats = {k: [] for k in feature_keys if k in feat}
        for key in feats:
            feats[key].append(feat[key].detach().cpu().numpy())
    if not feats:
        return {}
    return {k: np.concatenate(v, axis=0) for k, v in feats.items()}


@torch.no_grad()
def collect_logits_and_penultimate(model, x_embed, x_fin, device, batch_size, use_amp):
    model.eval()
    logits_list = []
    penultimate_list = []
    for start in range(0, x_embed.shape[0], batch_size):
        end = start + batch_size
        xb = torch.as_tensor(x_embed[start:end], dtype=torch.float32).to(device, non_blocking=use_amp)
        if x_fin is not None:
            xf = torch.as_tensor(x_fin[start:end], dtype=torch.float32).to(device, non_blocking=use_amp)
        else:
            xf = None
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, feats = model(xb, xf, return_features=True)
        logits_list.append(logits.detach().cpu().numpy())
        penultimate_list.append(feats["penultimate"].detach().cpu().numpy())
    return np.concatenate(logits_list, axis=0), np.concatenate(penultimate_list, axis=0)


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


def compute_projections(layer_feats, random_state, n_neighbors, min_dist, n_jobs):
    projections = {}
    for name, feats in layer_feats.items():
        pca_cls = cuPCA if USE_CUML else PCA
        pca_kwargs = {"n_components": 2, "random_state": random_state}
        pca = pca_cls(**_filter_kwargs(pca_cls, pca_kwargs)).fit_transform(feats)
        umap_cls = cuUMAP if USE_CUML else UMAP
        umap_state = random_state if (USE_CUML or n_jobs == 1) else None
        umap_kwargs = {
            "n_components": 2,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "random_state": umap_state,
            "n_jobs": n_jobs,
        }
        umap = umap_cls(**_filter_kwargs(umap_cls, umap_kwargs)).fit_transform(feats)
        projections[f"{name}_pca"] = _to_numpy(pca)
        projections[f"{name}_umap"] = _to_numpy(umap)
    return projections


def run_linear_probes(
    train_feats,
    y_train,
    val_feats,
    y_val,
    max_iter,
    c_value,
    n_jobs,
    tol,
    parallel_layers=False,
    parallel_jobs=None,
):
    results = []
    layer_names = list(train_feats.keys())
    if parallel_jobs is None:
        parallel_jobs = n_jobs
    solver = "qn" if USE_CUML else "sag"

    def _fit_layer(layer_name, lr_jobs, limit_threads):
        x_train = train_feats[layer_name]
        x_val = val_feats[layer_name]
        start = time.perf_counter()
        print(f"    Probe {layer_name}: fitting {x_train.shape[0]}x{x_train.shape[1]}...")
        logreg_cls = cuLogisticRegression if USE_CUML else LogisticRegression
        logreg_kwargs = dict(
            penalty="l2",
            solver=solver,
            C=c_value,
            max_iter=max_iter,
            tol=tol,
            n_jobs=lr_jobs,
        )
        clf = logreg_cls(**_filter_kwargs(logreg_cls, logreg_kwargs))
        try:
            with threadpool_limits(limits=limit_threads):
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    clf.fit(x_train, y_train)
        except ConvergenceWarning:
            retry_c = min(c_value, 0.2)
            retry_iter = max_iter * 2
            retry_tol = max(tol, 1e-2)
            print(
                f"    Probe {layer_name}: max_iter hit; retrying with "
                f"C={retry_c}, max_iter={retry_iter}, tol={retry_tol}"
            )
            retry_kwargs = dict(
                penalty="l2",
                solver=solver,
                C=retry_c,
                max_iter=retry_iter,
                tol=retry_tol,
                n_jobs=lr_jobs,
            )
            clf = logreg_cls(**_filter_kwargs(logreg_cls, retry_kwargs))
            try:
                with threadpool_limits(limits=limit_threads):
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        clf.fit(x_train, y_train)
            except ConvergenceWarning:
                print(f"    Probe {layer_name}: still not converged; using last iterate.")
        pred = _to_numpy(clf.predict(x_val))
        acc = (pred == y_val).mean()
        f1 = f1_score(y_val, pred, average="macro")
        elapsed = time.perf_counter() - start
        print(f"    Probe {layer_name}: done in {elapsed:.1f}s")
        return {"layer": layer_name, "probe_acc": float(acc), "probe_f1": float(f1)}

    if parallel_layers and parallel_jobs > 1 and len(layer_names) > 1:
        jobs = min(parallel_jobs, len(layer_names))
        results = Parallel(n_jobs=jobs, prefer="threads")(
            delayed(_fit_layer)(layer, 1, 1) for layer in layer_names
        )
    else:
        for layer in layer_names:
            results.append(_fit_layer(layer, n_jobs, n_jobs))

    return results


def make_directions(params, seed):
    torch.manual_seed(seed)
    dirs = [torch.randn_like(p) for p in params]
    norm = torch.sqrt(sum((d ** 2).sum() for d in dirs))
    return [d / (norm + 1e-12) for d in dirs]


def unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


@torch.inference_mode()
def compute_loss_landscape(model, criterion, x_batch, y_batch, dir1, dir2, alphas, betas, use_amp):
    model = unwrap_model(model)
    was_training = model.training
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    base_params = [p.detach().clone() for p in params]
    losses = np.zeros((len(alphas), len(betas)), dtype=np.float32)
    device_type = x_batch[0].device.type

    for i, alpha in enumerate(tqdm(alphas, desc="Landscape X", leave=False)):
        alpha_params = [base + alpha * d1 for base, d1 in zip(base_params, dir1)]
        for j, beta in enumerate(betas):
            for p, base_alpha, d2 in zip(params, alpha_params, dir2):
                p.copy_(base_alpha + beta * d2)
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                logits = model(x_batch[0], x_batch[1])
                loss = criterion(logits, y_batch)
            losses[i, j] = loss.item()

    for p, base in zip(params, base_params):
        p.copy_(base)
    if was_training:
        model.train()
    return losses


def save_snapshot(run_dir, epoch, history, proj_coords, labels):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax0, ax1 = axes

    ax0.plot(history["train_loss"], label="train loss")
    ax0.plot(history["val_loss"], label="val loss")
    if history.get("excl_loss"):
        ax0.plot(history["excl_loss"], label="excl loss")
    ax0.set_yscale("log")
    ax0.set_xlabel("epoch")
    ax0.set_title("Loss curves")
    ax0.legend(loc="upper right")

    ax1.scatter(
        proj_coords[:, 0],
        proj_coords[:, 1],
        c=labels,
        s=10,
        cmap="tab20",
        alpha=0.7,
        linewidths=0,
    )
    ax1.set_title("Penultimate projection")
    ax1.set_xlabel("dim 1")
    ax1.set_ylabel("dim 2")

    frame_path = Path(run_dir) / "frames" / f"epoch_{epoch:03d}.png"
    fig.tight_layout()
    fig.savefig(frame_path, dpi=150)
    plt.close(fig)


def export_mp4(frames_dir, output_path, fps):
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("imageio is required for MP4 export") from exc

    frames = sorted(Path(frames_dir).glob("epoch_*.png"))
    if not frames:
        return
    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(output_path, images, fps=fps)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP on embeddings with live artifacts.")
    parser.add_argument("--data", required=True, help="Path to .npz data file")
    parser.add_argument("--run-dir", default="runs", help="Base directory for run artifacts")
    parser.add_argument("--use-financial", action="store_true", help="Use financial features if present")
    parser.add_argument("--fusion", choices=["concat", "film"], default="film")
    parser.add_argument("--hidden-dims", default="2048,2048,2048", help="Comma-separated hidden sizes")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--activation", choices=["relu", "gelu"], default="relu")
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split for pickle input")
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce", help="Loss function: ce (CrossEntropy) or focal")
    parser.add_argument("--focal-gamma", type=float, default=1.5, help="Focal loss gamma parameter")
    parser.add_argument("--patience", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast-train", action="store_true", help="Fast training: disable progress bars, diagnostics, enable compile")
    parser.add_argument("--val-every", type=int, default=20, help="Validate every N epochs (default=1)")
    parser.add_argument("--full-gpu", action=argparse.BooleanOptionalAction, default=True, help="Load entire dataset to GPU (faster for small datasets)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers (default: all cores)",
    )
    parser.add_argument(
        "--cpu-jobs",
        type=int,
        default=None,
        help="Parallel CPU jobs for sklearn/UMAP (default: all cores)",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="torch.set_num_threads (default: all cores)",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=None,
        help="torch.set_num_interop_threads (default: min(4, cores))",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor per worker",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch.compile on CUDA",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TF32 on Ampere+ GPUs",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable cuDNN benchmark autotuner",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default="high",
    )
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument(
        "--diagnostics",
        choices=["live", "offline", "off"],
        default="offline",
        help="Diagnostics mode: live (background), offline (post-run), off",
    )
    parser.add_argument("--skip-viz", action="store_true", help="Skip saving visualization datapoints entirely")
    parser.add_argument(
        "--save-viz-every",
        type=int,
        default=0,  # default=1
        help="Save logits/penultimate for viz every N epochs (0=disable)",
    )
    parser.add_argument(
        "--save-epoch-every",
        type=int,
        default=0,
        help="Save epoch checkpoints every N epochs for offline diagnostics (0=auto)",
    )
    parser.add_argument("--viz-every", type=int, default=0)  # default=100
    parser.add_argument("--probe-every", type=int, default=0)  # default=100
    parser.add_argument("--landscape-every", type=int, default=0)  # default=100
    parser.add_argument("--viz-max-points", type=int, default=0)  # default=15000
    parser.add_argument("--probe-max-train", type=int, default=0)  # default=20000
    parser.add_argument("--probe-max-val", type=int, default=0)  # default=20000
    parser.add_argument("--probe-max-iter", type=int, default=0)  # default=5000
    parser.add_argument("--probe-tol", type=float, default=0)  # default=1e-2
    parser.add_argument("--probe-c", type=float, default=0)  # default=0.5
    parser.add_argument("--landscape-points", type=int, default=0)  # default=41
    parser.add_argument("--landscape-radius", type=float, default=0)  # default=1.5
    parser.add_argument("--export-mp4", action="store_false")  # action="store_true"
    parser.add_argument("--mp4-fps", type=int, default=0)  # default=5
    parser.add_argument("--skip-baseline", action="store_true")  # action="store_true"
    return parser.parse_args()


def main():
    args = parse_args()
    if args.compile and platform.system() == "Windows":
        if os.environ.get("FORCE_TORCH_COMPILE") == "1":
            print("FORCE_TORCH_COMPILE=1 set: attempting torch.compile on Windows.")
        else:
            print("torch.compile disabled on Windows due to known Triton/Inductor instability.")
            args.compile = False
    
    # Fast training mode: optimize for speed
    if args.fast_train:
        print("[fast-train] Enabling speed optimizations...")
        args.diagnostics = "off"
        args.skip_viz = True
        args.save_viz_every = 0
        args.save_epoch_every = 0
        args.compile = True  # Enable torch.compile on Linux/WSL
        if args.val_every == 1:
            args.val_every = 5  # Validate less frequently
        print(f"[fast-train] val_every={args.val_every}, compile={args.compile}, diagnostics=off")
    
    cpu_count = resolve_cpu_count()
    if args.num_workers is None:
        args.num_workers = cpu_count
    if args.cpu_jobs is None:
        args.cpu_jobs = cpu_count
    if args.torch_threads is None:
        args.torch_threads = cpu_count
    if args.torch_interop_threads is None:
        args.torch_interop_threads = max(1, min(4, cpu_count))
    args.num_workers = max(0, args.num_workers)
    args.cpu_jobs = max(1, args.cpu_jobs)
    args.prefetch_factor = max(1, args.prefetch_factor)
    if args.save_epoch_every <= 0:
        if args.diagnostics == "offline":
            args.save_epoch_every = max(1, args.viz_every)
        else:
            args.save_epoch_every = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA available: {cuda_available} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"CUDA available: {cuda_available} (using CPU)")
    log_accel_status()
    configure_runtime(args, device)
    set_seed(args.seed)

    data_path = Path(args.data)
    if data_path.suffix.lower() == ".npz":
        payload = load_npz(data_path)
    else:
        payload = load_from_pickles(data_path, args.use_financial, args.val_split, args.seed)
    x_train_embed = payload["X_train_embed"].astype(np.float32)
    x_val_embed = payload["X_val_embed"].astype(np.float32)
    y_train = payload["y_train"].astype(np.int64)
    y_val = payload["y_val"].astype(np.int64)
    classes = payload.get("classes") or list(range(int(y_train.max()) + 1))
    num_classes = len(classes)

    x_train_fin = payload.get("financial_train")
    x_val_fin = payload.get("financial_val")
    if args.use_financial:
        if x_train_fin is None or x_val_fin is None:
            raise ValueError("financial_train/financial_val missing in .npz")
        x_train_fin = x_train_fin.astype(np.float32)
        x_val_fin = x_val_fin.astype(np.float32)
    else:
        x_train_fin = None
        x_val_fin = None

    x_excl = payload.get("X_excl")
    y_excl = payload.get("y_excl")
    x_excl_fin = None
    use_excl = False
    if x_excl is not None and y_excl is not None:
        x_excl = x_excl.astype(np.float32)
        y_excl = y_excl.astype(np.int64)
        if args.use_financial and payload.get("financial_excl") is not None:
            x_excl_fin = payload["financial_excl"].astype(np.float32)
            use_excl = True
        elif not args.use_financial:
            use_excl = True
        else:
            print("Warning: X_excl provided without financial_excl; skipping excluded loss.")

    x_train_embed, x_val_embed, x_excl, _ = maybe_standardize(
        "embeddings",
        x_train_embed,
        x_val_embed,
        x_excl,
        force=args.standardize,
    )
    if args.use_financial:
        x_train_fin, x_val_fin, x_excl_fin, _ = maybe_standardize(
            "financial",
            x_train_fin,
            x_val_fin,
            x_excl_fin,
            force=args.standardize,
        )

    run_dir = make_run_dir(args.run_dir)
    config = vars(args).copy()
    config.update({"num_classes": num_classes, "classes": classes})
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    use_amp = device.type == "cuda"

    # Full GPU mode: load entire dataset to GPU for maximum speed
    if args.full_gpu and device.type == "cuda":
        print("[full-gpu] Loading entire dataset to GPU...")
        x_train_gpu = torch.as_tensor(x_train_embed, dtype=torch.float32, device=device)
        y_train_gpu = torch.as_tensor(y_train, dtype=torch.long, device=device)
        x_train_fin_gpu = torch.as_tensor(x_train_fin, dtype=torch.float32, device=device) if x_train_fin is not None else None
        x_val_gpu = torch.as_tensor(x_val_embed, dtype=torch.float32, device=device)
        y_val_gpu = torch.as_tensor(y_val, dtype=torch.long, device=device)
        x_val_fin_gpu = torch.as_tensor(x_val_fin, dtype=torch.float32, device=device) if x_val_fin is not None else None
        train_loader = None
        val_loader = None
        print(f"[full-gpu] Train: {x_train_gpu.shape}, Val: {x_val_gpu.shape}")
    else:
        x_train_gpu = x_val_gpu = y_train_gpu = y_val_gpu = x_train_fin_gpu = x_val_fin_gpu = None
        train_ds = EmbeddingDataset(x_train_embed, y_train, x_train_fin)
        val_ds = EmbeddingDataset(x_val_embed, y_val, x_val_fin)
        loader_kwargs = dict(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=use_amp,
            collate_fn=collate_batch,
            persistent_workers=args.num_workers > 0,
        )
        if args.num_workers > 0:
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    model = MLPClassifier(
        embed_dim=x_train_embed.shape[1],
        num_classes=num_classes,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
        activation=args.activation,
        layernorm=True,
        residual=args.residual,
        fusion=args.fusion,
        fin_dim=x_train_fin.shape[1] if args.use_financial else None,
    ).to(device)
    if args.compile and device.type == "cuda":
        if not triton_available():
            print("torch.compile disabled: Triton not available. Use --no-compile or install triton.")
        else:
            try:
                try:
                    import torch._inductor.config as inductor_config
                    import torch._inductor.select_algorithm as select_algorithm

                    inductor_config.autotune_num_choices_displayed = 0
                    select_algorithm.PRINT_AUTOTUNE = False
                except Exception:
                    pass
                model = torch.compile(model, mode="max-autotune")
            except Exception as exc:
                print(f"torch.compile failed: {exc}. Continuing without compile.")

    class_weights = compute_class_weights(y_train, num_classes).to(device)
    if args.loss == "focal":
        criterion = FocalLoss(weight=class_weights, gamma=args.focal_gamma, label_smoothing=0.1)
        print(f"Using FocalLoss with gamma={args.focal_gamma}")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=use_amp,
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    use_live_diag = args.diagnostics == "live"
    use_offline_diag = args.diagnostics == "offline"
    diag_queue = None
    diag_process = None
    if use_live_diag:
        diag_queue = mp.Queue()
        diag_process = mp.Process(
            target=run_diagnostic_worker,
            args=(
                diag_queue,
                run_dir,
                args.seed,
                15,
                0.1,
                args.cpu_jobs,
                {
                    "max_iter": args.probe_max_iter,
                    "c_value": args.probe_c,
                    "n_jobs": args.cpu_jobs,
                    "tol": args.probe_tol,
                },
                args.viz_max_points,
                args.probe_max_train,
                args.probe_max_val,
            ),
        )
        diag_process.start()
        print(f"Started diagnostic background process (PID: {diag_process.pid})")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "excl_loss": []}
    best_score = -1.0
    best_epoch = -1
    patience = 0

    rng = np.random.default_rng(args.seed)
    viz_size = max(1, min(args.viz_max_points, x_val_embed.shape[0])) if args.viz_max_points > 0 else 1
    viz_idx = rng.choice(x_val_embed.shape[0], size=viz_size, replace=False)
    probe_train_size = max(1, min(args.probe_max_train, x_train_embed.shape[0])) if args.probe_max_train > 0 else 1
    probe_train_idx = rng.choice(x_train_embed.shape[0], size=probe_train_size, replace=False)
    probe_val_size = max(1, min(args.probe_max_val, x_val_embed.shape[0])) if args.probe_max_val > 0 else 1
    probe_val_idx = rng.choice(x_val_embed.shape[0], size=probe_val_size, replace=False)
    land_idx = rng.choice(x_val_embed.shape[0], size=min(256, x_val_embed.shape[0]), replace=False)
    np.savez(
        run_dir / "diag_indices.npz",
        viz_idx=viz_idx,
        probe_train_idx=probe_train_idx,
        probe_val_idx=probe_val_idx,
        land_idx=land_idx,
    )

    land_batch = None
    land_y = None
    dir1 = None
    dir2 = None
    alphas = None
    betas = None
    if use_live_diag:
        land_x = torch.as_tensor(x_val_embed[land_idx], dtype=torch.float32, device=device)
        land_y = torch.as_tensor(y_val[land_idx], dtype=torch.long, device=device)
        land_x_fin = (
            torch.as_tensor(x_val_fin[land_idx], dtype=torch.float32, device=device) if args.use_financial else None
        )
        land_batch = (land_x, land_x_fin)
        dir1 = make_directions([p for p in model.parameters() if p.requires_grad], seed=args.seed + 1)
        dir2 = make_directions([p for p in model.parameters() if p.requires_grad], seed=args.seed + 2)
        alphas = np.linspace(-args.landscape_radius, args.landscape_radius, args.landscape_points)
        betas = np.linspace(-args.landscape_radius, args.landscape_radius, args.landscape_points)

    last_proj = None
    viz_cache_dir = run_dir / "viz_cache"
    viz_cache_dir.mkdir(parents=True, exist_ok=True)
    show_progress = not args.fast_train

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        # Use GPU-optimized functions if full-gpu mode
        if args.full_gpu and x_train_gpu is not None:
            train_loss, train_acc = train_one_epoch_gpu(
                model, x_train_gpu, x_train_fin_gpu, y_train_gpu,
                optimizer, scaler, criterion, args.batch_size, use_amp
            )
        else:
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device, use_amp, show_progress)
        
        scheduler.step()
        
        # Only validate every val_every epochs (or on last epoch)
        if epoch % args.val_every == 0 or epoch == args.epochs:
            if args.full_gpu and x_val_gpu is not None:
                val_loss, val_proba, y_val_true = predict_proba_gpu(
                    model, x_val_gpu, x_val_fin_gpu, y_val_gpu,
                    criterion, args.batch_size, use_amp
                )
            else:
                val_loss, val_proba, y_val_true = predict_proba(model, val_loader, criterion, device, use_amp, show_progress)
            metrics = compute_metrics(y_val_true, val_proba, classes)
        else:
            val_loss = history["val_loss"][-1] if history["val_loss"] else 0.0
            metrics = {"accuracy": 0, "f1_macro": 0, "official_global_f1": 0}
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": metrics["accuracy"],
            "val_f1_macro": metrics["f1_macro"],
            "val_official_f1": metrics["official_global_f1"],
        }

        if use_excl:
            excl_ds = EmbeddingDataset(x_excl, y_excl, x_excl_fin)
            excl_loader = DataLoader(
                excl_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=use_amp,
                collate_fn=collate_batch,
            )
            excl_loss, _, _ = predict_proba(model, excl_loader, criterion, device, use_amp)
            history["excl_loss"].append(excl_loss)
            row["excl_loss"] = excl_loss

        append_csv(run_dir / "metrics.csv", list(row.keys()), row)

        if not args.skip_viz and args.save_viz_every > 0 and epoch % args.save_viz_every == 0:
            logits_viz, penultimate_viz = collect_logits_and_penultimate(
                model,
                x_val_embed[viz_idx],
                x_val_fin[viz_idx] if args.use_financial else None,
                device,
                batch_size=args.batch_size,
                use_amp=use_amp,
            )
            np.savez(
                viz_cache_dir / f"epoch_{epoch:03d}.npz",
                logits_viz=logits_viz,
                penultimate_viz=penultimate_viz,
                labels=y_val[viz_idx],
                viz_idx=viz_idx,
            )

        if use_live_diag and epoch % args.viz_every == 0:
            print(f"[{epoch}] Queuing UMAP/PCA projections...")
            feats_val = extract_features(
                model,
                x_val_embed[viz_idx],
                x_val_fin[viz_idx] if args.use_financial else None,
                device,
                batch_size=args.batch_size,
                use_amp=use_amp,
            )
            diag_queue.put(("viz", (epoch, feats_val, y_val[viz_idx], classes)))

        if use_live_diag and epoch % args.probe_every == 0:
            print(f"[{epoch}] Queuing Linear Probes...")
            feats_train = extract_features(
                model,
                x_train_embed[probe_train_idx],
                x_train_fin[probe_train_idx] if args.use_financial else None,
                device,
                batch_size=args.batch_size,
                use_amp=use_amp,
            )
            feats_val = extract_features(
                model,
                x_val_embed[probe_val_idx],
                x_val_fin[probe_val_idx] if args.use_financial else None,
                device,
                batch_size=args.batch_size,
                use_amp=use_amp,
            )
            diag_queue.put(("probe", (epoch, feats_train, y_train[probe_train_idx], feats_val, y_val[probe_val_idx])))

        if use_live_diag and epoch % args.landscape_every == 0:
            print(f"[{epoch}] Computing Loss Landscape...")
            losses = compute_loss_landscape(
                model, criterion, land_batch, land_y, dir1, dir2, alphas, betas, use_amp
            )
            land_path = run_dir / "landscapes" / f"epoch_{epoch:03d}.npz"
            np.savez(land_path, alphas=alphas, betas=betas, loss=losses)

        if args.save_epoch_every > 0 and epoch % args.save_epoch_every == 0:
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch},
                run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
            )



        score = metrics["official_global_f1"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience = 0
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "score": best_score},
                run_dir / "checkpoints" / "best.pt",
            )
        else:
            patience += 1
            if patience >= args.patience:
                break

    if use_live_diag and diag_queue is not None and diag_process is not None:
        print("Stopping diagnostic worker...")
        diag_queue.put(None)
        diag_process.join()

    if args.export_mp4:
        export_mp4(run_dir / "frames", run_dir / "training.mp4", fps=args.mp4_fps)

    best_ckpt = run_dir / "checkpoints" / "best.pt"
    if best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

    val_loss, val_proba, y_val_true = predict_proba(model, val_loader, criterion, device, use_amp)
    best_metrics = compute_metrics(y_val_true, val_proba, classes)

    results = [
        {
            "name": "MLP",
            "val_accuracy": best_metrics["accuracy"],
            "val_f1_macro": best_metrics["f1_macro"],
            "val_official_f1": best_metrics["official_global_f1"],
        }
    ]

    try:
        df = pd.DataFrame(results).sort_values("val_official_f1", ascending=False)
        df.to_csv(run_dir / "results.csv", index=False)
        print(df.to_string(index=False))
    except Exception:
        for row in results:
            print(row)


if __name__ == "__main__":
    main()
