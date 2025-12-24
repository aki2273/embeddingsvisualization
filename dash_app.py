import argparse
import base64
import json
import re
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.model_selection import train_test_split
import dash
from dash import Dash, Input, Output, State, dcc, html
import torch
from torch import nn

GROK_BG = "#05070a"  # Deep clean background
GROK_PANEL = "rgba(13, 17, 26, 0.7)"  # Glassy panel
GROK_TEXT = "#e1e4e8"
GROK_MUTED = "#8b949e"
GROK_GRID = "#1f2430"
GROK_ACCENT = "#ff4b4b"
FONT_FAMILY = "'Space Grotesk', sans-serif"

# Set global Plotly template for consistent font across all graphs
pio.templates["grok"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT_FAMILY, color=GROK_TEXT),
        paper_bgcolor=GROK_BG,
        plot_bgcolor=GROK_BG,
        xaxis=dict(
            tickfont=dict(family=FONT_FAMILY),
            title_font=dict(family=FONT_FAMILY),
        ),
        yaxis=dict(
            tickfont=dict(family=FONT_FAMILY),
            title_font=dict(family=FONT_FAMILY),
        ),
        coloraxis=dict(
            colorbar=dict(
                tickfont=dict(family=FONT_FAMILY),
                title_font=dict(family=FONT_FAMILY),
            )
        ),
    )
)
pio.templates.default = "grok"

NEON_CYAN = "#00f0ff"
NEON_MAGENTA = "#ff00aa"
NEON_YELLOW = "#fcee0a"
NEON_BLUE = "#44aaff"
NEON_GREEN = "#0aff84"
NEON_TAN = "#e0c0a0"
NEON_RED = "#ff2a2a"

GROK_COLORWAY = [
    NEON_CYAN,
    NEON_MAGENTA,
    NEON_YELLOW,
    NEON_BLUE,
    NEON_GREEN,
    NEON_TAN,
    NEON_RED,
]

GROK_SEQ_SCALE = [
    [0.0, GROK_BG],
    [0.35, "#162032"],
    [0.6, "#2d4059"],
    [0.8, NEON_BLUE],
    [1.0, NEON_YELLOW],
]

PLAYGROUND_DATASETS = ["Moons", "Circles", "Spiral", "XOR", "Gaussian"]
PLAYGROUND_FEATURES = [
    "x1",
    "x2",
    "x1^2",
    "x2^2",
    "x1*x2",
    "sin(x1)",
    "sin(x2)",
]
PLAYGROUND_MAX_NODES = 8

EPOCH_RE = re.compile(r"epoch_(\d+)")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

:root {
    --bg-color: #05070a;
    --bg-gradient: radial-gradient(circle at 50% 0%, #1a1f2c 0%, #05070a 100%);
    --panel-bg: rgba(13, 17, 26, 0.6);
    --panel-border: 1px solid rgba(255, 255, 255, 0.08);
    --glass-blur: 20px;
    
    --text-primary: #e1e4e8;
    --text-secondary: #8b949e;
    
    --accent-red: #ff4b4b;
    --accent-blue: #44aaff;
    --accent-glow: 0 0 10px rgba(68, 170, 255, 0.3);
    
    --radius-lg: 16px;
    --radius-md: 8px;
    
    --transition-speed: 0.2s;
}

body {
    margin: 0;
    padding: 0;
    background: var(--bg-color);
    background-image: var(--bg-gradient);
    background-attachment: fixed;
    color: var(--text-primary);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px;
    overflow-x: hidden;
    -webkit-font-smoothing: antialiased;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.2);
}

.grok-app {
    display: flex;
    min-height: 100vh;
    width: 100%;
}

/* Glassy Sidebar */
.grok-sidebar {
    width: 340px;
    background: var(--panel-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border-right: var(--panel-border);
    display: flex;
    flex-direction: column;
    padding: 2rem 1.5rem;
    position: fixed;
    top: 0;
    left: 0;
    bottom: 0;
    overflow-y: auto;
    z-index: 100;
    box-shadow: 5px 0 30px rgba(0,0,0,0.3);
}

.grok-main {
    margin-left: 340px;
    flex-grow: 1;
    padding: 2.5rem 3rem 4rem 3rem; /* Symmetric padding-left and padding-right */
    position: relative;
    /* Ensure main content is above any background layers but below overlays */
    z-index: 1;
}

/* Title Styling */
.grok-title {
    margin: 0 0 2rem 0;
    padding: 0 0.5rem;
    font-weight: 700;
    font-size: 28px;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #fff 0%, #b2b9c6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}

h1, h2, h3, h4, h5 {
    color: var(--text-primary);
    letter-spacing: -0.01em;
    font-weight: 600;
}

/* Widget Groups */
.widget-group {
    margin-bottom: 24px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.widget-label {
    color: var(--text-secondary);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 8px;
    display: block;
    text-align: center; /* Center label text */
}

/* Customizing React Select (Dash Dropdowns) */
.Select-control {
    background-color: rgba(0, 0, 0, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    transition: all var(--transition-speed) ease;
}
.Select-control:hover {
    border-color: rgba(255, 255, 255, 0.2) !important;
    background-color: rgba(0, 0, 0, 0.6) !important;
}
.Select-menu-outer {
    background-color: #0f141e !important;
    border: 1px solid #283041 !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5) !important;
}
.Select-value-label {
    color: var(--text-primary) !important;
}
.Select-placeholder {
    color: var(--text-secondary) !important;
}
.Select-arrow {
    border-color: var(--text-secondary) transparent transparent !important;
}

/* Radio Items */
.grok-radio .grok-radio-item {
    display: block;
    padding: 10px 14px;
    margin-bottom: 6px;
    border-radius: var(--radius-md);
    cursor: pointer;
    color: var(--text-secondary);
    transition: all var(--transition-speed);
    border: 1px solid transparent;
}
.grok-radio .grok-radio-item:hover {
    background: rgba(255,255,255,0.05);
    color: var(--text-primary);
}
.grok-radio input:checked + label {
    color: var(--accent-blue);
    font-weight: 600;
}

/* Buttons */
.grok-button {
    background: linear-gradient(180deg, rgba(30,35,45,1) 0%, rgba(20,25,35,1) 100%);
    color: var(--text-primary);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: var(--radius-md);
    padding: 10px 20px;
    cursor: pointer;
    width: 100%;
    margin-bottom: 10px;
    font-family: inherit;
    font-weight: 500;
    transition: all var(--transition-speed);
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    text-align: center;
}
.grok-button:hover {
    border-color: var(--accent-blue);
    box-shadow: var(--accent-glow);
    transform: translateY(-1px);
}
.grok-button:active {
    transform: translateY(1px);
}

.grok-status {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 1rem;
    padding-left: 0.5rem;
    border-left: 2px solid var(--accent-blue);
    line-height: 1.4;
}

/* Glassy Cards */
.grok-card {
    background: var(--panel-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: var(--panel-border);
    border-radius: var(--radius-lg);
    padding: 20px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    transition: transform 0.3s ease;
}
.grok-card:hover {
    border-color: rgba(255,255,255,0.15);
}

/* Controls inside cards */
.control-row {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: flex-end;
    justify-content: center; /* Center items */
    margin-bottom: 20px;
    background: rgba(0,0,0,0.2);
    padding: 16px;
    border-radius: var(--radius-md);
}
.control-group {
    min-width: 160px;
    max-width: 400px; /* Prevent excessive stretching */
    flex: 1;
}

/* Sliders */
.rc-slider-rail {
    background-color: rgba(255,255,255,0.1) !important;
}
.rc-slider-track {
    background-color: var(--accent-blue) !important;
}
.rc-slider-handle {
    border: 2px solid var(--accent-blue) !important;
    background-color: #fff !important;
    box-shadow: var(--accent-glow) !important;
    opacity: 1 !important;
}

/* Plotly Tweaks */
.js-plotly-plot .plotly .modebar {
    left: 50% !important;
    transform: translateX(-50%);
}

/* Tabs Styling */
._dash-tab-content {
    margin-top: 20px;
    background: transparent !important;
    border: none !important;
}
.tab-parent {
    margin-bottom: 20px;
}
.grok-tabs > div:first-of-type {
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
    display: flex !important;
    flex-wrap: nowrap !important;
    align-items: stretch;
    gap: 6px;
    overflow-x: auto !important;
    overflow-y: hidden;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: thin;
    white-space: nowrap !important;
}
.grok-tabs .tab {
    background-color: transparent !important;
    border: 1px solid transparent !important;
    border-bottom: 1px solid transparent !important;
    color: var(--text-secondary) !important;
    padding: 8px 16px !important;
    transition: all 0.2s;
    font-weight: 500;
    display: flex !important;
    align-items: center;
    justify-content: center;
    height: 52px;
    min-height: 52px;
    line-height: 1.2;
    text-align: center;
    white-space: nowrap !important;
    flex: 0 0 auto !important;
    min-width: fit-content;
}
.grok-tabs .tab > span,
.grok-tabs .tab > div {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    height: 100% !important;
    width: 100% !important;
    white-space: inherit !important;
}
.grok-tabs .tab--selected {
    background-color: rgba(68, 170, 255, 0.1) !important;
    border: 1px solid var(--accent-blue) !important;
    border-bottom: 1px solid transparent !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-md) var(--radius-md) 0 0;
}
.grok-tabs .tab:hover {
    color: var(--text-primary) !important;
    background-color: rgba(255,255,255,0.05) !important;
}

/* Dash-specific tab container overrides */
.grok-tabs .tab-container,
.grok-tabs [role="tablist"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    white-space: nowrap !important;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: thin;
    gap: 6px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
}
/* Force horizontal row layout even on narrow screens */
.grok-tabs {
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.grok-card {
    animation: fadeIn 0.4s ease-out forwards;
}

/* Collapsible Sidebar Additions */
.grok-sidebar {
    transition: transform var(--transition-speed) ease-in-out;
}
.grok-sidebar.collapsed {
    transform: translateX(-100%);
    box-shadow: none;
}
.grok-main {
    transition: margin-left var(--transition-speed) ease-in-out;
    max-width: 100vw;
    overflow-x: hidden;
}
.grok-main.collapsed {
    margin-left: 0;
}
.sidebar-toggle {
    position: fixed;
    top: 20px;
    left: 20px;
    z-index: 200;
    background: var(--panel-bg);
    border: var(--panel-border);
    color: var(--text-primary);
    border-radius: var(--radius-md);
    padding: 8px 12px;
    cursor: pointer;
    backdrop-filter: blur(var(--glass-blur));
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.sidebar-toggle:hover {
    background: rgba(255,255,255,0.1);
    color: var(--accent-blue);
}
</style>
"""

INDEX_STRING = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        __CUSTOM_CSS__
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


def list_runs(runs_dir):
    if not runs_dir.exists():
        return []
    return sorted([p.name for p in runs_dir.iterdir() if p.is_dir()])


def list_epochs(folder, pattern="epoch_*.npz"):
    if folder is None or not folder.exists():
        return []
    epochs = []
    for path in folder.glob(pattern):
        match = EPOCH_RE.search(path.name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(set(epochs))


def load_config(run_dir):
    if run_dir is None:
        return {}
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return {}


def safe_read_npz(path):
    if path is None or not path.exists():
        return None, "File not found."
    try:
        with np.load(path) as data:
            return {k: data[k] for k in data.files}, None
    except Exception:
        return None, "File is updating; retrying on next refresh."


def safe_read_csv(path):
    if path is None or not path.exists():
        return None, "File not found."
    try:
        return pd.read_csv(path), None
    except Exception:
        return None, "File is updating; retrying on next refresh."


def map_labels(labels, class_names):
    if class_names:
        mapped = []
        for i in labels:
            idx = int(i)
            if 0 <= idx < len(class_names):
                mapped.append(class_names[idx])
            else:
                mapped.append(str(idx))
        return np.array(mapped)
    return labels.astype(str)


def build_color_map(items):
    items = [str(item) for item in items]
    return {item: GROK_COLORWAY[i % len(GROK_COLORWAY)] for i, item in enumerate(items)}


def _apply_vertical_legend_right(fig, width_px=260):
    fig.update_layout(
        legend=dict(
            orientation="v",
            x=1.02,
            xanchor="left",
            y=1.0,
            yanchor="top",
            font=dict(color=GROK_TEXT, family=FONT_FAMILY, size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=40, r=width_px, t=40, b=40),
    )
    return fig


def surface_metric(proba, metric, class_idx=None):
    if metric == "Confidence (max prob)":
        return proba.max(axis=2)
    if metric == "Entropy":
        clipped = np.clip(proba, 1e-8, 1.0)
        return -(clipped * np.log(clipped)).sum(axis=2)
    if metric == "Class probability" and class_idx is not None:
        return proba[:, :, class_idx]
    return proba.max(axis=2)


def select_filmstrip_epochs(available, count, stride, anchor):
    if not available:
        return []
    if anchor == "Latest":
        step = max(1, stride)
        picked = available[::-step][:count]
        return list(reversed(picked))
    sampled = available[:: max(1, stride)]
    return sampled[:count]


def _normalize_playground_data(x):
    x = x - x.mean(axis=0, keepdims=True)
    span = np.max(np.abs(x)) if x.size else 1.0
    if span <= 0:
        span = 1.0
    return (x / span).astype(np.float32)


def _make_spiral(n_points, noise, seed):
    rng = np.random.default_rng(seed)
    n0 = max(2, int(n_points // 2))
    n1 = max(2, int(n_points - n0))

    def _arm(n, phase):
        t = np.linspace(0.0, 1.0, n)
        angle = 4 * np.pi * t + phase
        radius = t
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        pts = np.stack([x, y], axis=1)
        if noise > 0:
            pts += rng.normal(scale=noise, size=pts.shape)
        return pts

    x0 = _arm(n0, 0.0)
    x1 = _arm(n1, np.pi)
    x = np.vstack([x0, x1]).astype(np.float32)
    y = np.concatenate([np.zeros(n0, dtype=np.int64), np.ones(n1, dtype=np.int64)])
    return _normalize_playground_data(x), y


def _make_xor(n_points, noise, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n_points, 2)).astype(np.float32)
    if noise > 0:
        x += rng.normal(scale=noise, size=x.shape).astype(np.float32)
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(np.int64)
    return _normalize_playground_data(x), y


def build_playground_dataset(kind, n_points, noise, seed):
    n_points = max(50, int(n_points or 400))
    noise = float(noise or 0.0)
    seed = int(seed or 0)
    if kind == "Moons":
        x, y = make_moons(n_samples=n_points, noise=noise, random_state=seed)
    elif kind == "Circles":
        x, y = make_circles(
            n_samples=n_points,
            noise=noise,
            factor=0.5,
            random_state=seed,
        )
    elif kind == "Gaussian":
        std = max(0.05, noise)
        x, y = make_blobs(
            n_samples=n_points,
            centers=[(-1.0, -1.0), (1.0, 1.0)],
            cluster_std=std,
            random_state=seed,
        )
    elif kind == "XOR":
        return _make_xor(n_points, noise, seed)
    else:
        return _make_spiral(n_points, noise, seed)
    return _normalize_playground_data(x), y.astype(np.int64)


def compute_playground_features(x, selected):
    if isinstance(selected, str):
        selected = [selected]
    if not selected:
        selected = ["x1", "x2"]
    x1 = x[:, 0]
    x2 = x[:, 1]
    mapping = {
        "x1": x1,
        "x2": x2,
        "x1^2": x1**2,
        "x2^2": x2**2,
        "x1*x2": x1 * x2,
        "sin(x1)": np.sin(np.pi * x1),
        "sin(x2)": np.sin(np.pi * x2),
    }
    feats = [mapping[name] for name in selected if name in mapping]
    if not feats:
        feats = [x1, x2]
    return np.stack(feats, axis=1).astype(np.float32)


def resolve_playground_layers(layer_count, l1, l2, l3, l4):
    count = max(1, int(layer_count or 1))
    sizes = [l1, l2, l3, l4]
    resolved = []
    for idx in range(min(count, len(sizes))):
        resolved.append(max(1, int(sizes[idx] or 1)))
    return resolved


def build_playground_model(input_dim, layer_sizes, activation):
    act_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
    act_cls = act_map.get(activation, nn.Tanh)
    layers = []
    in_dim = input_dim
    for width in layer_sizes:
        layers.append(nn.Linear(in_dim, int(width)))
        layers.append(act_cls())
        in_dim = int(width)
    layers.append(nn.Linear(in_dim, 2))
    return nn.Sequential(*layers)


def serialize_state_dict(state_dict):
    return {k: v.detach().cpu().numpy().tolist() for k, v in state_dict.items()}


def deserialize_state_dict(state_dict):
    restored = {}
    for key, value in (state_dict or {}).items():
        restored[key] = torch.tensor(np.array(value), dtype=torch.float32)
    return restored


def init_playground_state(config_key, seed, input_dim, layer_sizes, activation):
    torch.manual_seed(int(seed or 0))
    model = build_playground_model(input_dim, layer_sizes, activation)
    return {
        "config_key": config_key,
        "state_dict": serialize_state_dict(model.state_dict()),
        "step": 0,
        "loss_history": [],
        "acc_history": [],
        "val_acc_history": [],
    }


def train_playground_steps(
    model,
    state,
    x_train,
    y_train,
    x_val,
    y_val,
    lr,
    l2,
    batch_size,
    steps,
    seed,
):
    if x_train is None or x_train.size == 0:
        return state, model
    criterion = nn.CrossEntropyLoss()
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    x_val_t = torch.tensor(x_val, dtype=torch.float32) if x_val is not None else None
    y_val_t = torch.tensor(y_val, dtype=torch.long) if y_val is not None else None

    n_train = x_train_t.shape[0]
    batch = min(max(1, int(batch_size or n_train)), n_train)
    rng = np.random.default_rng(int(seed or 0) + int(state.get("step", 0)))

    for _ in range(max(1, int(steps or 1))):
        if batch >= n_train:
            idx = np.arange(n_train)
        else:
            idx = rng.choice(n_train, size=batch, replace=False)
        logits = model(x_train_t[idx])
        loss = criterion(logits, y_train_t[idx])
        if l2 and l2 > 0:
            l2_term = 0.0
            for p in model.parameters():
                l2_term += (p**2).sum()
            loss = loss + float(l2) * l2_term

        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= float(lr) * p.grad
        model.zero_grad(set_to_none=True)

        with torch.no_grad():
            train_logits = model(x_train_t)
            train_acc = (train_logits.argmax(dim=1) == y_train_t).float().mean().item()
            if x_val_t is not None and x_val_t.numel() > 0:
                val_logits = model(x_val_t)
                val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()
            else:
                val_acc = float("nan")

        state["loss_history"].append(float(loss.item()))
        state["acc_history"].append(float(train_acc))
        state["val_acc_history"].append(float(val_acc))
        state["step"] = int(state.get("step", 0)) + 1

    max_history = 400
    for key in ("loss_history", "acc_history", "val_acc_history"):
        if len(state.get(key, [])) > max_history:
            state[key] = state[key][-max_history:]

    state["state_dict"] = serialize_state_dict(model.state_dict())
    return state, model


def build_playground_boundary_figure(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    features,
    grid_res=140,
):
    if x_train is None or x_train.size == 0:
        return empty_figure("Playground data not available.", height=420)

    x_all = x_train if x_val is None or x_val.size == 0 else np.vstack([x_train, x_val])
    x_min, x_max = x_all[:, 0].min(), x_all[:, 0].max()
    y_min, y_max = x_all[:, 1].min(), x_all[:, 1].max()
    pad_x = (x_max - x_min) * 0.15 + 1e-3
    pad_y = (y_max - y_min) * 0.15 + 1e-3
    grid_x = np.linspace(x_min - pad_x, x_max + pad_x, int(grid_res))
    grid_y = np.linspace(y_min - pad_y, y_max + pad_y, int(grid_res))
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    grid = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1).astype(np.float32)
    grid_feats = compute_playground_features(grid, features)

    with torch.no_grad():
        logits = model(torch.tensor(grid_feats, dtype=torch.float32))
        proba = torch.softmax(logits, dim=1)[:, 1].numpy().reshape(mesh_x.shape)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=grid_x,
            y=grid_y,
            z=proba,
            colorscale=GROK_SEQ_SCALE,
            showscale=False,
            opacity=0.85,
        )
    )
    fig.add_trace(
        go.Contour(
            x=grid_x,
            y=grid_y,
            z=proba,
            contours=dict(start=0.5, end=0.5, size=0.5),
            line=dict(color=GROK_TEXT, width=1),
            showscale=False,
            hoverinfo="skip",
        )
    )

    class_colors = {0: NEON_CYAN, 1: NEON_MAGENTA}
    for split_name, xs, ys, symbol in (
        ("train", x_train, y_train, "circle"),
        ("val", x_val, y_val, "x"),
    ):
        if xs is None or xs.size == 0:
            continue
        for label in np.unique(ys):
            mask = ys == label
            fig.add_trace(
                go.Scatter(
                    x=xs[mask, 0],
                    y=xs[mask, 1],
                    mode="markers",
                    name=f"{split_name} class {label}",
                    marker=dict(
                        color=class_colors.get(int(label), NEON_YELLOW),
                        size=6,
                        symbol=symbol,
                        line=dict(width=0),
                    ),
                )
            )

    apply_grok_layout(fig, height=420, showlegend=True)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=0.0, xanchor="left"))
    return fig


def build_playground_loss_figure(state):
    if not state or not state.get("loss_history"):
        return empty_figure("Click Train to start.", height=320)

    steps = list(range(1, len(state["loss_history"]) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=state["loss_history"],
            name="loss",
            line=dict(color=NEON_YELLOW, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=state.get("acc_history", []),
            name="train acc",
            line=dict(color=NEON_CYAN, width=2),
            yaxis="y2",
        )
    )
    val_acc = state.get("val_acc_history", [])
    if any(np.isfinite(val_acc)):
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=val_acc,
                name="val acc",
                line=dict(color=NEON_MAGENTA, width=2),
                yaxis="y2",
            )
        )

    apply_grok_layout(fig, height=320, showlegend=True)
    fig.update_layout(
        yaxis2=dict(
            overlaying="y",
            side="right",
            range=[0, 1],
            showgrid=False,
            tickfont=dict(color=GROK_MUTED, family=FONT_FAMILY, size=11),
        )
    )
    return fig


def encode_image(path):
    if path is None or not path.exists():
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None


def read_metrics(runs_dir, run_name):
    if not run_name:
        return None, None, "Select a run to view metrics."
    metrics_path = runs_dir / run_name / "metrics.csv"
    df, message = safe_read_csv(metrics_path)
    if df is None:
        return None, None, message or "metrics.csv not found for this run."
    mtime = metrics_path.stat().st_mtime
    return df, mtime, None


def compute_zoom_ranges(coords, percentile=99):
    """Compute axis ranges to clip outliers based on percentile."""
    if coords is None or len(coords) == 0:
        return None
        
    # Calculate percentiles
    x = coords[:, 0]
    y = coords[:, 1]
    
    lower = (100 - percentile) / 2.0
    upper = 100 - lower
    
    x_min, x_max = np.percentile(x, [lower, upper])
    y_min, y_max = np.percentile(y, [lower, upper])
    
    # Add small padding
    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)
    pad_x = span_x * 0.05
    pad_y = span_y * 0.05
    
    return [x_min - pad_x, x_max + pad_x], [y_min - pad_y, y_max + pad_y]


def compute_component_ranges(coords, percentile=99):
    """Compute per-dimension ranges for outlier clipping."""
    if coords is None or len(coords) == 0:
        return None
    coords = np.asarray(coords)
    if coords.ndim != 2:
        return None
    lower = (100 - percentile) / 2.0
    upper = 100 - lower
    ranges = []
    for idx in range(coords.shape[1]):
        vals = coords[:, idx]
        lo, hi = np.percentile(vals, [lower, upper])
        span = max(hi - lo, 1e-6)
        pad = span * 0.05
        ranges.append([lo - pad, hi + pad])
    return ranges


def build_loss_figure(metrics_df):
    fig = go.Figure()
    if metrics_df is None or metrics_df.empty:
        fig.add_annotation(
            text="metrics.csv not available yet.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=GROK_MUTED, size=16, family=FONT_FAMILY),
        )
    else:
        epoch = (
            metrics_df["epoch"]
            if "epoch" in metrics_df.columns
            else list(range(1, len(metrics_df) + 1))
        )
        if "train_loss" in metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=epoch,
                    y=metrics_df["train_loss"],
                    name="train loss",
                    line=dict(width=2),
                )
            )
        if "val_loss" in metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=epoch,
                    y=metrics_df["val_loss"],
                    name="val loss",
                    line=dict(width=2),
                )
            )
        if "excl_loss" in metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=epoch,
                    y=metrics_df["excl_loss"],
                    name="excl loss",
                    line=dict(width=2),
                )
            )
        fig.update_yaxes(type="log")
    fig.update_layout(
        paper_bgcolor=GROK_BG,
        plot_bgcolor=GROK_BG,
        font=dict(color=GROK_TEXT, family=FONT_FAMILY, size=13),
        margin=dict(l=40, r=20, t=40, b=40),
        height=420,
        legend=dict(
            orientation="h",
            x=0.0,
            xanchor="left",
            y=1.1,
            yanchor="bottom",
            font=dict(color=GROK_TEXT, family=FONT_FAMILY, size=12),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=GROK_GRID,
            zeroline=False,
            tickfont=dict(color=GROK_MUTED, family=FONT_FAMILY, size=12),
            title_font=dict(color=GROK_TEXT, family=FONT_FAMILY),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GROK_GRID,
            zeroline=False,
            tickfont=dict(color=GROK_MUTED, family=FONT_FAMILY, size=12),
            title_font=dict(color=GROK_TEXT, family=FONT_FAMILY),
        ),
        colorway=GROK_COLORWAY,
    )
    return fig



def build_metrics_figure(metrics_df):
    fig = go.Figure()
    if metrics_df is None or metrics_df.empty:
        fig.add_annotation(
            text="metrics.csv not available yet.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color=GROK_MUTED, size=16, family=FONT_FAMILY),
        )
    else:
        epoch = (
            metrics_df["epoch"]
            if "epoch" in metrics_df.columns
            else list(range(1, len(metrics_df) + 1))
        )
        for col in ["val_accuracy", "val_f1_macro", "val_official_f1"]:
            if col in metrics_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=epoch,
                        y=metrics_df[col],
                        name=col,
                        line=dict(width=2),
                    )
                )
    return apply_grok_layout(fig, height=420)


def build_interpolation_figure(run_dir, epoch, pair_idx):
    if not run_dir or epoch is None:
        return empty_figure("Select a run and epoch.")
    
    interp_dir = run_dir / "interpolation"
    path = interp_dir / f"epoch_{epoch:03d}.npz"
    data, msg = safe_read_npz(path)
    
    if data is None:
        return empty_figure(msg or "Interpolation file not found.")
        
    t = data["t"]
    proba_a = data["proba_a"]
    proba_b = data["proba_b"]
    pair_labels = data["pair_labels"]
    pair_types = data["pair_types"]
    
    if proba_a.shape[0] == 0:
        return empty_figure("No pairs available.")
        
    if pair_idx >= proba_a.shape[0]:
        pair_idx = 0
        
    config = load_config(run_dir)
    class_names = config.get("classes", [])
    
    label_a, label_b = pair_labels[pair_idx]
    name_a = class_names[int(label_a)] if class_names else str(label_a)
    name_b = class_names[int(label_b)] if class_names else str(label_b)
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=proba_a[pair_idx],
            name=f"P({name_a})",
            mode="lines",
            line=dict(color=NEON_YELLOW, width=3, shape="spline"),
        )
    )
    if label_a != label_b:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=proba_b[pair_idx],
                name=f"P({name_b})",
                mode="lines",
                line=dict(color=NEON_MAGENTA, width=3, shape="spline"),
            )
        )
        
    kind = "same-class" if pair_types[pair_idx] == 0 else "different-class"
    fig.update_layout(
        title=dict(
            text=f"Pair {pair_idx} ({kind}): {name_a} -> {name_b}",
            font=dict(size=14, color=GROK_TEXT, family=FONT_FAMILY)
        )
    )
    return apply_grok_layout(fig, height=350)


def build_neuron_figure(run_dir, epoch, unit_idx, view_mode="Single unit"):
    if not run_dir or epoch is None:
        return empty_figure("Select a run and epoch.")

    resp_dir = run_dir / "neuron_responses"
    path = resp_dir / f"epoch_{epoch:03d}.npz"
    data, msg = safe_read_npz(path)
    
    if data is None:
        return empty_figure(msg or "Response file not found.")

    pc1 = data["pc1"]
    activations = data["activations"]
    unit_indices = data["unit_indices"].astype(int)
    labels_raw = data["labels"]
    
    config = load_config(run_dir)
    class_names = config.get("classes", [])
    labels_named = map_labels(labels_raw, class_names)
    label_names = class_names if class_names else sorted(set(labels_named.tolist()))
    class_color_map = build_color_map(label_names)
    
    if unit_idx not in unit_indices:
        unit_idx = unit_indices[0] if len(unit_indices) > 0 else 0
        
    try:
        unit_pos = list(unit_indices).index(unit_idx)
    except ValueError:
        return empty_figure("Unit not found in data.")

    fig = px.scatter(
        x=pc1,
        y=activations[:, unit_pos],
        color=labels_named,
        color_discrete_map=class_color_map,
        opacity=0.85,
        render_mode="webgl",
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig.update_layout(title=f"Unit {unit_idx} Response")
    
    return apply_grok_layout(fig, height=400)


def build_loss_landscape_figure(run_dir, epoch, view_mode="3D"):
    if not run_dir or epoch is None:
        return empty_figure("Select a run and epoch.")

    land_dir = run_dir / "loss_landscape"
    path = land_dir / f"epoch_{epoch:03d}.npz"
    data, msg = safe_read_npz(path)
    
    if data is None:
        return empty_figure(msg or "Landscape file not found.")

    alphas = data["alphas"]
    betas = data["betas"]
    loss = data["loss"]

    if view_mode == "3D":
        x_grid, y_grid = np.meshgrid(betas, alphas)
        fig = go.Figure(
            data=[
                go.Surface(
                    x=x_grid,
                    y=y_grid,
                    z=loss,
                    colorscale=GROK_SEQ_SCALE,
                )
            ]
        )
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    backgroundcolor=GROK_BG,
                    gridcolor=GROK_GRID,
                    title="Beta",
                    tickfont=dict(color=GROK_MUTED, family=FONT_FAMILY, size=12),
                    titlefont=dict(color=GROK_TEXT, family=FONT_FAMILY),
                ),
                yaxis=dict(
                    backgroundcolor=GROK_BG,
                    gridcolor=GROK_GRID,
                    title="Alpha",
                    tickfont=dict(color=GROK_MUTED, family=FONT_FAMILY, size=12),
                    titlefont=dict(color=GROK_TEXT, family=FONT_FAMILY),
                ),
                zaxis=dict(
                    backgroundcolor=GROK_BG,
                    gridcolor=GROK_GRID,
                    title="Loss",
                    tickfont=dict(color=GROK_MUTED, family=FONT_FAMILY, size=12),
                    titlefont=dict(color=GROK_TEXT, family=FONT_FAMILY),
                ),
            ),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        # Apply themes manually for 3D scene since apply_grok_layout is mostly 2D
        fig.update_layout(
            paper_bgcolor=GROK_BG,
            font=dict(color=GROK_TEXT, family=FONT_FAMILY),
        )
        return fig
    else:
        fig = px.imshow(
            loss,
            x=betas,
            y=alphas,
            aspect="auto",
            origin="lower",
            color_continuous_scale=GROK_SEQ_SCALE,
            labels=dict(x="Beta", y="Alpha", color="Loss")
        )
        return apply_grok_layout(fig, height=500)


def apply_grok_layout(fig, height=420, showlegend=True):
    top_margin = 80 if showlegend else 40
    fig.update_layout(
        paper_bgcolor=GROK_BG,
        plot_bgcolor=GROK_BG,
        font=dict(color=GROK_TEXT, family=FONT_FAMILY, size=13),
        margin=dict(l=40, r=20, t=top_margin, b=40),
        height=height,
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            x=0.0,
            xanchor="left",
            y=1.12,
            yanchor="bottom",
            font=dict(color=GROK_TEXT, family=FONT_FAMILY, size=11),
            bgcolor="rgba(0,0,0,0)",
        )
        if showlegend
        else None,
        colorway=GROK_COLORWAY,
    )
    # Apply axis styling to *all* subplot axes (xaxis2, yaxis3, ...)
    fig.update_xaxes(
        showgrid=True,
        gridcolor=GROK_GRID,
        zeroline=False,
        tickfont=dict(color=GROK_MUTED, family=FONT_FAMILY, size=12),
        title_font=dict(color=GROK_TEXT, family=FONT_FAMILY),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GROK_GRID,
        zeroline=False,
        tickfont=dict(color=GROK_MUTED, family=FONT_FAMILY, size=12),
        title_font=dict(color=GROK_TEXT, family=FONT_FAMILY),
    )
    return fig


def empty_figure(message, height=420):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(color=GROK_MUTED, size=16, family=FONT_FAMILY),
    )
    return apply_grok_layout(fig, height=height, showlegend=False)


def build_playground_tab():
    dataset_options = [{"label": name, "value": name} for name in PLAYGROUND_DATASETS]
    feature_options = [
        {"label": f" {name}", "value": name} for name in PLAYGROUND_FEATURES
    ]
    lr_options = [0.0005, 0.001, 0.003, 0.01, 0.03, 0.1]
    batch_options = [16, 32, 64, 128, 256]

    return dcc.Tab(
        label="Neural Playground",
        value="playground",
        children=[
            dcc.Store(id="playground-state"),
            dcc.Store(id="playground-play", data=False),
            dcc.Interval(id="playground-interval", interval=400, disabled=True),
            html.Div(
                className="grok-card",
                children=[
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "minmax(0, 1fr) minmax(0, 1fr)",
                            "gap": "16px",
                        },
                        children=[
                            html.Div(
                                children=[
                                    html.H4("Data", style={"margin": "0 0 10px 0"}),
                                    html.Div(
                                        className="control-row",
                                        children=[
                                            html.Div(
                                                className="control-group",
                                                children=[
                                                    html.Label(
                                                        "Dataset", className="widget-label"
                                                    ),
                                                    dcc.Dropdown(
                                                        id="playground-dataset",
                                                        options=dataset_options,
                                                        value="Moons",
                                                        clearable=False,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group",
                                                children=[
                                                    html.Label(
                                                        "Points", className="widget-label"
                                                    ),
                                                    dcc.Slider(
                                                        id="playground-points",
                                                        min=200,
                                                        max=2000,
                                                        step=50,
                                                        value=600,
                                                        marks=None,
                                                        tooltip={
                                                            "placement": "bottom",
                                                            "always_visible": True,
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group",
                                                children=[
                                                    html.Label(
                                                        "Noise", className="widget-label"
                                                    ),
                                                    dcc.Slider(
                                                        id="playground-noise",
                                                        min=0.0,
                                                        max=0.6,
                                                        step=0.02,
                                                        value=0.15,
                                                        marks=None,
                                                        tooltip={
                                                            "placement": "bottom",
                                                            "always_visible": True,
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group",
                                                children=[
                                                    html.Label(
                                                        "Val Split", className="widget-label"
                                                    ),
                                                    dcc.Slider(
                                                        id="playground-split",
                                                        min=0.05,
                                                        max=0.5,
                                                        step=0.05,
                                                        value=0.2,
                                                        marks=None,
                                                        tooltip={
                                                            "placement": "bottom",
                                                            "always_visible": True,
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group",
                                                children=[
                                                    html.Label(
                                                        "Seed", className="widget-label"
                                                    ),
                                                    dcc.Input(
                                                        id="playground-seed",
                                                        type="number",
                                                        value=7,
                                                        min=0,
                                                        step=1,
                                                        style={
                                                            "width": "100%",
                                                            "background": "rgba(0,0,0,0.4)",
                                                            "color": GROK_TEXT,
                                                            "border": "1px solid #283041",
                                                            "borderRadius": "8px",
                                                            "padding": "8px",
                                                        },
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="control-row",
                                        children=[
                                            html.Div(
                                                className="control-group",
                                                children=[
                                                    html.Label(
                                                        "Features", className="widget-label"
                                                    ),
                                                    dcc.Checklist(
                                                        id="playground-features",
                                                        options=feature_options,
                                                        value=["x1", "x2"],
                                                        labelStyle={
                                                            "display": "inline-block",
                                                            "marginRight": "10px",
                                                        },
                                                        style={
                                                            "maxHeight": "120px",
                                                            "overflowY": "auto",
                                                        },
                                                    ),
                                                ],
                                            )
                                        ],
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.H4("Model", style={"margin": "0 0 10px 0"}),
                                    html.Div(
                                        className="control-row",
                                        children=[
                                            html.Div(
                                                className="control-group",
                                                children=[
                                                    html.Label(
                                                        "Layers", className="widget-label"
                                                    ),
                                                    dcc.Slider(
                                                        id="playground-layer-count",
                                                        min=1,
                                                        max=4,
                                                        step=1,
                                                        value=2,
                                                        marks=None,
                                                        tooltip={
                                                            "placement": "bottom",
                                                            "always_visible": True,
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group",
                                                children=[
                                                    html.Label(
                                                        "Layer 1", className="widget-label"
                                                    ),
                                                    dcc.Slider(
                                                        id="playground-layer-1",
                                                        min=1,
                                                        max=PLAYGROUND_MAX_NODES,
                                                        step=1,
                                                        value=6,
                                                        marks=None,
                                                        tooltip={
                                                            "placement": "bottom",
                                                            "always_visible": True,
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group",
                                                children=[
                                                    html.Label(
                                                        "Layer 2", className="widget-label"
                                                    ),
                                                    dcc.Slider(
                                                        id="playground-layer-2",
                                                        min=1,
                                                        max=PLAYGROUND_MAX_NODES,
                                                        step=1,
                                                        value=4,
                                                        marks=None,
                                                        tooltip={
                                                            "placement": "bottom",
                                                            "always_visible": True,
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group",
                                                children=[
                                                    html.Label(
                                                        "Layer 3", className="widget-label"
                                                    ),
                                                    dcc.Slider(
                                                        id="playground-layer-3",
                                                        min=1,
                                                        max=PLAYGROUND_MAX_NODES,
                                                        step=1,
                                                        value=3,
                                                        marks=None,
                                                        tooltip={
                                                            "placement": "bottom",
                                                            "always_visible": True,
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group",
                                                children=[
                                                    html.Label(
                                                        "Layer 4", className="widget-label"
                                                    ),
                                                    dcc.Slider(
                                                        id="playground-layer-4",
                                                        min=1,
                                                        max=PLAYGROUND_MAX_NODES,
                                                        step=1,
                                                        value=3,
                                                        marks=None,
                                                        tooltip={
                                                            "placement": "bottom",
                                                            "always_visible": True,
                                                        },
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="control-row",
                                        children=[
                                            html.Div(
                                                className="control-group",
                                                children=[
                                                    html.Label(
                                                        "Activation", className="widget-label"
                                                    ),
                                                    dcc.RadioItems(
                                                        id="playground-activation",
                                                        options=[
                                                            {"label": " Tanh", "value": "tanh"},
                                                            {"label": " ReLU", "value": "relu"},
                                                            {
                                                                "label": " Sigmoid",
                                                                "value": "sigmoid",
                                                            },
                                                        ],
                                                        value="tanh",
                                                        className="grok-radio",
                                                        labelClassName="grok-radio-item",
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group",
                                                children=[
                                                    html.Label(
                                                        "Learning Rate",
                                                        className="widget-label",
                                                    ),
                                                    dcc.Dropdown(
                                                        id="playground-lr",
                                                        options=[
                                                            {
                                                                "label": f"{val:g}",
                                                                "value": val,
                                                            }
                                                            for val in lr_options
                                                        ],
                                                        value=0.03,
                                                        clearable=False,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group",
                                                children=[
                                                    html.Label(
                                                        "L2", className="widget-label"
                                                    ),
                                                    dcc.Slider(
                                                        id="playground-l2",
                                                        min=0.0,
                                                        max=0.2,
                                                        step=0.01,
                                                        value=0.0,
                                                        marks=None,
                                                        tooltip={
                                                            "placement": "bottom",
                                                            "always_visible": True,
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group",
                                                children=[
                                                    html.Label(
                                                        "Batch Size", className="widget-label"
                                                    ),
                                                    dcc.Dropdown(
                                                        id="playground-batch",
                                                        options=[
                                                            {
                                                                "label": str(val),
                                                                "value": val,
                                                            }
                                                            for val in batch_options
                                                        ],
                                                        value=64,
                                                        clearable=False,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="control-group",
                                                children=[
                                                    html.Label(
                                                        "Steps/Tick",
                                                        className="widget-label",
                                                    ),
                                                    dcc.Slider(
                                                        id="playground-steps",
                                                        min=1,
                                                        max=50,
                                                        step=1,
                                                        value=10,
                                                        marks=None,
                                                        tooltip={
                                                            "placement": "bottom",
                                                            "always_visible": True,
                                                        },
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="control-row",
                                        children=[
                                            html.Button(
                                                "Train Step",
                                                id="playground-train",
                                                className="grok-button",
                                                style={"flex": "1", "margin": "0"},
                                            ),
                                            html.Button(
                                                "? Play",
                                                id="playground-play-button",
                                                className="grok-button",
                                                style={
                                                    "flex": "1",
                                                    "margin": "0",
                                                    "background": "rgba(68, 170, 255, 0.2)",
                                                    "border-color": NEON_BLUE,
                                                },
                                            ),
                                            html.Button(
                                                "Reset",
                                                id="playground-reset",
                                                className="grok-button",
                                                style={"flex": "1", "margin": "0"},
                                            ),
                                        ],
                                    ),
                                ]
                            ),
                        ],
                    )
                ],
            ),
            html.Div(
                className="grok-card",
                children=[
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "minmax(0, 1.2fr) minmax(0, 1fr)",
                            "gap": "16px",
                            "alignItems": "stretch",
                        },
                        children=[
                            dcc.Graph(
                                id="playground-boundary",
                                config={"displayModeBar": False},
                            ),
                            dcc.Graph(
                                id="playground-loss",
                                config={"displayModeBar": False},
                            ),
                        ],
                    ),
                    html.Div(id="playground-status", className="grok-status"),
                ],
            ),
        ],
        className="tab",
        selected_className="tab--selected",
    )


def build_tabs():
    tabs = [
        dcc.Tab(
            label="Loss",
            value="loss",
            children=[
                html.Div(
                    className="grok-card",
                    children=[
                        dcc.Graph(
                            id="loss-graph",
                            config={"displayModeBar": False},
                        ),
                        html.Div(id="loss-status", className="grok-status"),
                    ],
                )
            ],
            className="tab",
            selected_className="tab--selected",
        ),
        dcc.Tab(
            label="Metrics",
            value="metrics",
            children=[
                html.Div(
                    className="grok-card",
                    children=[
                        dcc.Graph(
                            id="metrics-graph",
                            config={"displayModeBar": False},
                        ),
                        html.Div(id="metrics-status", className="grok-status"),
                    ],
                )
            ],
            className="tab",
            selected_className="tab--selected",
        ),
        dcc.Tab(
            label="Representation Grid",
            value="rep-grid",
            children=[
                html.Div(
                    className="grok-card",
                    children=[
                        html.Div(
                            className="control-row",
                            children=[
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Epoch", className="widget-label"),
                                        dcc.Dropdown(id="rep-epoch", clearable=False),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Projection", className="widget-label"),
                                        dcc.RadioItems(
                                            id="rep-method",
                                            options=[
                                                {"label": " PCA", "value": "pca"},
                                                {"label": " UMAP", "value": "umap"},
                                            ],
                                            value="pca",
                                            className="grok-radio",
                                            labelClassName="grok-radio-item",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("View", className="widget-label"),
                                        dcc.RadioItems(
                                            id="rep-grid-view",
                                            options=[
                                                {"label": " Grid", "value": "grid"},
                                                {"label": " Matrix", "value": "matrix"},
                                            ],
                                            value="grid",
                                            className="grok-radio",
                                            labelClassName="grok-radio-item",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Layers", className="widget-label"),
                                        dcc.Dropdown(id="rep-layers", multi=True),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("PCs (Matrix)", className="widget-label"),
                                        dcc.Slider(
                                            id="rep-pcs",
                                            min=2,
                                            max=6,
                                            step=1,
                                            value=3,
                                            marks=None,
                                            tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Grid Cols", className="widget-label"),
                                        dcc.Slider(
                                            id="rep-grid-cols",
                                            min=1,
                                            max=6,
                                            step=1,
                                            value=2,
                                            marks=None,
                                            tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Options", className="widget-label"),
                                        dcc.Checklist(
                                            id="rep-show-train",
                                            options=[{"label": " Train", "value": "train"}],
                                            value=["train"],
                                        ),
                                        dcc.Checklist(
                                            id="rep-show-val",
                                            options=[{"label": " Val", "value": "val"}],
                                            value=["val"],
                                        ),
                                        dcc.Checklist(
                                            id="rep-disable-sampling",
                                            options=[{"label": " Disable Sampling", "value": "disable"}],
                                            value=[],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Auto-Zoom", className="widget-label"),
                                        dcc.Checklist(
                                            id="rep-auto-zoom",
                                            options=[{"label": " Outliers", "value": "zoom"}],
                                            value=["zoom"],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label(id="rep-zoom-label", children="Pct: 99", className="widget-label"),
                                        dcc.Slider(
                                            id="rep-zoom-pct",
                                            min=90, max=100, step=0.5, value=99,
                                            marks=None,
                                            tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Graph(
                            id="rep-grid-graph",
                            config={"displayModeBar": False},
                            style={"height": "800px"},
                            figure={"data": [], "layout": {"height": 800}},
                        ),
                        html.Div(id="rep-grid-status", className="grok-status"),
                    ],
                )
            ],
            className="tab",
            selected_className="tab--selected",
        ),
        dcc.Tab(
            label="Single Projection",
            value="rep",
            children=[
                html.Div(
                    className="grok-card",
                    children=[
                        html.Div(
                            className="control-row",
                            children=[
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Epoch", className="widget-label"),
                                        dcc.Dropdown(id="proj-epoch", clearable=False),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Layer", className="widget-label"),
                                        dcc.Dropdown(id="proj-layer", clearable=False),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Method", className="widget-label"),
                                        dcc.RadioItems(
                                            id="proj-method",
                                            options=[
                                                {"label": " PCA", "value": "pca"},
                                                {"label": " UMAP", "value": "umap"},
                                            ],
                                            value="umap",
                                            className="grok-radio",
                                            labelClassName="grok-radio-item",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Auto-Zoom", className="widget-label"),
                                        dcc.Checklist(
                                            id="proj-auto-zoom",
                                            options=[{"label": " Outliers", "value": "zoom"}],
                                            value=["zoom"],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label(id="proj-zoom-label", children="Pct: 99", className="widget-label"),
                                        dcc.Slider(
                                            id="proj-zoom-pct",
                                            min=90, max=100, step=0.5, value=99,
                                            marks=None,
                                            tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Graph(id="proj-graph", config={"displayModeBar": False}),
                        html.Div(id="proj-status", className="grok-status"),
                    ],
                )
            ],
            className="tab",
            selected_className="tab--selected",
        ),
        dcc.Tab(
            label="Decision Surface",
            value="surface",
            children=[
                html.Div(
                    className="grok-card",
                    children=[
                        html.Div(
                            className="control-row",
                            children=[
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Epoch", className="widget-label"),
                                        dcc.Dropdown(id="surface-epoch", clearable=False),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Metric", className="widget-label"),
                                        dcc.Dropdown(
                                            id="surface-metric",
                                            options=[
                                                {
                                                    "label": "Confidence (max prob)",
                                                    "value": "Confidence (max prob)",
                                                },
                                                {"label": "Entropy", "value": "Entropy"},
                                                {
                                                    "label": "Class probability",
                                                    "value": "Class probability",
                                                },
                                            ],
                                            value="Confidence (max prob)",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    id="surface-class-container",
                                    className="control-group",
                                    children=[
                                        html.Label("Class", className="widget-label"),
                                        dcc.Dropdown(id="surface-class", clearable=False),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Overlay", className="widget-label"),
                                        dcc.Checklist(
                                            id="surface-overlay",
                                            options=[{"label": " Points", "value": "on"}],
                                            value=["on"],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Graph(id="surface-graph", config={"displayModeBar": False}),
                        html.Div(id="surface-status", className="grok-status"),
                    ],
                )
            ],
            className="tab",
            selected_className="tab--selected",
        ),
        dcc.Tab(
            label="Linear Probes",
            value="probes",
            children=[
                html.Div(
                    className="grok-card",
                    children=[
                        html.Div(
                            className="control-row",
                            children=[
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Metric", className="widget-label"),
                                        dcc.Dropdown(
                                            id="probe-metric",
                                            options=[
                                                {"label": "probe_acc", "value": "probe_acc"},
                                                {"label": "probe_f1", "value": "probe_f1"},
                                            ],
                                            value="probe_acc",
                                            clearable=False,
                                        ),
                                    ],
                                )
                            ],
                        ),
                        dcc.Graph(id="probe-graph", config={"displayModeBar": False}),
                        html.Div(id="probe-status", className="grok-status"),
                    ],
                )
            ],
            className="tab",
            selected_className="tab--selected",
        ),
        dcc.Tab(
            label="Interpolation",
            value="interp",
            children=[
                html.Div(
                    className="grok-card",
                    children=[
                        html.Div(
                            className="control-row",
                            children=[
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Epoch", className="widget-label"),
                                        dcc.Dropdown(id="interp-epoch", clearable=False),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Pair Index", className="widget-label"),
                                        dcc.Slider(
                                            id="interp-pair",
                                            min=0,
                                            max=0,
                                            step=1,
                                            value=0,
                                            marks=None,
                                            tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Graph(id="interp-graph", config={"displayModeBar": False}),
                        html.Div(id="interp-info", className="grok-status"),
                    ],
                )
            ],
            className="tab",
            selected_className="tab--selected",
        ),
        dcc.Tab(
            label="Neuron Responses",
            value="responses",
            children=[
                html.Div(
                    className="grok-card",
                    children=[
                        html.Div(
                            className="control-row",
                            children=[
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Epoch", className="widget-label"),
                                        dcc.Dropdown(id="resp-epoch", clearable=False),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("View Mode", className="widget-label"),
                                        dcc.RadioItems(
                                            id="resp-view",
                                            options=[
                                                {"label": " Single unit", "value": "single"},
                                                {"label": " Small multiples", "value": "grid"},
                                            ],
                                            value="single",
                                            className="grok-radio",
                                            labelClassName="grok-radio-item",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Unit", className="widget-label"),
                                        dcc.Dropdown(id="resp-unit", clearable=False),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Units", className="widget-label"),
                                        dcc.Dropdown(
                                            id="resp-units",
                                            multi=True,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label(
                                            "Grid columns", className="widget-label"
                                        ),
                                        dcc.Dropdown(
                                            id="resp-cols",
                                            options=[
                                                {"label": "2", "value": 2},
                                                {"label": "3", "value": 3},
                                            ],
                                            value=3,
                                            clearable=False,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Graph(id="resp-graph", config={"displayModeBar": False}),
                        html.Div(id="resp-status", className="grok-status"),
                    ],
                )
            ],
            className="tab",
            selected_className="tab--selected",
        ),
        dcc.Tab(
            label="Loss Landscape",
            value="land",
            children=[
                html.Div(
                    className="grok-card",
                    children=[
                        html.Div(
                            className="control-row",
                            children=[
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Epoch", className="widget-label"),
                                        dcc.Dropdown(id="land-epoch", clearable=False),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("View", className="widget-label"),
                                        dcc.RadioItems(
                                            id="land-view",
                                            options=[
                                                {"label": " 3D Surface", "value": "3d"},
                                                {"label": " Heatmap", "value": "heat"},
                                            ],
                                            value="heat",
                                            className="grok-radio",
                                            labelClassName="grok-radio-item",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Graph(id="land-graph", config={"displayModeBar": False}),
                        html.Div(id="land-status", className="grok-status"),
                    ],
                )
            ],
            className="tab",
            selected_className="tab--selected",
        ),
        dcc.Tab(
            label="Training Filmstrip",
            value="film",
            children=[
                html.Div(
                    className="grok-card",
                    children=[
                        html.Div(
                            className="control-row",
                            children=[
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Artifact", className="widget-label"),
                                        dcc.Dropdown(
                                            id="film-artifact",
                                            options=[
                                                {"label": "Decision surface", "value": "surface"},
                                                {"label": "Representation scatter", "value": "proj"},
                                                {"label": "Probe snapshot", "value": "probe"},
                                            ],
                                            value="surface",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Cols", className="widget-label"),
                                        dcc.Slider(
                                            id="film-cols",
                                            min=2, max=4, step=1, value=3,
                                            marks=None,
                                            tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Panels", className="widget-label"),
                                        dcc.Slider(
                                            id="film-panels",
                                            min=4, max=12, step=1, value=8,
                                            marks=None,
                                            tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Stride", className="widget-label"),
                                        dcc.Slider(
                                            id="film-stride",
                                            min=1, max=20, step=1, value=5,
                                            marks=None,
                                            tooltip={"placement": "bottom", "always_visible": True},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Anchor", className="widget-label"),
                                        dcc.RadioItems(
                                            id="film-anchor",
                                            options=[
                                                {"label": " Latest", "value": "Latest"},
                                                {"label": " Start", "value": "Start"},
                                            ],
                                            value="Latest",
                                            className="grok-radio",
                                            labelClassName="grok-radio-item",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Layer", className="widget-label"),
                                        dcc.Dropdown(id="film-layer", clearable=False),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Projection", className="widget-label"),
                                        dcc.RadioItems(
                                            id="film-method",
                                            options=[
                                                {"label": " PCA", "value": "pca"},
                                                {"label": " UMAP", "value": "umap"},
                                            ],
                                            value="pca",
                                            className="grok-radio",
                                            labelClassName="grok-radio-item",
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Surface metric", className="widget-label"),
                                        dcc.Dropdown(
                                            id="film-surface-metric",
                                            options=[
                                                {
                                                    "label": "Confidence (max prob)",
                                                    "value": "Confidence (max prob)",
                                                },
                                                {"label": "Entropy", "value": "Entropy"},
                                                {"label": "Class probability", "value": "Class probability"},
                                            ],
                                            value="Confidence (max prob)",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Surface class", className="widget-label"),
                                        dcc.Dropdown(id="film-surface-class", clearable=False),
                                    ],
                                ),
                                html.Div(
                                    className="control-group",
                                    children=[
                                        html.Label("Probe metric", className="widget-label"),
                                        dcc.Dropdown(
                                            id="film-probe-metric",
                                            options=[
                                                {"label": "probe_acc", "value": "probe_acc"},
                                                {"label": "probe_f1", "value": "probe_f1"},
                                            ],
                                            value="probe_acc",
                                            clearable=False,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Graph(id="film-graph", config={"displayModeBar": False}),
                        html.Div(id="film-status", className="grok-status"),
                    ],
                )
            ],
            className="tab",
            selected_className="tab--selected",
        ),
        dcc.Tab(
            label="Training Snapshots",
            value="snapshots",
            children=[
                html.Div(
                    className="grok-card",
                    children=[
                        html.Img(
                            id="snapshot-img",
                            style={
                                "maxWidth": "100%",
                                "borderRadius": "12px",
                                "border": "1px solid #1d2430",
                            },
                        ),
                        html.Div(id="snapshot-status", className="grok-status"),
                    ],
                )
            ],
            className="tab",
            selected_className="tab--selected",
        ),
        build_playground_tab(),
    ]
    return dcc.Tabs(id="main-tabs", value="loss", className="grok-tabs", children=tabs)


def build_layout(app, runs_dir):
    return html.Div(
        className="grok-app",
        style={"background": GROK_BG, "minHeight": "100vh", "color": GROK_TEXT},
        children=[
            dcc.Location(id="url", refresh=False),
            # CSS is injected via app.index_string
            
            # Store for sidebar state
            dcc.Store(id="sidebar-store", data={"collapsed": False}),
            
            # Sidebar Toggle Button
            html.Button(
                "☰",
                id="sidebar-toggle",
                className="sidebar-toggle",
            ),
            
            html.Div(
                id="grok-sidebar",
                className="grok-sidebar",
                children=[
                    dcc.Store(id="global-epoch", data=None),
                    dcc.Store(id="is-playing", data=False),
                    dcc.Interval(id="animate-interval", interval=500, disabled=True),
                    
                    html.Div(
                        className="widget-group",
                        style={"border-bottom": "1px solid rgba(255,255,255,0.1)", "padding-bottom": "20px"},
                        children=[
                            html.Label("Global Control", className="widget-label", style={"color": NEON_BLUE}),
                            html.Div(
                                className="control-row",
                                style={"background": "transparent", "padding": "0", "gap": "8px", "margin-bottom": "8px"},
                                children=[
                                    html.Button(
                                        "▶ Play", 
                                        id="play-button", 
                                        className="grok-button", 
                                        style={"flex": "1", "margin": "0", "background": "rgba(68, 170, 255, 0.2)", "border-color": NEON_BLUE}
                                    ),
                                ]
                            ),
                            html.Label("Speed (sec/step)", className="widget-label", style={"font-size": "10px"}),
                            dcc.Slider(
                                id="play-speed",
                                min=0.1, max=2.0, step=0.1, value=0.5,
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            html.Label("Global Epoch", className="widget-label", style={"font-size": "10px", "margin-top": "8px"}),
                            dcc.Slider(
                                id="global-epoch-slider",
                                min=0, max=100, step=1, value=0,
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            dcc.Checklist(
                                id="link-global",
                                options=[{"label": " Link Tabs to Global", "value": "linked"}],
                                value=["linked"],
                                labelStyle={"color": GROK_MUTED, "cursor": "pointer", "font-size": "12px", "margin-top": "8px"},
                            ),
                        ]
                    ),

                    html.Div(
                        className="widget-group",
                        children=[
                            html.Label("Run", className="widget-label"),
                            dcc.Dropdown(
                                id="run-select",
                                options=[
                                    {"label": run, "value": run}
                                    for run in list_runs(runs_dir)
                                ],
                                value=None,
                                placeholder="Select a run...",
                                clearable=False,
                            ),
                        ]
                    ),
                    html.Div(
                        className="widget-group",
                        children=[
                            html.Button("Refresh Runs", id="refresh-runs", className="grok-button"),
                            html.Button("Refresh Data", id="refresh-metrics", className="grok-button"),
                        ]
                    ),
                    html.Div(
                        className="widget-group",
                        children=[
                            html.Label("Auto-refresh", className="widget-label"),
                            dcc.Checklist(
                                id="auto-refresh",
                                options=[{"label": " Enabled", "value": "on"}],
                                value=[],
                                labelStyle={"color": GROK_MUTED, "cursor": "pointer"},
                            ),
                            html.Div(style={"height": "12px"}),
                            dcc.Slider(
                                id="refresh-sec",
                                min=1, max=30, step=1, value=5,
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ]
                    ),
                    dcc.Interval(id="refresh-interval", interval=5000, disabled=True),
                ]
            ),
            
            html.Div(
                id="grok-main",
                className="grok-main",
                children=[
                    html.H1("Grokking Training Dashboard", className="grok-title"),
                    build_tabs(),
                ]
            )
        ],
    )


def create_app(runs_dir):
    app = Dash(__name__)
    app.title = "Grokking Dashboard"
    app.index_string = INDEX_STRING.replace("__CUSTOM_CSS__", CUSTOM_CSS)
    app.layout = build_layout(app, runs_dir)

    @app.callback(
        Output("refresh-interval", "interval"),
        Output("refresh-interval", "disabled"),
        Input("auto-refresh", "value"),
        Input("refresh-sec", "value"),
    )
    def update_interval(auto_values, refresh_sec):
        enabled = "on" in (auto_values or [])
        interval_ms = int((refresh_sec or 5) * 1000)
        return interval_ms, not enabled

    @app.callback(
        Output("run-select", "options"),
        Output("run-select", "value"),
        Input("refresh-runs", "n_clicks"),
        Input("refresh-interval", "n_intervals"),
        State("run-select", "value"),
    )
    def refresh_runs_list(_n_clicks, _n_intervals, current_value):
        runs = list_runs(runs_dir)
        options = [{"label": run, "value": run} for run in runs]
        # Preserve selection if possible, else select latest
        if current_value in runs:
            value = current_value
        else:
            value = runs[-1] if runs else None
        return options, value

    @app.callback(
        Output("rep-epoch", "options"),
        Output("proj-epoch", "options"),
        Output("surface-epoch", "options"),
        Output("interp-epoch", "options"),
        Output("resp-epoch", "options"),
        Output("land-epoch", "options"),
        Output("global-epoch-slider", "min"),
        Output("global-epoch-slider", "max"),
        Output("global-epoch-slider", "marks"),
        Input("run-select", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
    )
    def update_epoch_options(
        run_name,
        _n_intervals,
        _n_clicks,
    ):
        def build_options(epochs):
            return [{"label": str(e), "value": e} for e in epochs]

        if not run_name:
            return [], [], [], [], [], [], 0, 100, None

        run_dir = runs_dir / run_name
        rep_epochs = list_epochs(run_dir / "diagnostics" / "rep")
        proj_epochs = list_epochs(run_dir / "projections")
        surface_epochs = list_epochs(run_dir / "surfaces")
        interp_epochs = list_epochs(run_dir / "interpolations")
        resp_epochs = list_epochs(run_dir / "responses")
        land_epochs = list_epochs(run_dir / "landscapes")
        
        all_epochs = sorted(list(set(rep_epochs + proj_epochs + surface_epochs + interp_epochs + resp_epochs + land_epochs)))
        global_min = min(all_epochs) if all_epochs else 0
        global_max = max(all_epochs) if all_epochs else 100
        
        return (
            build_options(rep_epochs),
            build_options(proj_epochs),
            build_options(surface_epochs),
            build_options(interp_epochs),
            build_options(resp_epochs),
            build_options(land_epochs),
            global_min,
            global_max,
            None,
        )

    @app.callback(
        Output("is-playing", "data"),
        Output("play-button", "children"),
        Output("play-button", "style"),
        Output("animate-interval", "disabled"),
        Input("play-button", "n_clicks"),
        State("is-playing", "data"),
    )
    def toggle_play(n_clicks, is_playing):
        if not n_clicks:
            return False, "▶ Play", {"flex": "1", "margin": "0", "background": "rgba(68, 170, 255, 0.2)", "border-color": NEON_BLUE}, True
        new_state = not is_playing
        label = "⏸ Pause" if new_state else "▶ Play"
        style = {
            "flex": "1", 
            "margin": "0", 
            "background": "rgba(68, 170, 255, 0.5)" if new_state else "rgba(68, 170, 255, 0.2)", 
            "border-color": NEON_BLUE
        }
        return new_state, label, style, not new_state

    @app.callback(
        Output("animate-interval", "interval"),
        Input("play-speed", "value"),
    )
    def update_speed(sec):
        return int((sec or 0.5) * 1000)

    @app.callback(
        Output("global-epoch-slider", "value"),
        Input("animate-interval", "n_intervals"),
        State("global-epoch-slider", "value"),
        State("global-epoch-slider", "max"),
        State("global-epoch-slider", "min"),
    )
    def animate_epoch(_n, current, max_val, min_val):
        if current is None:
            return min_val or 0
        if max_val is None:
            return current
        nxt = current + 1
        if nxt > max_val:
            nxt = min_val or 0
        return nxt

    @app.callback(
        Output("global-epoch", "data"),
        Input("global-epoch-slider", "value"),
    )
    def sync_global_store(value):
        return value

    @app.callback(
        Output("rep-epoch", "value"),
        Output("proj-epoch", "value"),
        Output("surface-epoch", "value"),
        Output("interp-epoch", "value"),
        Output("resp-epoch", "value"),
        Output("land-epoch", "value"),
        Input("global-epoch", "data"),
        Input("link-global", "value"),
        Input("is-playing", "data"),
        Input("rep-epoch", "options"),
        Input("proj-epoch", "options"),
        Input("surface-epoch", "options"),
        Input("interp-epoch", "options"),
        Input("resp-epoch", "options"),
        Input("land-epoch", "options"),
        State("rep-epoch", "value"),
        State("proj-epoch", "value"),
        State("surface-epoch", "value"),
        State("interp-epoch", "value"),
        State("resp-epoch", "value"),
        State("land-epoch", "value"),
    )
    def update_tab_values(
        global_epoch, link_values, is_playing,
        rep_opt, proj_opt, surface_opt, interp_opt, resp_opt, land_opt,
        rep_val, proj_val, surface_val, interp_val, resp_val, land_val
    ):
        linked = "linked" in (link_values or [])
        
        # If playing, skip updating dropdowns to prevent callback overload/double-fire
        if is_playing and linked:
             return [dash.no_update] * 6

        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        def get_nearest(target, options):
            if not options or target is None:
                return None
            vals = [o["value"] for o in options]
            if target in vals:
                return target
            return min(vals, key=lambda x: abs(x - target))
        
        def validate(current, options):
            if not options: return None
            vals = [o["value"] for o in options]
            if current in vals: return current
            return vals[-1] # Default to latest

        # If triggered by Global Epoch and Linked
        if linked and global_epoch is not None:
            return (
                get_nearest(global_epoch, rep_opt),
                get_nearest(global_epoch, proj_opt),
                get_nearest(global_epoch, surface_opt),
                get_nearest(global_epoch, interp_opt),
                get_nearest(global_epoch, resp_opt),
                get_nearest(global_epoch, land_opt),
            )
        
        # If triggered by Options update (new run loaded), ensuring default values
        # Or if not linked, just validate current
        return (
            validate(rep_val, rep_opt),
            validate(proj_val, proj_opt),
            validate(surface_val, surface_opt),
            validate(interp_val, interp_opt),
            validate(resp_val, resp_opt),
            validate(land_val, land_opt),
        )

    @app.callback(
        Output("rep-layers", "options"),
        Output("rep-layers", "value"),
        Input("run-select", "value"),
        Input("rep-epoch", "value"),
        Input("rep-method", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
        State("rep-layers", "value"),
    )
    def update_rep_layers(run_name, rep_epoch, method, _n_intervals, _n_clicks, current_value):
        if not run_name or rep_epoch is None:
            return [], []
        run_dir = runs_dir / run_name
        rep_path = run_dir / "diagnostics" / "rep" / f"epoch_{rep_epoch:03d}.npz"
        data, _ = safe_read_npz(rep_path)
        if data is None:
            return [], []
        if method == "pca":
            suffixes = ("_pca", "_pc")
        else:
            suffixes = (f"_{method}",)
        layers = sorted(
            {
                key.rsplit("_", 1)[0]
                for key in data
                if any(key.endswith(sfx) for sfx in suffixes)
            }
        )
        options = [{"label": layer, "value": layer} for layer in layers]
        if current_value:
            if isinstance(current_value, str):
                current_value = [current_value]
            kept = [val for val in current_value if val in layers]
            if kept:
                return options, kept
        defaults = [
            name
            for name in ("input", "block1", "penultimate", "logits")
            if name in layers
        ]
        if not defaults:
            defaults = layers[: min(4, len(layers))]
        return options, defaults

    @app.callback(
        Output("rep-pcs", "max"),
        Output("rep-pcs", "value"),
        Input("run-select", "value"),
        Input("rep-epoch", "value"),
        Input("rep-method", "value"),
        Input("rep-layers", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
        State("rep-pcs", "value"),
    )
    def update_rep_pcs_bounds(
        run_name,
        rep_epoch,
        method,
        layers,
        _n_intervals,
        _n_clicks,
        current_value,
    ):
        """Dynamically update the PCs slider max based on available PC data."""
        min_val = 2
        default_max = min_val
        max_components = None

        if run_name and rep_epoch is not None:
            run_dir = runs_dir / run_name
            rep_path = run_dir / "diagnostics" / "rep" / f"epoch_{rep_epoch:03d}.npz"
            data, _ = safe_read_npz(rep_path)
        else:
            data = None

        if data:
            if method == "umap":
                umap_k = data.get("umap_k")
                if umap_k is not None:
                    try:
                        max_components = int(np.array(umap_k).item())
                    except Exception:
                        max_components = None
                if max_components is None:
                    layer_list = layers if layers else data.get("layers")
                    if layer_list is not None and not isinstance(layer_list, list):
                        layer_list = [str(x) for x in np.array(layer_list).tolist()]
                    for layer in layer_list or []:
                        arr = data.get(f"{layer}_umap")
                        if arr is not None and getattr(arr, "ndim", 0) == 2:
                            max_components = arr.shape[1]
                            break
                if max_components is None:
                    for key, arr in data.items():
                        if key.endswith("_umap") and getattr(arr, "ndim", 0) == 2:
                            max_components = arr.shape[1]
                            break
            else:
                pc_k = data.get("pc_k")
                if pc_k is not None:
                    try:
                        max_components = int(np.array(pc_k).item())
                    except Exception:
                        max_components = None
                if max_components is None:
                    layer_list = layers if layers else data.get("layers")
                    if layer_list is not None and not isinstance(layer_list, list):
                        layer_list = [str(x) for x in np.array(layer_list).tolist()]
                    for layer in layer_list or []:
                        arr = data.get(f"{layer}_pc")
                        if arr is not None and getattr(arr, "ndim", 0) == 2:
                            max_components = arr.shape[1]
                            break
                if max_components is None:
                    for key, arr in data.items():
                        if key.endswith("_pca") and getattr(arr, "ndim", 0) == 2:
                            max_components = arr.shape[1]
                            break
                if max_components is None:
                    for key, arr in data.items():
                        if key.endswith("_pc") and getattr(arr, "ndim", 0) == 2:
                            max_components = arr.shape[1]
                            break

        max_val = max(min_val, int(max_components or default_max))
        value = int(current_value or 3)
        value = max(min_val, min(max_val, value))
        return max_val, value

    @app.callback(
        Output("proj-layer", "options"),
        Output("proj-layer", "value"),
        Input("run-select", "value"),
        Input("proj-epoch", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
    )
    def update_proj_layers(run_name, proj_epoch, _n_intervals, _n_clicks):
        if not run_name or proj_epoch is None:
            return [], None
        run_dir = runs_dir / run_name
        proj_path = run_dir / "projections" / f"epoch_{proj_epoch:03d}.npz"
        data, _ = safe_read_npz(proj_path)
        if data is None:
            return [], None
        layers = sorted(
            {key.rsplit("_", 1)[0] for key in data if key.endswith(("_pca", "_umap"))}
        )
        options = [{"label": layer, "value": layer} for layer in layers]
        value = layers[-1] if layers else None
        return options, value

    @app.callback(
        Output("surface-class", "options"),
        Output("surface-class", "value"),
        Output("surface-class", "disabled"),
        Input("run-select", "value"),
        Input("surface-epoch", "value"),
        Input("surface-metric", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
    )
    def update_surface_class(run_name, surface_epoch, metric, _n_intervals, _n_clicks):
        if metric != "Class probability":
            return [], None, True
        if not run_name or surface_epoch is None:
            return [], None, True
        run_dir = runs_dir / run_name
        surface_path = run_dir / "surfaces" / f"epoch_{surface_epoch:03d}.npz"
        data, _ = safe_read_npz(surface_path)
        if data is None:
            return [], None, True
        classes = list(range(data["proba"].shape[2]))
        class_names = load_config(run_dir).get("classes")
        if class_names:
            options = [{"label": name, "value": idx} for idx, name in enumerate(class_names)]
        else:
            options = [{"label": str(idx), "value": idx} for idx in classes]
        return options, options[0]["value"] if options else None, False

    @app.callback(
        Output("film-layer", "options"),
        Output("film-layer", "value"),
        Input("run-select", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
    )
    def update_film_layers(run_name, _n_intervals, _n_clicks):
        if not run_name:
            return [], None
        run_dir = runs_dir / run_name
        proj_epochs = list_epochs(run_dir / "projections")
        if not proj_epochs:
            return [], None
        latest = proj_epochs[-1]
        proj_path = run_dir / "projections" / f"epoch_{latest:03d}.npz"
        data, _ = safe_read_npz(proj_path)
        if data is None:
            return [], None
        layers = sorted(
            {key.rsplit("_", 1)[0] for key in data if key.endswith(("_pca", "_umap"))}
        )
        options = [{"label": layer, "value": layer} for layer in layers]
        value = layers[-1] if layers else None
        return options, value

    @app.callback(
        Output("film-surface-class", "options"),
        Output("film-surface-class", "value"),
        Output("film-surface-class", "disabled"),
        Input("run-select", "value"),
        Input("film-surface-metric", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
    )
    def update_film_surface_class(run_name, metric, _n_intervals, _n_clicks):
        if metric != "Class probability":
            return [], None, True
        if not run_name:
            return [], None, True
        run_dir = runs_dir / run_name
        surface_epochs = list_epochs(run_dir / "surfaces")
        if not surface_epochs:
            return [], None, True
        latest = surface_epochs[-1]
        surface_path = run_dir / "surfaces" / f"epoch_{latest:03d}.npz"
        data, _ = safe_read_npz(surface_path)
        if data is None:
            return [], None, True
        classes = list(range(data["proba"].shape[2]))
        class_names = load_config(run_dir).get("classes")
        if class_names:
            options = [{"label": name, "value": idx} for idx, name in enumerate(class_names)]
        else:
            options = [{"label": str(idx), "value": idx} for idx in classes]
        return options, options[0]["value"] if options else None, False

    @app.callback(
        Output("resp-unit", "options"),
        Output("resp-unit", "value"),
        Output("resp-units", "options"),
        Output("resp-units", "value"),
        Input("run-select", "value"),
        Input("resp-epoch", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
    )
    def update_response_units(run_name, resp_epoch, _n_intervals, _n_clicks):
        if not run_name or resp_epoch is None:
            return [], None, [], []
        run_dir = runs_dir / run_name
        resp_path = run_dir / "responses" / f"epoch_{resp_epoch:03d}.npz"
        data, _ = safe_read_npz(resp_path)
        if data is None:
            return [], None, [], []
        unit_indices = data["unit_indices"].astype(int).tolist()
        options = [{"label": str(u), "value": int(u)} for u in unit_indices]
        default_units = unit_indices[: min(6, len(unit_indices))]
        default_unit = default_units[0] if default_units else None
        return options, default_unit, options, default_units

    @app.callback(
        Output("interp-pair", "max"),
        Output("interp-pair", "value"),
        Input("run-select", "value"),
        Input("interp-epoch", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
    )
    def update_interp_slider(run_name, interp_epoch, _n_intervals, _n_clicks):
        if not run_name or interp_epoch is None:
            return 0, 0
        run_dir = runs_dir / run_name
        interp_path = run_dir / "interpolations" / f"epoch_{interp_epoch:03d}.npz"
        data, _ = safe_read_npz(interp_path)
        if data is None:
            return 0, 0
        n_pairs = int(data["proba_a"].shape[0])
        max_idx = max(0, n_pairs - 1)
        return max_idx, 0

    @app.callback(
        Output("loss-graph", "figure"),
        Output("loss-status", "children"),
        Input("run-select", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
    )
    def update_loss_curves(run_name, _n_intervals, _n_clicks):
        if not run_name:
            return empty_figure("Select a run to view metrics."), ""
        metrics_df, mtime, message = read_metrics(runs_dir, run_name)
        fig = build_loss_figure(metrics_df)
        if message:
            status = f"Status: {message}"
        elif mtime:
            t_str = time.strftime("%H:%M:%S", time.localtime(mtime))
            status = f"Last updated: {t_str}"
        else:
            status = ""
        return fig, status

    @app.callback(
        Output("metrics-graph", "figure"),
        Output("metrics-status", "children"),
        Input("run-select", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
    )
    def update_metrics_tab(run_name, _n_intervals, _n_clicks):
        if not run_name:
            return empty_figure("Select a run to view metrics."), ""
        metrics_df, mtime, message = read_metrics(runs_dir, run_name)
        fig = build_metrics_figure(metrics_df)
        if message:
            status = f"Status: {message}"
        elif mtime:
            t_str = time.strftime("%H:%M:%S", time.localtime(mtime))
            status = f"Last updated: {t_str}"
        else:
            status = ""
        return fig, status

    @app.callback(
        Output("surface-class-container", "style"),
        Input("surface-metric", "value"),
    )
    def toggle_surface_class_visibility(metric):
        if metric == "Class probability":
             return {} # Use default CSS
        return {"display": "none"}

    @app.callback(
        Output("rep-zoom-label", "children"),
        Input("rep-zoom-pct", "value"),
    )
    def update_rep_zoom_label(val):
        return f"Pct: {val}"

    @app.callback(
        Output("rep-grid-graph", "figure"),
        Output("rep-grid-status", "children"),
        Input("run-select", "value"),
        Input("rep-epoch", "value"),
        Input("rep-method", "value"),
        Input("rep-layers", "value"),
        Input("rep-show-train", "value"),
        Input("rep-show-val", "value"),
        Input("rep-disable-sampling", "value"),
        Input("rep-pcs", "value"),
        Input("rep-grid-cols", "value"),
        Input("rep-grid-view", "value"),
        Input("rep-auto-zoom", "value"),
        Input("rep-zoom-pct", "value"),
        Input("global-epoch", "data"),
        Input("link-global", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
        State("rep-epoch", "options"),
    )
    def update_rep_grid(
        run_name,
        rep_epoch_val,
        method,
        layers,
        split_train_opts,
        split_val_opts,
        sampling_opts,
        pcs_value,
        grid_cols,
        view_mode,
        zoom_opts,
        zoom_pct,
        global_epoch,
        link_global,
        _n_intervals,
        _n_clicks,
        rep_options,
    ):
        split_filter = []
        if split_train_opts and "train" in split_train_opts:
            split_filter.append("train")
        if split_val_opts and "val" in split_val_opts:
            split_filter.append("val")
        
        disable_sampling = "disable" in (sampling_opts or [])
        """Render representation Grid or Matrix view."""
        # Determine which epoch to use (sync with global if linked)
        rep_epoch = rep_epoch_val
        if "linked" in (link_global or []) and global_epoch is not None and rep_options:
            vals = [o["value"] for o in rep_options]
            if global_epoch in vals:
                rep_epoch = global_epoch
            else:
                rep_epoch = min(vals, key=lambda x: abs(x - global_epoch))

        # Early exits for missing data
        if not run_name or rep_epoch is None:
            return empty_figure("No representation data available.", height=500), ""
        
        if not layers:
            return empty_figure("Select at least one layer.", height=500), ""
        
        if isinstance(layers, str):
            layers = [layers]

        # Load data
        run_dir = runs_dir / run_name
        rep_path = run_dir / "diagnostics" / "rep" / f"epoch_{rep_epoch:03d}.npz"
        data, message = safe_read_npz(rep_path)
        if data is None:
            return empty_figure(message or "Representation file not ready.", height=500), ""
        
        labels_raw = data.get("labels")
        split_raw = data.get("split")
        if labels_raw is None or split_raw is None:
            return empty_figure("Missing labels/split metadata.", height=500), ""

        view_mode = (view_mode or "grid").lower()

        class_names = load_config(run_dir).get("classes")
        labels_named = map_labels(labels_raw, class_names)
        split_names = np.where(split_raw == 0, "train", "val")
        label_names = class_names if class_names else sorted(set(labels_named.tolist()))
        class_color_map = build_color_map(label_names)

        # Build split mask (applies to both views)
        show_train = "train" in (split_filter or [])
        show_val = "val" in (split_filter or [])
        mask = np.ones(labels_named.shape[0], dtype=bool)
        if not show_train:
            mask &= split_raw != 0
        if not show_val:
            mask &= split_raw != 1
        if not np.any(mask):
            return empty_figure("No points match the current split filter.", height=500), ""

        # MATRIX VIEW (scatter matrix via plotly.express.scatter_matrix)
        if view_mode == "matrix":
            target_layer = layers[0]

            # Prefer UMAP coords when selected; otherwise use PCs for multi-dim control.
            coords = None
            coord_label = "PC"
            if method == "umap":
                coord_label = "UMAP"
                coords = data.get(f"{target_layer}_umap")
                if coords is None:
                    coords = data.get(f"{target_layer}_{method}")
            if coords is None:
                coords = data.get(f"{target_layer}_pc")
                coord_label = "PC"
            if coords is None:
                coords = data.get(f"{target_layer}_pca")
                coord_label = "PC"
            if coords is None and method != "umap":
                coords = data.get(f"{target_layer}_{method}")
            if coords is None:
                coords = data.get(f"{target_layer}_umap")
                coord_label = "UMAP"
            if coords is None:
                return empty_figure(f"No projection data for '{target_layer}'.", height=500), ""
            if coords.ndim != 2 or coords.shape[1] < 2:
                return empty_figure(f"'{target_layer}' has fewer than 2 components.", height=500), ""

            requested_components = int(pcs_value or 3)
            n_components = max(2, min(coords.shape[1], requested_components))

            if coords.shape[0] != labels_named.shape[0]:
                return (
                    empty_figure(
                        f"Data mismatch: {coords.shape[0]} points vs {labels_named.shape[0]} labels.",
                        height=500,
                    ),
                    "",
                )

            dims = [f"{coord_label}{i+1}" for i in range(n_components)]
            df = pd.DataFrame(coords[:, :n_components], columns=dims)
            df["label"] = labels_named
            df["split"] = split_names
            df = df[mask]
            if df.empty:
                return empty_figure("No points selected (check Split).", height=500), ""

            auto_zoom = "zoom" in (zoom_opts or [])
            clip_msg = ""
            axis_ranges = None
            if auto_zoom:
                coords_masked = coords[mask][:, :n_components]
                axis_ranges = compute_component_ranges(
                    coords_masked, percentile=float(zoom_pct or 99)
                )
                if axis_ranges:
                    in_range = np.ones(len(df), dtype=bool)
                    for dim, (lo, hi) in zip(dims, axis_ranges):
                        in_range &= (df[dim] >= lo) & (df[dim] <= hi)
                    df = df[in_range]
                    clip_msg = f" (clipped pct {float(zoom_pct or 99):.1f})"
                    if df.empty:
                        return empty_figure("No points after outlier clipping.", height=500), ""

            # Scatter-matrices can get heavy; cap point count to keep the UI responsive.
            max_points = 8000
            if disable_sampling:
                max_points = 1_000_000

            sampled_msg = ""
            n_visible = len(df)
            if n_visible > max_points:
                rng = np.random.default_rng(int(rep_epoch) if rep_epoch is not None else 0)
                label_arr = df["label"].to_numpy()
                uniq, counts = np.unique(label_arr, return_counts=True)
                total = int(counts.sum())
                alloc = np.maximum(1, np.floor(max_points * counts / max(total, 1)).astype(int))
                delta = int(alloc.sum() - max_points)
                if delta > 0:
                    order = np.argsort(-alloc)
                    for i in order:
                        if delta <= 0:
                            break
                        if alloc[i] > 1:
                            take = min(delta, int(alloc[i] - 1))
                            alloc[i] -= take
                            delta -= take
                elif delta < 0:
                    order = np.argsort(-counts)
                    for i in order:
                        if delta >= 0:
                            break
                        alloc[i] += 1
                        delta += 1
                
                sampled_idx = []
                for lbl, n in zip(uniq, alloc):
                    idx = np.flatnonzero(label_arr == lbl)
                    if len(idx) <= n:
                        sampled_idx.extend(idx.tolist())
                    else:
                        sampled_idx.extend(rng.choice(idx, size=int(n), replace=False).tolist())
                df = df.iloc[sampled_idx]
                sampled_msg = f" (sampled {len(df):,}/{n_visible:,} points)"
            elif disable_sampling and n_visible > 8000:
                 sampled_msg = f" (showing all {n_visible:,} points - heavy!)"

            fig = px.scatter_matrix(
                df,
                dimensions=dims,
                color="label",
                symbol="split",
                color_discrete_map=class_color_map,
                symbol_map={"train": "circle", "val": "diamond"},
                opacity=0.7,
                hover_data={"split": True},
            )
            fig.update_traces(diagonal_visible=False, showupperhalf=True, showlowerhalf=True)

            # Compact axis titles for SPLOM.
            axis_title_size = 9
            for i in range(1, n_components + 1):
                axis_layout = {
                    "title": {"font": {"size": axis_title_size, "family": FONT_FAMILY, "color": GROK_TEXT}}
                }
                if axis_ranges:
                    axis_layout["range"] = axis_ranges[i - 1]
                fig.update_layout(
                    **{
                        f"xaxis{i}": axis_layout,
                        f"yaxis{i}": axis_layout,
                    }
                )

            apply_grok_layout(fig, height=800, showlegend=True)
            fig.update_layout(dragmode="select")
            
            # Persist zoom/pan state when refreshing data, unless axes change drastically
            # Use run_name + layer + method + n_components so switching context resets zoom, but staying within it persists
            uirevision_key = f"{run_name}-{target_layer}-{method}-matrix"
            fig.update_layout(uirevision=uirevision_key)
            
            fig.update_annotations(font=dict(color=GROK_TEXT, family=FONT_FAMILY, size=12))
            if len(label_names) > 12:
                _apply_vertical_legend_right(fig, width_px=280)

            if coord_label == "UMAP":
                status = f"Scatter Matrix: {target_layer} (UMAP {n_components}D){clip_msg}{sampled_msg}"
            else:
                status = f"Scatter Matrix: {target_layer} ({n_components} PCs){clip_msg}{sampled_msg}"
            return fig, status

        # GRID VIEW (standard): one 2D scatter per selected layer.
        cols = max(1, int(grid_cols or 2))
        rows = int(np.ceil(len(layers) / cols))
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=layers)
        any_coords = False
        trace_count = 0

        for idx, layer in enumerate(layers):
            coords = None
            if method == "umap":
                coords = data.get(f"{layer}_umap")
            else:
                coords = data.get(f"{layer}_pca")
            if coords is None:
                pc = data.get(f"{layer}_pc")
                if pc is not None and pc.ndim == 2 and pc.shape[1] >= 2:
                    coords = pc[:, :2]
            if coords is None:
                continue
            any_coords = True

            coords_masked = coords[mask]

            # Compute zoom ranges on visible points
            auto_zoom = "zoom" in (zoom_opts or [])
            x_range, y_range = None, None
            if auto_zoom:
                ranges = compute_zoom_ranges(coords_masked, percentile=float(zoom_pct or 99))
                if ranges:
                    x_range, y_range = ranges

            row = idx // cols + 1
            col = idx % cols + 1

            # One legend entry per class; render train/val as separate traces.
            for class_name in label_names:
                class_mask_base = labels_named == class_name
                if not np.any(class_mask_base):
                    continue

                for split_value, split_name, symbol in (
                    (0, "train", "circle"),
                    (1, "val", "diamond"),
                ):
                    split_mask = mask & class_mask_base & (split_raw == split_value)
                    if not np.any(split_mask):
                        continue
                    show_legend = (idx == 0) and (split_value == 0)
                    fig.add_trace(
                        go.Scattergl(
                            x=coords[split_mask, 0],
                            y=coords[split_mask, 1],
                            mode="markers",
                            name=str(class_name),
                            legendgroup=str(class_name),
                            showlegend=show_legend,
                            marker=dict(
                                size=5,
                                color=class_color_map.get(str(class_name), GROK_TEXT),
                                symbol=symbol,
                                opacity=0.85,
                                line=dict(width=0),
                            ),
                            hovertemplate=f"label={class_name}<br>split={split_name}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )
                    trace_count += 1

            if x_range and y_range:
                fig.update_xaxes(range=x_range, row=row, col=col)
                fig.update_yaxes(range=y_range, row=row, col=col)

        if not any_coords:
            return empty_figure("No projection data for selected layers/method.", height=500), ""
        if trace_count == 0:
            return empty_figure("No points match the current split filter.", height=500), ""

        height = 320 * rows + 80
        apply_grok_layout(fig, height=height, showlegend=True)
        fig.update_annotations(font=dict(color=GROK_TEXT, family=FONT_FAMILY, size=12))
        fig.update_layout(margin=dict(l=40, r=20, t=170, b=40), legend=dict(y=1.18, x=0.0))
        fig.update_xaxes(
            showgrid=True,
            gridcolor=GROK_GRID,
            zeroline=False,
            tickfont=dict(color=GROK_MUTED, family=FONT_FAMILY, size=12),
            title_font=dict(color=GROK_TEXT, family=FONT_FAMILY),
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=GROK_GRID,
            zeroline=False,
            tickfont=dict(color=GROK_MUTED, family=FONT_FAMILY, size=12),
            title_font=dict(color=GROK_TEXT, family=FONT_FAMILY),
        )
        return fig, ""

    @app.callback(
        Output("proj-zoom-label", "children"),
        Input("proj-zoom-pct", "value"),
    )
    def update_proj_zoom_label(val):
        return f"Pct: {val}"

    @app.callback(
        Output("proj-graph", "figure"),
        Output("proj-status", "children"),
        Input("run-select", "value"),
        Input("proj-epoch", "value"),
        Input("proj-layer", "value"),
        Input("proj-method", "value"),
        Input("proj-auto-zoom", "value"),
        Input("proj-zoom-pct", "value"),
        Input("global-epoch", "data"),
        Input("link-global", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
        State("proj-epoch", "options"),
    )
    def update_projection(run_name, proj_epoch_val, layer, method, zoom_opts, zoom_pct, global_epoch, link_global, _n_intervals, _n_clicks, proj_options):
        # Determine epoch
        proj_epoch = proj_epoch_val
        if "linked" in (link_global or []) and global_epoch is not None and proj_options:
             vals = [o["value"] for o in proj_options]
             if global_epoch in vals:
                 proj_epoch = global_epoch
             else:
                 proj_epoch = min(vals, key=lambda x: abs(x - global_epoch))

        if not run_name or proj_epoch is None or not layer:
            return empty_figure("No projection data available.", height=500), ""
        run_dir = runs_dir / run_name
        proj_path = run_dir / "projections" / f"epoch_{proj_epoch:03d}.npz"
        data, message = safe_read_npz(proj_path)
        if data is None:
            return empty_figure(message or "Projection file not ready.", height=500), ""
        key = f"{layer}_{method}"
        if key not in data:
            return empty_figure("Selected projection not found.", height=500), ""
        coords = data[key]
        labels_raw = data.get("labels")
        if labels_raw is None:
            return empty_figure("Projection labels missing.", height=500), ""
        class_names = load_config(run_dir).get("classes")
        labels_named = map_labels(labels_raw, class_names)
        label_names = (
            class_names if class_names else sorted(set(labels_named.tolist()))
        )
        class_color_map = build_color_map(label_names)
        
        # Auto-Zoom
        auto_zoom = "zoom" in (zoom_opts or [])
        x_range, y_range = None, None
        if auto_zoom:
             ranges = compute_zoom_ranges(coords, percentile=float(zoom_pct or 99))
             if ranges:
                 x_range, y_range = ranges
        
        fig = px.scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            color=labels_named,
            color_discrete_map=class_color_map,
            opacity=0.85,
            render_mode="webgl",
        )
        if x_range and y_range:
            fig.update_xaxes(range=x_range)
            fig.update_yaxes(range=y_range)
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
        apply_grok_layout(fig, height=500, showlegend=True)
        if len(label_names) > 12:
            _apply_vertical_legend_right(fig, width_px=280)
        return fig, ""

    @app.callback(
        Output("surface-graph", "figure"),
        Output("surface-status", "children"),
        Input("run-select", "value"),
        Input("surface-epoch", "value"),
        Input("surface-metric", "value"),
        Input("surface-class", "value"),
        Input("surface-overlay", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
    )
    def update_surface(
        run_name,
        surface_epoch,
        metric,
        class_idx,
        overlay_opts,
        _n_intervals,
        _n_clicks,
    ):
        if not run_name or surface_epoch is None:
            return empty_figure("No surface data available.", height=500), ""
        run_dir = runs_dir / run_name
        surface_path = run_dir / "surfaces" / f"epoch_{surface_epoch:03d}.npz"
        data, message = safe_read_npz(surface_path)
        if data is None:
            return empty_figure(message or "Surface file not ready.", height=500), ""
        grid_x = data["grid_x"][0]
        grid_y = data["grid_y"][:, 0]
        proba = data["proba"].astype(np.float32)
        if metric == "Class probability" and class_idx is None:
            class_idx = 0
        z = surface_metric(proba, metric, class_idx)
        fig = go.Figure(
            data=[
                go.Heatmap(
                    x=grid_x,
                    y=grid_y,
                    z=z,
                    colorscale=GROK_SEQ_SCALE,
                    showscale=False,
                )
            ]
        )
        if overlay_opts and "on" in overlay_opts:
            ref_proj = data.get("ref_proj")
            ref_labels = data.get("ref_labels")
            if ref_proj is not None and ref_labels is not None:
                class_names = load_config(run_dir).get("classes")
                labels_named = map_labels(ref_labels, class_names)
                label_names = (
                    class_names if class_names else sorted(set(labels_named.tolist()))
                )
                class_color_map = build_color_map(label_names)
                overlay_fig = px.scatter(
                    x=ref_proj[:, 0],
                    y=ref_proj[:, 1],
                    color=labels_named,
                    color_discrete_map=class_color_map,
                    opacity=0.6,
                    render_mode="webgl",
                )
                for trace in overlay_fig.data:
                    fig.add_trace(trace)
        apply_grok_layout(fig, height=500, showlegend=True)
        status = ""
        if overlay_opts and "on" in overlay_opts:
            # Legends can get huge with many classes; keep them readable.
            if "label_names" in locals() and len(label_names) > 12:
                _apply_vertical_legend_right(fig, width_px=280)
                status = "Legend moved to the right (many classes)."
        return fig, status

    @app.callback(
        Output("probe-graph", "figure"),
        Output("probe-status", "children"),
        Input("run-select", "value"),
        Input("probe-metric", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
    )
    def update_probes(run_name, metric, _n_intervals, _n_clicks):
        if not run_name:
            return empty_figure("No probe data available.", height=420), ""
        run_dir = runs_dir / run_name
        probes_path = run_dir / "probes.csv"
        probes_df, message = safe_read_csv(probes_path)
        if probes_df is None:
            return empty_figure(message or "probes.csv not found.", height=420), ""
        if metric not in probes_df.columns:
            return empty_figure("Selected metric not found.", height=420), ""
        fig = px.line(
            probes_df,
            x="epoch",
            y=metric,
            color="layer",
            line_group="layer",
        )
        fig.update_traces(line=dict(width=2))
        apply_grok_layout(fig, height=420, showlegend=True)
        return fig, ""

    @app.callback(
        Output("interp-graph", "figure"),
        Output("interp-info", "children"),
        Input("run-select", "value"),
        Input("interp-epoch", "value"),
        Input("interp-pair", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
    )
    def update_interp(run_name, interp_epoch, pair_idx, _n_intervals, _n_clicks):
        if not run_name or interp_epoch is None:
            return empty_figure("No interpolation data available.", height=360), ""
        run_dir = runs_dir / run_name
        interp_path = run_dir / "interpolations" / f"epoch_{interp_epoch:03d}.npz"
        data, message = safe_read_npz(interp_path)
        if data is None:
            return empty_figure(message or "Interpolation file not ready.", height=360), ""
        t = data["t"]
        proba_a = data["proba_a"]
        proba_b = data["proba_b"]
        pair_labels = data["pair_labels"]
        pair_types = data.get("pair_types")
        n_pairs = proba_a.shape[0]
        idx = int(pair_idx or 0)
        if idx >= n_pairs:
            idx = n_pairs - 1
        class_names = load_config(run_dir).get("classes")
        label_a = int(pair_labels[idx][0])
        label_b = int(pair_labels[idx][1])
        name_a = (
            class_names[label_a]
            if class_names and label_a < len(class_names)
            else str(label_a)
        )
        name_b = (
            class_names[label_b]
            if class_names and label_b < len(class_names)
            else str(label_b)
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=t,
                y=proba_a[idx],
                name=f"P({name_a})",
                mode="lines",
                line=dict(color=NEON_YELLOW, width=3),
            )
        )
        if label_a != label_b:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=proba_b[idx],
                    name=f"P({name_b})",
                    mode="lines",
                    line=dict(color=NEON_MAGENTA, width=3),
                )
            )
        apply_grok_layout(fig, height=360, showlegend=True)
        kind = ""
        if pair_types is not None:
            kind = "same-class" if int(pair_types[idx]) == 0 else "different-class"
        info = f"Pair {idx} ({kind}): {name_a} -> {name_b}"
        return fig, info

    @app.callback(
        Output("resp-graph", "figure"),
        Output("resp-status", "children"),
        Input("run-select", "value"),
        Input("resp-epoch", "value"),
        Input("resp-view", "value"),
        Input("resp-unit", "value"),
        Input("resp-units", "value"),
        Input("resp-cols", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
    )
    def update_responses(
        run_name,
        resp_epoch,
        view_mode,
        unit_value,
        unit_values,
        cols_value,
        _n_intervals,
        _n_clicks,
    ):
        if not run_name or resp_epoch is None:
            return empty_figure("No response data available.", height=360), ""
        run_dir = runs_dir / run_name
        resp_path = run_dir / "responses" / f"epoch_{resp_epoch:03d}.npz"
        data, message = safe_read_npz(resp_path)
        if data is None:
            return empty_figure(message or "Response file not ready.", height=360), ""
        pc1 = data["pc1"]
        activations = data["activations"]
        unit_indices = data["unit_indices"].astype(int).tolist()
        labels_raw = data["labels"]
        class_names = load_config(run_dir).get("classes")
        labels_named = map_labels(labels_raw, class_names)
        label_names = (
            class_names if class_names else sorted(set(labels_named.tolist()))
        )
        class_color_map = build_color_map(label_names)

        if view_mode == "grid":
            if not unit_values:
                return empty_figure("Select units to display.", height=360), ""
            cols = int(cols_value or 2)
            rows = int(np.ceil(len(unit_values) / cols))
            fig = make_subplots(rows=rows, cols=cols)
            for idx, unit in enumerate(unit_values):
                if unit not in unit_indices:
                    continue
                unit_pos = unit_indices.index(unit)
                layer_fig = px.scatter(
                    x=pc1,
                    y=activations[:, unit_pos],
                    color=labels_named,
                    color_discrete_map=class_color_map,
                    opacity=0.85,
                    render_mode="webgl",
                )
                row = idx // cols + 1
                col = idx % cols + 1
                for trace in layer_fig.data:
                    trace.showlegend = idx == 0
                    fig.add_trace(trace, row=row, col=col)
            height = 260 * rows + 120
            apply_grok_layout(fig, height=height, showlegend=True)
            fig.update_xaxes(showgrid=True, gridcolor=GROK_GRID, zeroline=False)
            fig.update_yaxes(showgrid=True, gridcolor=GROK_GRID, zeroline=False)
            return fig, ""
        if unit_value is None:
            return empty_figure("Select a unit to display.", height=360), ""
        if unit_value not in unit_indices:
            return empty_figure("Selected unit not found.", height=360), ""
        unit_pos = unit_indices.index(unit_value)
        fig = px.scatter(
            x=pc1,
            y=activations[:, unit_pos],
            color=labels_named,
            color_discrete_map=class_color_map,
            opacity=0.85,
            render_mode="webgl",
        )
        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
        apply_grok_layout(fig, height=360, showlegend=True)
        return fig, ""

    @app.callback(
        Output("land-graph", "figure"),
        Output("land-status", "children"),
        Input("run-select", "value"),
        Input("land-epoch", "value"),
        Input("land-view", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
    )
    def update_landscape(run_name, land_epoch, view, _n_intervals, _n_clicks):
        if not run_name or land_epoch is None:
            return empty_figure("No landscape data available.", height=500), ""
        run_dir = runs_dir / run_name
        land_path = run_dir / "landscapes" / f"epoch_{land_epoch:03d}.npz"
        data, message = safe_read_npz(land_path)
        if data is None:
            return empty_figure(message or "Landscape file not ready.", height=500), ""
        alphas = data["alphas"]
        betas = data["betas"]
        loss = data["loss"]
        if view == "3d":
            x_grid, y_grid = np.meshgrid(betas, alphas)
            fig = go.Figure(
                data=[
                    go.Surface(
                        x=x_grid,
                        y=y_grid,
                        z=loss,
                        colorscale=GROK_SEQ_SCALE,
                        showscale=False,
                    )
                ]
            )
            fig.update_layout(
                height=500,
                margin=dict(l=30, r=20, t=40, b=40),
                paper_bgcolor=GROK_BG,
                plot_bgcolor=GROK_BG,
                font=dict(color=GROK_TEXT, family=FONT_FAMILY),
                scene=dict(
                    xaxis=dict(
                        backgroundcolor=GROK_BG,
                        gridcolor=GROK_GRID,
                        color=GROK_TEXT,
                        tickfont=dict(color=GROK_MUTED, family=FONT_FAMILY, size=12),
                        titlefont=dict(color=GROK_TEXT, family=FONT_FAMILY),
                    ),
                    yaxis=dict(
                        backgroundcolor=GROK_BG,
                        gridcolor=GROK_GRID,
                        color=GROK_TEXT,
                        tickfont=dict(color=GROK_MUTED, family=FONT_FAMILY, size=12),
                        titlefont=dict(color=GROK_TEXT, family=FONT_FAMILY),
                    ),
                    zaxis=dict(
                        backgroundcolor=GROK_BG,
                        gridcolor=GROK_GRID,
                        color=GROK_TEXT,
                        tickfont=dict(color=GROK_MUTED, family=FONT_FAMILY, size=12),
                        titlefont=dict(color=GROK_TEXT, family=FONT_FAMILY),
                    ),
                ),
            )
            return fig, ""
        fig = px.imshow(
            loss,
            x=betas,
            y=alphas,
            aspect="auto",
            origin="lower",
            color_continuous_scale=GROK_SEQ_SCALE,
        )
        apply_grok_layout(fig, height=500, showlegend=False)
        return fig, ""

    @app.callback(
        Output("film-graph", "figure"),
        Output("film-status", "children"),
        Input("run-select", "value"),
        Input("film-artifact", "value"),
        Input("film-cols", "value"),
        Input("film-panels", "value"),
        Input("film-stride", "value"),
        Input("film-anchor", "value"),
        Input("film-layer", "value"),
        Input("film-method", "value"),
        Input("film-surface-metric", "value"),
        Input("film-surface-class", "value"),
        Input("film-probe-metric", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
    )
    def update_filmstrip(
        run_name,
        artifact,
        cols_value,
        panels_value,
        stride_value,
        anchor,
        layer,
        method,
        surface_metric_choice,
        surface_class,
        probe_metric,
        _n_intervals,
        _n_clicks,
    ):
        if not run_name:
            return empty_figure("No filmstrip data available.", height=500), ""
        run_dir = runs_dir / run_name
        cols = int(cols_value or 3)
        panels = int(panels_value or 6)
        stride = int(stride_value or 1)

        if artifact == "surface":
            epochs = list_epochs(run_dir / "surfaces")
            if not epochs:
                return empty_figure("No decision surface files.", height=500), ""
            selected = select_filmstrip_epochs(epochs, panels, stride, anchor)
            rows = int(np.ceil(len(selected) / cols))
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[f"Epoch {e}" for e in selected],
            )
            for idx, epoch in enumerate(selected):
                surface_path = run_dir / "surfaces" / f"epoch_{epoch:03d}.npz"
                data, _ = safe_read_npz(surface_path)
                if data is None:
                    continue
                grid_x = data["grid_x"][0]
                grid_y = data["grid_y"][:, 0]
                proba = data["proba"].astype(np.float32)
                class_idx = surface_class
                if surface_metric_choice == "Class probability" and class_idx is None:
                    class_idx = 0
                z = surface_metric(proba, surface_metric_choice, class_idx)
                row = idx // cols + 1
                col = idx % cols + 1
                fig.add_trace(
                    go.Heatmap(
                        x=grid_x,
                        y=grid_y,
                        z=z,
                        colorscale=GROK_SEQ_SCALE,
                        showscale=False,
                    ),
                    row=row,
                    col=col,
                )
            height = 260 * rows + 140
            apply_grok_layout(fig, height=height, showlegend=False)
            return fig, ""

        if artifact == "proj":
            epochs = list_epochs(run_dir / "projections")
            if not epochs:
                return empty_figure("No projection files.", height=500), ""
            if not layer:
                return empty_figure("Select a layer.", height=500), ""
            selected = select_filmstrip_epochs(epochs, panels, stride, anchor)
            rows = int(np.ceil(len(selected) / cols))
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[f"Epoch {e}" for e in selected],
            )
            class_names = load_config(run_dir).get("classes")
            for idx, epoch in enumerate(selected):
                proj_path = run_dir / "projections" / f"epoch_{epoch:03d}.npz"
                data, _ = safe_read_npz(proj_path)
                if data is None:
                    continue
                key = f"{layer}_{method}"
                if key not in data:
                    continue
                coords = data[key]
                labels_raw = data.get("labels")
                if labels_raw is None:
                    continue
                labels_named = map_labels(labels_raw, class_names)
                label_names = (
                    class_names if class_names else sorted(set(labels_named.tolist()))
                )
                class_color_map = build_color_map(label_names)
                scatter_fig = px.scatter(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    color=labels_named,
                    color_discrete_map=class_color_map,
                    opacity=0.85,
                    render_mode="webgl",
                )
                row = idx // cols + 1
                col = idx % cols + 1
                for trace in scatter_fig.data:
                    trace.showlegend = False
                    fig.add_trace(trace, row=row, col=col)
            height = 280 * rows + 140
            apply_grok_layout(fig, height=height, showlegend=False)
            fig.update_xaxes(showgrid=True, gridcolor=GROK_GRID, zeroline=False)
            fig.update_yaxes(showgrid=True, gridcolor=GROK_GRID, zeroline=False)
            return fig, "Legend hidden in filmstrip (too many entries). Use Single Proj for full legend."

        if artifact == "probe":
            probes_path = run_dir / "probes.csv"
            probes_df, message = safe_read_csv(probes_path)
            if probes_df is None:
                return empty_figure(message or "probes.csv not found.", height=500), ""
            epochs = sorted(probes_df["epoch"].dropna().unique().astype(int).tolist())
            if not epochs:
                return empty_figure("No probe epochs found.", height=500), ""
            selected = select_filmstrip_epochs(epochs, panels, stride, anchor)
            rows = int(np.ceil(len(selected) / cols))
            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[f"Epoch {e}" for e in selected],
            )
            for idx, epoch in enumerate(selected):
                snapshot = probes_df[probes_df["epoch"] == epoch]
                bar_fig = px.bar(
                    snapshot,
                    x="layer",
                    y=probe_metric,
                    color="layer",
                    color_discrete_sequence=GROK_COLORWAY,
                )
                row = idx // cols + 1
                col = idx % cols + 1
                for trace in bar_fig.data:
                    trace.showlegend = False
                    fig.add_trace(trace, row=row, col=col)
            height = 280 * rows + 140
            apply_grok_layout(fig, height=height, showlegend=False)
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor=GROK_GRID, zeroline=False)
            return fig, ""

        return empty_figure("Filmstrip view not available.", height=500), ""

    @app.callback(
        Output("snapshot-img", "src"),
        Output("snapshot-status", "children"),
        Input("run-select", "value"),
        Input("refresh-interval", "n_intervals"),
        Input("refresh-metrics", "n_clicks"),
    )
    def update_snapshots(run_name, _n_intervals, _n_clicks):
        if not run_name:
            return None, "No run selected."
        run_dir = runs_dir / run_name
        frames_dir = run_dir / "frames"
        if not frames_dir.exists():
            return None, "Frames directory not found."
        frames = sorted(frames_dir.glob("epoch_*.png"))
        if not frames:
            return None, "No snapshots yet."
        latest = frames[-1]
        encoded = encode_image(latest)
        if not encoded:
            return None, "Snapshot not ready."
        return f"data:image/png;base64,{encoded}", f"Snapshot: {latest.name}"

    @app.callback(
        Output("playground-interval", "disabled"),
        Output("playground-play", "data"),
        Output("playground-play-button", "children"),
        Output("playground-play-button", "style"),
        Input("playground-play-button", "n_clicks"),
        State("playground-play", "data"),
    )
    def toggle_playground_play(n_clicks, is_playing):
        if not n_clicks:
            return True, False, "? Play", {
                "flex": "1",
                "margin": "0",
                "background": "rgba(68, 170, 255, 0.2)",
                "border-color": NEON_BLUE,
            }
        new_state = not bool(is_playing)
        label = "? Pause" if new_state else "? Play"
        style = {
            "flex": "1",
            "margin": "0",
            "background": "rgba(68, 170, 255, 0.5)"
            if new_state
            else "rgba(68, 170, 255, 0.2)",
            "border-color": NEON_BLUE,
        }
        return not new_state, new_state, label, style

    @app.callback(
        Output("playground-state", "data"),
        Output("playground-boundary", "figure"),
        Output("playground-loss", "figure"),
        Output("playground-status", "children"),
        Input("playground-train", "n_clicks"),
        Input("playground-reset", "n_clicks"),
        Input("playground-interval", "n_intervals"),
        Input("playground-dataset", "value"),
        Input("playground-points", "value"),
        Input("playground-noise", "value"),
        Input("playground-split", "value"),
        Input("playground-seed", "value"),
        Input("playground-features", "value"),
        Input("playground-layer-count", "value"),
        Input("playground-layer-1", "value"),
        Input("playground-layer-2", "value"),
        Input("playground-layer-3", "value"),
        Input("playground-layer-4", "value"),
        Input("playground-activation", "value"),
        Input("playground-lr", "value"),
        Input("playground-l2", "value"),
        Input("playground-batch", "value"),
        Input("playground-steps", "value"),
        State("playground-state", "data"),
    )
    def update_playground(
        _train_clicks,
        _reset_clicks,
        _n_intervals,
        dataset,
        n_points,
        noise,
        split,
        seed,
        features,
        layer_count,
        layer_1,
        layer_2,
        layer_3,
        layer_4,
        activation,
        lr,
        l2,
        batch_size,
        steps_per_tick,
        state,
    ):
        ctx = dash.callback_context
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        dataset = dataset or "Moons"
        activation = activation or "tanh"
        layer_sizes = resolve_playground_layers(
            layer_count, layer_1, layer_2, layer_3, layer_4
        )
        features = features or ["x1", "x2"]

        config = {
            "dataset": dataset,
            "n_points": int(n_points or 600),
            "noise": float(noise or 0.0),
            "split": float(split or 0.2),
            "seed": int(seed or 0),
            "features": features,
            "layers": layer_sizes,
            "activation": activation,
            "lr": float(lr or 0.01),
            "l2": float(l2 or 0.0),
            "batch": int(batch_size or 64),
        }
        config_key = json.dumps(config, sort_keys=True)

        x, y = build_playground_dataset(
            config["dataset"], config["n_points"], config["noise"], config["seed"]
        )
        split_val = max(0.0, min(0.9, config["split"]))
        if split_val <= 0.0:
            x_train, y_train = x, y
            x_val, y_val = None, None
        else:
            try:
                x_train, x_val, y_train, y_val = train_test_split(
                    x,
                    y,
                    test_size=split_val,
                    random_state=config["seed"],
                    stratify=y,
                )
            except ValueError:
                x_train, x_val, y_train, y_val = train_test_split(
                    x,
                    y,
                    test_size=split_val,
                    random_state=config["seed"],
                )

        x_train_feat = compute_playground_features(x_train, features)
        x_val_feat = (
            compute_playground_features(x_val, features) if x_val is not None else None
        )

        config_changed = state is None or state.get("config_key") != config_key
        reset_triggered = trigger_id == "playground-reset" or config_changed
        if reset_triggered:
            state = init_playground_state(
                config_key,
                config["seed"],
                x_train_feat.shape[1],
                layer_sizes,
                activation,
            )

        model = build_playground_model(
            x_train_feat.shape[1], layer_sizes, activation
        )
        if state and state.get("state_dict"):
            model.load_state_dict(deserialize_state_dict(state["state_dict"]))

        if trigger_id in ("playground-train", "playground-interval") and not reset_triggered:
            state, model = train_playground_steps(
                model,
                state,
                x_train_feat,
                y_train,
                x_val_feat,
                y_val,
                config["lr"],
                config["l2"],
                config["batch"],
                int(steps_per_tick or 1),
                config["seed"],
            )

        boundary_fig = build_playground_boundary_figure(
            model, x_train, y_train, x_val, y_val, features
        )
        loss_fig = build_playground_loss_figure(state)

        status = "Ready."
        if state and state.get("loss_history"):
            last_loss = state["loss_history"][-1]
            last_acc = state["acc_history"][-1]
            last_val = state["val_acc_history"][-1]
            if np.isfinite(last_val):
                status = (
                    f"Step {state['step']} | Loss {last_loss:.4f} | "
                    f"Train {last_acc:.3f} | Val {last_val:.3f}"
                )
            else:
                status = (
                    f"Step {state['step']} | Loss {last_loss:.4f} | "
                    f"Train {last_acc:.3f}"
                )

        return state, boundary_fig, loss_fig, status

    @app.callback(
        Output("grok-sidebar", "className"),
        Output("grok-main", "className"),
        Output("sidebar-store", "data"),
        Input("sidebar-toggle", "n_clicks"),
        State("sidebar-store", "data"),
    )
    def toggle_sidebar(n_clicks, data):
        if n_clicks is None:
            return "grok-sidebar", "grok-main", data
        collapsed = not data.get("collapsed", False)
        sidebar_cls = "grok-sidebar collapsed" if collapsed else "grok-sidebar"
        main_cls = "grok-main collapsed" if collapsed else "grok-main"
        return sidebar_cls, main_cls, {"collapsed": collapsed}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug + hot reload.")
    args = parser.parse_args()
    
    # Suppress verbose werkzeug logging (GET / POST requests)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    log.propagate = False
    log.disabled = True
     
    runs_dir = Path(args.runs_dir)
    app = create_app(runs_dir)
    # Flask server logs can still be noisy even if werkzeug logger is quiet.
    try:
        app.server.logger.setLevel(logging.ERROR)
        app.server.logger.disabled = True
    except Exception:
        pass

    run_kwargs = dict(
        host=args.host,
        port=args.port,
        debug=args.debug,
        dev_tools_hot_reload=args.debug,
        dev_tools_ui=args.debug,
        dev_tools_silence_routes_logging=True,
        use_reloader=False,
    )
    try:
        app.run(**run_kwargs)
    except TypeError:
        # Older/newer Dash versions can have different `app.run` keyword support.
        for k in (
            "dev_tools_silence_routes_logging",
            "dev_tools_hot_reload",
            "dev_tools_ui",
            "use_reloader",
        ):
            run_kwargs.pop(k, None)
        app.run(**run_kwargs)


if __name__ == "__main__":
    main()
