import html
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

GROK_BG = "#0b0f14"
GROK_PANEL = "#0f141e"
GROK_TEXT = "#f3f4f7"
GROK_MUTED = "#b2b9c6"
GROK_GRID = "#283041"

NEON_CYAN = "#00ffff"
NEON_MAGENTA = "#ff00ff"
NEON_YELLOW = "#ffd35a"
NEON_BLUE = "#65c8d0"
NEON_GREEN = "#6c946f"
NEON_TAN = "#dfd0b9"
NEON_RED = "#f25f5c"

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
    [0.35, "#1a2b3d"],
    [0.6, "#3d5c6f"],
    [0.8, NEON_BLUE],
    [1.0, NEON_YELLOW],
]

GROK_DIVERGE_SCALE = [
    [0.0, NEON_CYAN],
    [0.5, GROK_BG],
    [1.0, NEON_MAGENTA],
]

GROK_MARGIN = dict(l=30, r=10, t=30, b=30)
LEGEND_OUTSIDE_RIGHT_MARGIN = 280
LEGEND_OUTSIDE_BOTTOM_MARGIN = 220
LEGEND_OUTSIDE_TOP_MARGIN = 180
FONT_FAMILY = "Space Grotesk"
AUTO_REFRESH_DEFAULT = 0
AUTO_REFRESH_GRACE = 1.0
MIN_PLAY_INTERVAL = 0.25
TAB_NAMES = [
    "Loss Curves",
    "Metrics",
    "Representation Evolution (Grid)",
    "Representation Evolution",
    "Embedding PCA Decision Surface",
    "Linear Probe Diagnostics",
    "Interpolation Curves",
    "Neuron Response Curves",
    "Loss Landscape",
    "Filmstrip Mode",
    "Snapshots",
]
NP_HASH = {np.ndarray: lambda arr: (arr.shape, str(arr.dtype))}
CHEAP_PLOTLY_CONFIG = {"staticPlot": True, "displayModeBar": False}
CHEAP_MODE = False


def inject_grok_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap');
        .stApp {
            background: #0b0f14;
            color: #f3f4f7;
            font-family: 'Space Grotesk', sans-serif;
        }
        [data-testid="stSidebar"] {
            background-color: #0d111a;
            border-right: 1px solid #1d2430;
        }
        h1, h2, h3, h4, h5 {
            color: #f3f4f7;
            letter-spacing: 0.02em;
        }
        .stCaption, .stMarkdown, .stText {
            color: #b2b9c6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_grok_template():
    axis_common = dict(
        showgrid=True,
        gridcolor=GROK_GRID,
        zeroline=False,
        showline=True,
        linecolor=GROK_GRID,
        tickcolor=GROK_GRID,
        tickfont=dict(color=GROK_MUTED),
        title_font=dict(color=GROK_TEXT),
    )
    return go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=GROK_BG,
            plot_bgcolor=GROK_BG,
            font=dict(family=FONT_FAMILY, color=GROK_TEXT),
            colorway=GROK_COLORWAY,
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1.0,
                font=dict(color=GROK_MUTED),
            ),
            hoverlabel=dict(bgcolor="#111827", font=dict(color=GROK_TEXT)),
            xaxis=axis_common,
            yaxis=axis_common,
        )
    )


GROK_TEMPLATE = build_grok_template()
px.defaults.template = GROK_TEMPLATE
px.defaults.color_discrete_sequence = GROK_COLORWAY


@st.cache_data(show_spinner=False)
def read_csv_cached(path_str, mtime, size):
    return pd.read_csv(path_str)


@st.cache_data(show_spinner=False)
def read_npz_cached(path_str, mtime, size):
    data = np.load(path_str, allow_pickle=False)
    return {k: data[k] for k in data.files}


def notify_artifact_update(message):
    if hasattr(st, "toast"):
        st.toast(message, icon="⏳")
    else:
        st.sidebar.warning(message)


def read_csv_fresh(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        stat = path.stat()
    except OSError:
        notify_artifact_update("Artifact is updating; retrying next refresh.")
        return None
    try:
        return read_csv_cached(str(path), stat.st_mtime, stat.st_size)
    except Exception:
        notify_artifact_update("Artifact is updating; retrying next refresh.")
        return None


def read_npz_fresh(path: Path) -> Optional[dict[str, np.ndarray]]:
    if not path.exists():
        return None
    try:
        stat = path.stat()
    except OSError:
        notify_artifact_update("Artifact is updating; retrying next refresh.")
        return None
    try:
        return read_npz_cached(str(path), stat.st_mtime, stat.st_size)
    except Exception:
        notify_artifact_update("Artifact is updating; retrying next refresh.")
        return None


@st.cache_data(show_spinner=False)
def list_runs_cached(runs_dir_str, mtime):
    runs_dir = Path(runs_dir_str)
    try:
        runs = [p for p in runs_dir.iterdir() if p.is_dir()]
    except OSError:
        return []
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def list_runs(runs_dir: Path):
    if not runs_dir.exists():
        return []
    try:
        mtime = runs_dir.stat().st_mtime
    except OSError:
        return []
    return list_runs_cached(str(runs_dir), mtime)


def load_config(run_dir):
    if run_dir is None:
        return {}
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except json.JSONDecodeError:
        notify_artifact_update("Config is updating; retrying next refresh.")
        return {}


@st.cache_data(show_spinner=False)
def list_epochs_cached(folder_str, mtime, pattern):
    folder = Path(folder_str)
    epochs = []
    try:
        for path in sorted(folder.glob(pattern)):
            try:
                epochs.append(int(path.stem.split("_")[-1]))
            except ValueError:
                continue
    except OSError:
        return []
    return sorted(set(epochs))


def list_epochs(folder: Path, pattern="epoch_*.npz"):
    if not folder.exists():
        return []
    try:
        mtime = folder.stat().st_mtime
    except OSError:
        return []
    return list_epochs_cached(str(folder), mtime, pattern)


def parse_rep_keys(data):
    layers_by_method = {}
    for key in data:
        if key.endswith("_pca") or key.endswith("_umap"):
            layer, method = key.rsplit("_", 1)
            layers_by_method.setdefault(method, set()).add(layer)
    methods = sorted(layers_by_method.keys())
    layers = {method: sorted(layers_by_method[method]) for method in methods}
    return methods, layers


def nearest_epoch(target, available):
    return min(available, key=lambda x: abs(x - target))


def previous_epoch(current, available):
    if current not in available:
        return None
    idx = available.index(current)
    if idx <= 0:
        return None
    return available[idx - 1]


def apply_grok_layout(
    fig, height=None, showlegend=True, margin=None, legend_outside=False
):
    layout_margin = dict(margin or GROK_MARGIN)
    legend = None
    legend_position = "right" if legend_outside is True else legend_outside
    if legend_position and showlegend:
        if legend_position == "bottom":
            layout_margin["b"] = max(
                layout_margin.get("b", 0), LEGEND_OUTSIDE_BOTTOM_MARGIN
            )
            legend = dict(
                orientation="h",
                x=0.0,
                xanchor="left",
                y=-0.25,
                yanchor="top",
            )
        elif legend_position == "top":
            layout_margin["t"] = max(
                layout_margin.get("t", 0), LEGEND_OUTSIDE_TOP_MARGIN
            )
            legend = dict(
                orientation="h",
                x=0.0,
                xanchor="left",
                y=1.12,
                yanchor="bottom",
            )
        else:
            layout_margin["r"] = max(
                layout_margin.get("r", 0), LEGEND_OUTSIDE_RIGHT_MARGIN
            )
            legend = dict(
                orientation="v",
                x=1.02,
                xanchor="left",
                y=1.0,
                yanchor="top",
            )
    fig.update_layout(
        template=GROK_TEMPLATE,
        height=height,
        showlegend=showlegend,
        margin=layout_margin,
        legend_title_text="",
    )
    if legend is not None:
        fig.update_layout(legend=legend)
    fig.update_xaxes(showgrid=True, gridcolor=GROK_GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GROK_GRID, zeroline=False)
    return fig


def mark_ui_event():
    st.session_state["last_ui_event"] = time.perf_counter()


def _legend_scalar(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 1:
            return value[0]
        return None
    return value


def render_external_legend(traces, key, max_cols=4):
    legend_traces = []
    for trace in traces:
        if not getattr(trace, "showlegend", True):
            continue
        name = getattr(trace, "name", None)
        if not name:
            continue
        marker = {}
        if hasattr(trace, "marker") and trace.marker is not None:
            color = _legend_scalar(trace.marker.color)
            symbol = _legend_scalar(trace.marker.symbol)
            if color is not None:
                marker["color"] = color
            if symbol is not None:
                marker["symbol"] = symbol
        marker["size"] = 8
        legend_traces.append(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                name=name,
                marker=marker,
                showlegend=True,
                hoverinfo="skip",
            )
        )
    if not legend_traces:
        return
    rows = max(1, math.ceil(len(legend_traces) / max_cols))
    height = min(280, 40 + rows * 22)
    legend_fig = go.Figure(legend_traces)
    legend_fig.update_layout(
        template=GROK_TEMPLATE,
        height=height,
        showlegend=True,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            x=0.0,
            xanchor="left",
            y=1.0,
            yanchor="top",
        ),
    )
    legend_fig.update_xaxes(visible=False)
    legend_fig.update_yaxes(visible=False)
    render_plotly(legend_fig, use_container_width=True, key=key)


def render_placeholder_plot(height, message):
    safe_message = html.escape(message)
    st.markdown(
        f"""
        <div style="height:{height}px; border-radius:12px; background:#0f141e;
                    border:1px solid #1d2430; display:flex; align-items:center;
                    justify-content:center; color:#b2b9c6; text-align:center;">
            {safe_message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def watch_scroll_events():
    return components.html(
        """
        <script>
        (function() {
            let timer = null;
            const send = () => {
                const y =
                    window.parent.scrollY ||
                    window.parent.pageYOffset ||
                    document.documentElement.scrollTop ||
                    0;
                window.parent.postMessage(
                    {isStreamlitMessage: true, type: "streamlit:setComponentValue", value: y},
                    "*"
                );
            };
            const schedule = () => {
                if (timer) clearTimeout(timer);
                timer = setTimeout(send, 250);
            };
            window.addEventListener("scroll", schedule, {passive: true});
            window.addEventListener("wheel", schedule, {passive: true});
            window.addEventListener("touchmove", schedule, {passive: true});
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def restore_scroll_position():
    scroll_y = st.session_state.get("scroll_y")
    if scroll_y is None:
        return
    try:
        scroll_y_int = int(scroll_y)
    except (TypeError, ValueError):
        return
    components.html(
        f"""
        <script>
        setTimeout(() => {{
            window.parent.scrollTo(0, {scroll_y_int});
        }}, 50);
        </script>
        """,
        height=0,
        width=0,
    )


def map_labels(labels, class_names):
    if class_names:
        return np.array([class_names[int(i)] for i in labels])
    return labels.astype(str)


def build_discrete_color_map(items, palette):
    items = [str(item) for item in items]
    return {item: palette[i % len(palette)] for i, item in enumerate(items)}


def surface_metric(proba, metric, class_idx=None):
    if metric == "Confidence (max prob)":
        return proba.max(axis=2)
    if metric == "Entropy":
        clipped = np.clip(proba, 1e-8, 1.0)
        return -(clipped * np.log(clipped)).sum(axis=2)
    if metric == "Class probability" and class_idx is not None:
        return proba[:, :, class_idx]
    return proba.max(axis=2)


def epoch_selector(label, available_epochs, global_epoch, key_prefix):
    if not available_epochs:
        return None
    link_key = f"{key_prefix}_link"
    epoch_key = f"{key_prefix}_epoch"
    link = st.checkbox(
        "Link to global epoch",
        value=st.session_state.get(link_key, True),
        key=link_key,
        on_change=mark_ui_event,
    )
    default_epoch = (
        global_epoch if global_epoch in available_epochs else available_epochs[-1]
    )
    if link:
        desired = nearest_epoch(global_epoch, available_epochs)
    else:
        desired = st.session_state.get(epoch_key, default_epoch)
        if desired not in available_epochs:
            desired = default_epoch
    if st.session_state.get(epoch_key) != desired:
        st.session_state[epoch_key] = desired
    if link:
        epoch = desired
        suffix = "" if epoch == global_epoch else " (nearest available)"
        st.caption(f"{label}: {epoch}{suffix}")
        return epoch
    selected_epoch = st.select_slider(
        label,
        options=available_epochs,
        value=desired,
        key=epoch_key,
        on_change=mark_ui_event,
    )
    return selected_epoch


def show_image(path, caption):
    try:
        st.image(path, caption=caption, use_container_width=True)
    except TypeError:
        st.image(path, caption=caption, use_column_width=True)


def select_filmstrip_epochs(available, count, stride, anchor_mode):
    if not available:
        return []
    sampled = available[::stride]
    if anchor_mode == "Latest":
        return sampled[-count:]
    return sampled[:count]


@st.cache_data(show_spinner=False, hash_funcs=NP_HASH)
def compute_zoom_ranges(
    coords,
    mask,
    zoom_pct,
    shared_across_layers=False,
    key="",
):
    _ = key
    _ = shared_across_layers
    if coords is None or coords.size == 0:
        return None
    if mask is not None:
        coords = coords[mask]
    if coords.size == 0:
        return None
    low = (100.0 - zoom_pct) / 2.0
    high = 100.0 - low
    x_min, x_max = np.percentile(coords[:, 0], [low, high])
    y_min, y_max = np.percentile(coords[:, 1], [low, high])
    return float(x_min), float(x_max), float(y_min), float(y_max)


@st.cache_data(show_spinner=False, hash_funcs=NP_HASH)
def compute_rep_surface_lr(coords_train, y_train, grid_res, pad, key=""):
    _ = key
    if coords_train is None or coords_train.size == 0:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return None
    x_min, x_max = coords_train[:, 0].min(), coords_train[:, 0].max()
    y_min, y_max = coords_train[:, 1].min(), coords_train[:, 1].max()
    span_x = max(x_max - x_min, 1e-6)
    span_y = max(y_max - y_min, 1e-6)
    pad_x = span_x * pad
    pad_y = span_y * pad
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min - pad_x, x_max + pad_x, grid_res),
        np.linspace(y_min - pad_y, y_max + pad_y, grid_res),
    )
    clf = LogisticRegression(max_iter=5000, n_jobs=1)
    clf.fit(coords_train, y_train)
    preds = clf.predict(np.c_[grid_x.ravel(), grid_y.ravel()])
    classes = clf.classes_
    label_to_idx = {label: i for i, label in enumerate(classes)}
    preds_idx = np.vectorize(label_to_idx.get)(preds).reshape(grid_x.shape)
    return grid_x, grid_y, preds_idx, classes


def render_plotly(fig, use_container_width=True, key=None):
    config = CHEAP_PLOTLY_CONFIG if CHEAP_MODE else {}
    st.plotly_chart(
        fig,
        use_container_width=use_container_width,
        config=config,
        key=key,
    )


def render_loss_curves(metrics_df):
    st.subheader("Loss Curves")
    if metrics_df is not None:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=metrics_df["epoch"],
                y=metrics_df["train_loss"],
                name="train loss",
                line=dict(width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=metrics_df["epoch"],
                y=metrics_df["val_loss"],
                name="val loss",
                line=dict(width=2),
            )
        )
        if "excl_loss" in metrics_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_df["epoch"],
                    y=metrics_df["excl_loss"],
                    name="excl loss",
                    line=dict(width=2),
                )
        )
        fig.update_yaxes(type="log")
        apply_grok_layout(fig, height=350)
        render_plotly(fig, use_container_width=True, key="loss_curves")
    else:
        st.info("metrics.csv not found yet.")


def render_metrics(metrics_df):
    st.subheader("Metrics")
    if metrics_df is not None:
        metric_fig = px.line(
            metrics_df,
            x="epoch",
            y=["val_accuracy", "val_f1_macro", "val_official_f1"],
            labels={"value": "score", "variable": "metric"},
        )
        metric_fig.update_traces(line=dict(width=2))
        apply_grok_layout(metric_fig, height=350)
        render_plotly(metric_fig, use_container_width=True, key="metrics")
    else:
        st.info("metrics.csv not found yet.")


def render_rep_grid(rep_dir, rep_epochs, global_epoch, class_names, metrics_df):
    st.subheader("Representation Evolution (Grid)")
    if rep_dir.exists():
        if not rep_epochs:
            st.info("No representation artifacts yet.")
        else:
            rep_epoch = epoch_selector(
                "Representation epoch",
                rep_epochs,
                global_epoch,
                "rep",
            )
            if rep_epoch is None:
                st.info("No representation epochs available.")
            else:
                rep_path = rep_dir / f"epoch_{rep_epoch:03d}.npz"
                if rep_path.exists():
                    rep_data = read_npz_fresh(rep_path)
                    if rep_data is None:
                        render_placeholder_plot(
                            320, "Representation file not ready yet."
                        )
                        return
                    try:
                        rep_mtime = rep_path.stat().st_mtime
                    except OSError:
                        rep_mtime = 0.0
                    methods, layers_by_method = parse_rep_keys(rep_data)
                    if not methods:
                        st.info("No representation projections found in this file.")
                    else:
                        method = (
                            st.radio(
                                "Projection",
                                methods,
                                horizontal=True,
                                key="rep_method",
                            )
                            if len(methods) > 1
                            else methods[0]
                        )
                        available_layers = layers_by_method.get(method, [])
                        default_layers = [
                            name
                            for name in ("input", "block1", "penultimate", "logits")
                            if name in available_layers
                        ]
                        if not default_layers:
                            default_layers = available_layers[
                                : min(4, len(available_layers))
                            ]
                        selected_layers = st.multiselect(
                            "Layers",
                            available_layers,
                            default=default_layers,
                            key="rep_layers",
                        )
                        view_mode = st.radio(
                            "View",
                            ["Grid", "Scatter-matrix"],
                            horizontal=True,
                            key="rep_view",
                        )
                        show_train = st.checkbox(
                            "Show train",
                            value=True,
                            key="rep_show_train",
                        )
                        show_val = st.checkbox(
                            "Show val",
                            value=True,
                            key="rep_show_val",
                        )
                        labels_raw = rep_data.get("labels")
                        split_raw = rep_data.get("split")
                        if labels_raw is None or split_raw is None:
                            render_placeholder_plot(
                                700, "Representation file missing labels/split metadata."
                            )
                        else:
                            labels_named = map_labels(labels_raw, class_names)
                            split_names = np.where(split_raw == 0, "train", "val")
                            label_names = (
                                class_names
                                if class_names
                                else sorted(set(labels_named.tolist()))
                            )
                            class_color_map = build_discrete_color_map(
                                label_names, GROK_COLORWAY
                            )
                            mask = np.ones(labels_named.shape[0], dtype=bool)
                            if not show_train:
                                mask &= split_raw != 0
                            if not show_val:
                                mask &= split_raw != 1

                            if view_mode == "Scatter-matrix":
                                layer = st.selectbox(
                                    "Layer",
                                    available_layers,
                                    index=available_layers.index(selected_layers[0])
                                    if selected_layers
                                    else 0,
                                    key="rep_matrix_layer",
                                )
                                pc_key = f"{layer}_pc"
                                if pc_key not in rep_data:
                                    st.info("PC data not found for this layer.")
                                else:
                                    pc_vals = rep_data[pc_key]
                                    max_pc = pc_vals.shape[1]
                                    max_dims = min(6, max_pc)
                                    default_dims = min(
                                        4 if CHEAP_MODE else max_dims, max_dims
                                    )
                                    dims = st.slider(
                                        "PCs",
                                        2,
                                        max_dims,
                                        default_dims,
                                        key="rep_matrix_pcs",
                                    )
                                    df = pd.DataFrame(
                                        pc_vals[:, :dims],
                                        columns=[f"PC{i+1}" for i in range(dims)],
                                    )
                                    df["label"] = labels_named
                                    df["split"] = split_names
                                    df = df[mask]
                                    fig = px.scatter_matrix(
                                        df,
                                        dimensions=[f"PC{i+1}" for i in range(dims)],
                                        color="label",
                                        color_discrete_map=class_color_map,
                                        opacity=0.7,
                                    )
                                    fig.update_traces(
                                        diagonal_visible=False,
                                        marker=dict(size=4),
                                    )
                                    render_external_legend(
                                        fig.data,
                                        key=f"rep_matrix_legend_{layer}_{rep_epoch}_{method}",
                                    )
                                    apply_grok_layout(
                                        fig,
                                        height=700,
                                        margin=GROK_MARGIN,
                                        showlegend=False,
                                    )
                                    render_plotly(
                                        fig,
                                        use_container_width=True,
                                        key=f"rep_matrix_{layer}",
                                    )
                            else:
                                if not selected_layers:
                                    st.info("Select at least one layer to display.")
                                else:
                                    auto_zoom = st.checkbox(
                                        "Auto-zoom (clip outliers)",
                                        value=True,
                                        key="rep_auto_zoom",
                                    )
                                    shared_axes = False
                                    axis_ranges = None
                                    zoom_pct = 99
                                    lock_axes = False
                                    mask_count = int(mask.sum())
                                    if auto_zoom:
                                        zoom_pct = st.slider(
                                            "Zoom percentile",
                                            90,
                                            100,
                                            99,
                                            key="rep_zoom_pct",
                                        )
                                        lock_axes_ui = st.checkbox(
                                            "Lock axes while playing",
                                            value=True,
                                            key="rep_lock_axes",
                                            disabled=not CHEAP_MODE,
                                        )
                                        lock_axes = CHEAP_MODE and lock_axes_ui
                                        shared_axes = st.checkbox(
                                            "Share axis range across layers",
                                            value=False,
                                            key="rep_shared_axes",
                                        )
                                        if shared_axes:
                                            combined = []
                                            for layer in selected_layers:
                                                proj_key = f"{layer}_{method}"
                                                if proj_key not in rep_data:
                                                    continue
                                                coords = rep_data[proj_key]
                                                coords_zoom = coords[mask]
                                                if coords_zoom.size == 0:
                                                    continue
                                                combined.append(coords_zoom)
                                            if combined:
                                                combined_coords = np.vstack(combined)
                                                axis_ranges = compute_zoom_ranges(
                                                    combined_coords,
                                                    None,
                                                    zoom_pct,
                                                    shared_across_layers=True,
                                                    key=(
                                                        f"rep_grid_shared_{'lock' if lock_axes else rep_epoch}_"
                                                        f"{method}_{zoom_pct}_"
                                                        f"{int(show_train)}_{int(show_val)}_"
                                                        f"{mask_count}_{rep_dir}"
                                                    ),
                                                )
                                    max_cols = min(4, max(1, len(selected_layers)))
                                    default_cols = min(3, max_cols)
                                    grid_cols = st.slider(
                                        "Grid columns",
                                        1,
                                        max_cols,
                                        default_cols,
                                        key="rep_grid_cols",
                                    )
                                    panel_height = 420 if grid_cols == 1 else 320
                                    cols = st.columns(grid_cols)
                                    legend_rendered = False
                                    for idx, layer in enumerate(selected_layers):
                                        proj_key = f"{layer}_{method}"
                                        if proj_key not in rep_data:
                                            continue
                                        coords = rep_data[proj_key]
                                        coords_zoom = coords[mask]
                                        df = pd.DataFrame(
                                            {
                                                "x": coords[:, 0],
                                                "y": coords[:, 1],
                                                "label": labels_named,
                                                "split": split_names,
                                            }
                                        )
                                        df = df[mask]
                                        fig = px.scatter(
                                            df,
                                            x="x",
                                            y="y",
                                            color="label",
                                            symbol="split",
                                            color_discrete_map=class_color_map,
                                            opacity=0.85,
                                            render_mode="webgl",
                                        )
                                        fig.update_traces(
                                            marker=dict(size=5, line=dict(width=0))
                                        )
                                        if auto_zoom and coords_zoom.size:
                                            ranges = axis_ranges
                                            if ranges is None:
                                                ranges = compute_zoom_ranges(
                                                    coords,
                                                    mask,
                                                    zoom_pct,
                                                    key=(
                                                        f"rep_grid_{'lock' if lock_axes else rep_epoch}_{layer}_"
                                                        f"{method}_{zoom_pct}_"
                                                        f"{int(show_train)}_{int(show_val)}_"
                                                        f"{mask_count}_{rep_dir}"
                                                    ),
                                                )
                                            if ranges is not None:
                                                x_min, x_max, y_min, y_max = ranges
                                                span_x = max(x_max - x_min, 1e-6)
                                                span_y = max(y_max - y_min, 1e-6)
                                                pad_x = span_x * 0.05
                                                pad_y = span_y * 0.05
                                                fig.update_xaxes(
                                                    range=[x_min - pad_x, x_max + pad_x]
                                                )
                                                fig.update_yaxes(
                                                    range=[y_min - pad_y, y_max + pad_y]
                                                )
                                        if not legend_rendered:
                                            render_external_legend(
                                                fig.data,
                                                key=(
                                                    f"rep_grid_legend_{rep_epoch}_"
                                                    f"{method}_{int(show_train)}_{int(show_val)}"
                                                ),
                                            )
                                            legend_rendered = True
                                        apply_grok_layout(
                                            fig,
                                            height=panel_height,
                                            showlegend=False,
                                        )
                                        with cols[idx % grid_cols]:
                                            render_plotly(
                                                fig,
                                                use_container_width=True,
                                                key=f"rep_grid_{layer}",
                                            )
                                            st.caption(layer)

                                show_surface = st.checkbox(
                                    "Show decision surface (2D)",
                                    value=False,
                                    key="rep_surface_toggle",
                                )
                                if show_surface and selected_layers:
                                    surface_layer = st.selectbox(
                                        "Surface layer",
                                        selected_layers,
                                        key="rep_surface_layer",
                                    )
                                    proj_key = f"{surface_layer}_{method}"
                                    if proj_key in rep_data:
                                        coords = rep_data[proj_key]
                                        labels_int = labels_raw.astype(int)
                                        split_mask = split_raw == 0
                                        if split_mask.sum() < 5:
                                            st.info(
                                                "Not enough train points for surface."
                                            )
                                        else:
                                            grid_res = 120 if CHEAP_MODE else 180
                                            surface_key = (
                                                f"rep_surface_{rep_epoch}_{surface_layer}_"
                                                f"{method}_{grid_res}_{rep_mtime}"
                                            )
                                            result = compute_rep_surface_lr(
                                                coords[split_mask],
                                                labels_int[split_mask],
                                                grid_res,
                                                0.05,
                                                key=surface_key,
                                            )
                                            if result is None:
                                                st.info(
                                                    "Decision surface not available."
                                                )
                                            else:
                                                grid_x, grid_y, preds_idx, classes = (
                                                    result
                                                )
                                                class_labels = [
                                                    class_names[int(label)]
                                                    if class_names
                                                    and int(label) < len(class_names)
                                                    else str(label)
                                                    for label in classes
                                                ]
                                                class_colors = build_discrete_color_map(
                                                    class_labels, GROK_COLORWAY
                                                )
                                                colorscale = [
                                                    (
                                                        i / max(1, len(classes) - 1),
                                                        class_colors[class_labels[i]],
                                                    )
                                                    for i in range(len(classes))
                                                ]
                                                fig = go.Figure()
                                                fig.add_trace(
                                                    go.Contour(
                                                        x=grid_x[0],
                                                        y=grid_y[:, 0],
                                                        z=preds_idx,
                                                        colorscale=colorscale,
                                                        showscale=False,
                                                        opacity=0.35,
                                                        showlegend=False,
                                                        contours=dict(
                                                            coloring="heatmap"
                                                        ),
                                                    )
                                                )
                                                df = pd.DataFrame(
                                                    {
                                                        "x": coords[:, 0],
                                                        "y": coords[:, 1],
                                                        "label": labels_named,
                                                        "split": split_names,
                                                    }
                                                )
                                                df = df[mask]
                                                fig_points = px.scatter(
                                                    df,
                                                    x="x",
                                                    y="y",
                                                    color="label",
                                                    symbol="split",
                                                    color_discrete_map=class_color_map,
                                                    opacity=0.9,
                                                    render_mode="webgl",
                                                )
                                                for trace in fig_points.data:
                                                    if hasattr(trace, "marker"):
                                                        trace.update(
                                                            marker=dict(
                                                                size=5,
                                                                line=dict(width=0),
                                                            )
                                                        )
                                                    fig.add_trace(trace)
                                                render_external_legend(
                                                    fig.data,
                                                    key=(
                                                        f"rep_surface_legend_{rep_epoch}_"
                                                        f"{surface_layer}_{method}"
                                                    ),
                                                )
                                                apply_grok_layout(
                                                    fig,
                                                    height=420,
                                                    showlegend=False,
                                                )
                                                render_plotly(
                                                    fig,
                                                    use_container_width=True,
                                                    key=f"rep_surface_{surface_layer}",
                                                )
                                    else:
                                        st.info(
                                            "Surface layer projection not available."
                                        )
                else:
                    render_placeholder_plot(
                        320, "No representation file found for selected epoch."
                    )
    else:
        st.info("Representation artifacts not found.")

    if metrics_df is not None:
        acc_columns = [
            col
            for col in ["train_accuracy", "val_accuracy"]
            if col in metrics_df.columns
        ]
        if acc_columns:
            acc_fig = px.line(
                metrics_df,
                x="epoch",
                y=acc_columns,
                labels={"value": "accuracy", "variable": "split"},
            )
            acc_fig.update_traces(line=dict(width=2))
            apply_grok_layout(acc_fig, height=260)
            render_plotly(acc_fig, use_container_width=True, key="rep_acc")


def render_rep_evolution(proj_dir, proj_epochs, global_epoch, class_names):
    st.subheader("Representation Evolution")
    if proj_dir.exists():
        if not proj_epochs:
            st.info("No projection files yet.")
        else:
            selected_epoch = epoch_selector(
                "Projection epoch", proj_epochs, global_epoch, "proj"
            )
            if selected_epoch is None:
                st.info("No projection epochs available.")
            else:
                proj_path = proj_dir / f"epoch_{selected_epoch:03d}.npz"
                if proj_path.exists():
                    data = read_npz_fresh(proj_path)
                    if data is None:
                        render_placeholder_plot(500, "Projection file not ready yet.")
                        return
                    try:
                        proj_mtime = proj_path.stat().st_mtime
                    except OSError:
                        proj_mtime = 0.0
                    available_layers = sorted(
                        {
                            k.split("_")[0]
                            for k in data
                            if k.endswith(("_pca", "_umap"))
                        }
                    )
                    if not available_layers:
                        st.info("No projection layers found in this file.")
                    else:
                        default_layer = (
                            "penultimate"
                            if "penultimate" in available_layers
                            else available_layers[-1]
                        )
                        layer = st.selectbox(
                            "Layer",
                            available_layers,
                            index=available_layers.index(default_layer),
                            key="proj_layer",
                        )
                        method = st.radio(
                            "Projection",
                            ["umap", "pca"],
                            horizontal=True,
                            key="proj_method",
                        )
                        key = f"{layer}_{method}"
                        if key in data:
                            coords = data[key]
                            labels = map_labels(data["labels"], class_names)
                            label_names = (
                                class_names
                                if class_names
                                else sorted(set(labels.tolist()))
                            )
                            class_color_map = build_discrete_color_map(
                                label_names, GROK_COLORWAY
                            )
                            fig = go.Figure()
                            prev_coords = None
                            ghost = st.checkbox(
                                "Ghost previous epoch",
                                value=True,
                                key="proj_ghost",
                            )
                            if ghost:
                                ghost_opacity = st.slider(
                                    "Ghost opacity",
                                    0.05,
                                    0.4,
                                    0.15,
                                    0.05,
                                    key="proj_ghost_opacity",
                                )
                                prev_epoch = previous_epoch(
                                    selected_epoch, proj_epochs
                                )
                                if prev_epoch is None:
                                    st.caption(
                                        "No previous epoch available for ghosting."
                                    )
                                else:
                                    prev_path = proj_dir / f"epoch_{prev_epoch:03d}.npz"
                                    if prev_path.exists():
                                        prev_data = read_npz_fresh(prev_path)
                                        if prev_data is not None and key in prev_data:
                                            prev_coords = prev_data[key]
                                            prev_labels = map_labels(
                                                prev_data["labels"], class_names
                                            )
                                            ghost_fig = px.scatter(
                                                x=prev_data[key][:, 0],
                                                y=prev_data[key][:, 1],
                                                color=prev_labels,
                                                color_discrete_map=class_color_map,
                                                opacity=ghost_opacity,
                                                render_mode="webgl",
                                            )
                                            for trace in ghost_fig.data:
                                                trace.showlegend = False
                                                fig.add_trace(trace)
                            current_fig = px.scatter(
                                x=coords[:, 0],
                                y=coords[:, 1],
                                color=labels,
                                color_discrete_map=class_color_map,
                                opacity=0.85,
                                render_mode="webgl",
                            )
                            for trace in current_fig.data:
                                fig.add_trace(trace)
                            fig.update_traces(marker=dict(size=6, line=dict(width=0)))
                            render_external_legend(
                                fig.data,
                                key=f"proj_legend_{selected_epoch}_{layer}_{method}",
                            )
                            auto_zoom = st.checkbox(
                                "Auto-zoom (clip outliers)",
                                value=True,
                                key="proj_auto_zoom",
                            )
                            if auto_zoom:
                                zoom_pct = st.slider(
                                    "Zoom percentile",
                                    90,
                                    100,
                                    99,
                                    key="proj_zoom_pct",
                                )
                                lock_axes_ui = st.checkbox(
                                    "Lock axes while playing",
                                    value=True,
                                    key="proj_lock_axes",
                                    disabled=not CHEAP_MODE,
                                )
                                lock_axes = CHEAP_MODE and lock_axes_ui
                                zoom_coords = (
                                    np.vstack([coords, prev_coords])
                                    if prev_coords is not None
                                    else coords
                                )
                                ranges = compute_zoom_ranges(
                                    zoom_coords,
                                    None,
                                    zoom_pct,
                                    key=(
                                        f"proj_zoom_{'lock' if lock_axes else selected_epoch}_"
                                        f"{layer}_{method}_{zoom_pct}_{proj_dir}_"
                                        f"{proj_mtime}_{int(prev_coords is not None)}"
                                    ),
                                )
                                if ranges is not None:
                                    x_min, x_max, y_min, y_max = ranges
                                    span_x = max(x_max - x_min, 1e-6)
                                    span_y = max(y_max - y_min, 1e-6)
                                    pad_x = span_x * 0.05
                                    pad_y = span_y * 0.05
                                    fig.update_xaxes(
                                        range=[x_min - pad_x, x_max + pad_x]
                                    )
                                    fig.update_yaxes(
                                        range=[y_min - pad_y, y_max + pad_y]
                                    )
                            apply_grok_layout(
                                fig,
                                height=500,
                                showlegend=False,
                            )
                            render_plotly(
                                fig,
                                use_container_width=True,
                                key=f"proj_{layer}_{method}",
                            )
                        else:
                            render_placeholder_plot(
                                500, "Selected projection not found for this epoch."
                            )
                else:
                    render_placeholder_plot(
                        500, "No projection file found for selected epoch."
                    )
    else:
        st.info("Projection artifacts not found.")


def render_decision_surface(surface_dir, surface_epochs, global_epoch, class_names):
    st.subheader("Embedding PCA Decision Surface")
    if surface_dir.exists():
        if not surface_epochs:
            st.info("No decision surface files yet.")
        else:
            surface_epoch = epoch_selector(
                "Surface epoch", surface_epochs, global_epoch, "surface"
            )
            if surface_epoch is None:
                st.info("No surface epochs available.")
            else:
                surface_path = surface_dir / f"epoch_{surface_epoch:03d}.npz"
                if surface_path.exists():
                    data = read_npz_fresh(surface_path)
                    if data is None:
                        render_placeholder_plot(500, "Surface file not ready yet.")
                        return
                    grid_x = data["grid_x"][0]
                    grid_y = data["grid_y"][:, 0]
                    proba = data["proba"].astype(np.float32)
                    ref_proj = data["ref_proj"]
                    ref_labels = data["ref_labels"]
                    metric_options = [
                        "Confidence (max prob)",
                        "Entropy",
                        "Class probability",
                    ]
                    metric_choice = st.selectbox(
                        "Surface metric",
                        metric_options,
                        index=0,
                        key="surface_metric",
                    )
                    class_idx = None
                    if metric_choice == "Class probability":
                        class_options = (
                            list(class_names)
                            if class_names
                            else [str(i) for i in range(proba.shape[2])]
                        )
                        class_choice = st.selectbox(
                            "Class", class_options, index=0, key="surface_class"
                        )
                        class_idx = class_options.index(class_choice)
                    z = surface_metric(proba, metric_choice, class_idx)
                    delta = st.checkbox(
                        "Delta from previous epoch",
                        value=False,
                        key="surface_delta",
                    )
                    if delta:
                        abs_delta = st.checkbox(
                            "Absolute delta", value=False, key="surface_abs"
                        )
                        prev_epoch = previous_epoch(surface_epoch, surface_epochs)
                        if prev_epoch is None:
                            st.info("No previous epoch available for delta view.")
                            delta = False
                        else:
                            prev_path = (
                                surface_dir / f"epoch_{prev_epoch:03d}.npz"
                            )
                            if prev_path.exists():
                                prev_data = read_npz_fresh(prev_path)
                                if prev_data is None:
                                    st.info("Previous surface file not ready yet.")
                                    delta = False
                                else:
                                    prev_proba = prev_data["proba"].astype(np.float32)
                                    prev_z = surface_metric(
                                        prev_proba, metric_choice, class_idx
                                    )
                                    z = z - prev_z
                                    if abs_delta:
                                        z = np.abs(z)
                            else:
                                st.info("Previous surface file not found.")
                                delta = False
                    view = st.radio(
                        "Surface view",
                        ["Contour", "Heatmap"],
                        horizontal=True,
                        key="surface_view",
                    )
                    colorscale = GROK_DIVERGE_SCALE if delta else GROK_SEQ_SCALE
                    trace_kwargs = dict(
                        z=z, x=grid_x, y=grid_y, colorscale=colorscale
                    )
                    if delta:
                        max_abs = max(float(np.max(np.abs(z))), 1e-6)
                        trace_kwargs.update(zmin=-max_abs, zmax=max_abs)
                    if view == "Contour":
                        trace_kwargs["contours"] = dict(showlabels=False)
                        trace = go.Contour(**trace_kwargs)
                    else:
                        trace = go.Heatmap(**trace_kwargs)
                    trace.showlegend = False
                    fig = go.Figure(data=[trace])
                    overlay = st.checkbox(
                        "Overlay val points",
                        value=True,
                        key="surface_overlay",
                    )
                    if overlay:
                        overlay_labels = map_labels(ref_labels, class_names)
                        label_names = (
                            class_names
                            if class_names
                            else sorted(set(overlay_labels.tolist()))
                        )
                        class_color_map = build_discrete_color_map(
                            label_names, GROK_COLORWAY
                        )
                        scatter_fig = px.scatter(
                            x=ref_proj[:, 0],
                            y=ref_proj[:, 1],
                            color=overlay_labels,
                            color_discrete_map=class_color_map,
                            opacity=0.6,
                            render_mode="webgl",
                        )
                        for trace in scatter_fig.data:
                            fig.add_trace(trace)
                        render_external_legend(
                            fig.data,
                            key=f"surface_legend_{surface_epoch}_{metric_choice}",
                        )
                    apply_grok_layout(
                        fig,
                        height=500,
                        showlegend=False,
                    )
                    render_plotly(fig, use_container_width=True, key="surface_plot")
                else:
                    render_placeholder_plot(
                        500, "No surface file found for selected epoch."
                    )
    else:
        st.info("Decision surface artifacts not found.")


def render_linear_probes(probes_df):
    st.subheader("Linear Probe Diagnostics")
    if probes_df is not None:
        metric = st.selectbox(
            "Probe metric", ["probe_acc", "probe_f1"], index=0, key="probe_metric"
        )
        smoothing_window = st.slider(
            "Smoothing window", 1, 15, 5, 1, key="probe_smoothing"
        )
        show_delta = st.checkbox("Delta per epoch", value=False)
        plot_df = probes_df.sort_values(["layer", "epoch"]).copy()
        y_col = metric
        if show_delta:
            plot_df["delta"] = plot_df.groupby("layer")[metric].diff()
            y_col = "delta"
        if smoothing_window > 1:
            plot_df["smooth"] = plot_df.groupby("layer")[y_col].transform(
                lambda s: s.rolling(
                    window=smoothing_window, min_periods=1, center=True
                ).mean()
            )
            y_col = "smooth"
        layers = sorted(plot_df["layer"].unique().tolist())
        layer_color_map = build_discrete_color_map(layers, GROK_COLORWAY)
        probe_fig = px.line(
            plot_df,
            x="epoch",
            y=y_col,
            color="layer",
            color_discrete_map=layer_color_map,
            markers=True,
        )
        probe_fig.update_traces(line=dict(width=2), line_shape="spline")
        if show_delta:
            probe_fig.add_hline(
                y=0,
                line_color=GROK_MUTED,
                line_dash="dot",
                opacity=0.6,
            )
        apply_grok_layout(probe_fig, height=350)
        render_plotly(probe_fig, use_container_width=True, key="probe_plot")
    else:
        st.info("probes.csv not found yet.")


def render_interpolation(interp_dir, interp_epochs, global_epoch, class_names):
    st.subheader("Interpolation Curves")
    if interp_dir.exists():
        if not interp_epochs:
            st.info("No interpolation files yet.")
        else:
            interp_epoch = epoch_selector(
                "Interpolation epoch", interp_epochs, global_epoch, "interp"
            )
            if interp_epoch is None:
                st.info("No interpolation epochs available.")
            else:
                interp_path = interp_dir / f"epoch_{interp_epoch:03d}.npz"
                if interp_path.exists():
                    data = read_npz_fresh(interp_path)
                    if data is None:
                        st.info("Interpolation file not ready yet.")
                        return
                    t = data["t"]
                    proba_a = data["proba_a"]
                    proba_b = data["proba_b"]
                    pair_labels = data["pair_labels"]
                    pair_types = data["pair_types"]
                    if proba_a.shape[0] == 0:
                        st.info("No interpolation pairs available.")
                    else:
                        pair_idx = st.slider(
                            "Pair index",
                            0,
                            proba_a.shape[0] - 1,
                            0,
                            key="interp_pair_idx",
                        )
                        label_a, label_b = pair_labels[pair_idx]
                        name_a = (
                            class_names[int(label_a)]
                            if class_names
                            else str(label_a)
                        )
                        name_b = (
                            class_names[int(label_b)]
                            if class_names
                            else str(label_b)
                        )
                        kind = (
                            "same-class"
                            if pair_types[pair_idx] == 0
                            else "different-class"
                        )
                        st.caption(
                            f"Pair {pair_idx} ({kind}): {name_a} -> {name_b}"
                        )
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=t,
                                y=proba_a[pair_idx],
                                name=f"P({name_a})",
                                mode="lines",
                                line=dict(
                                    color=NEON_YELLOW, width=3, shape="spline"
                                ),
                            )
                        )
                        if label_a != label_b:
                            fig.add_trace(
                                go.Scatter(
                                    x=t,
                                    y=proba_b[pair_idx],
                                    name=f"P({name_b})",
                                    mode="lines",
                                    line=dict(
                                        color=NEON_MAGENTA,
                                        width=3,
                                        shape="spline",
                                    ),
                                )
                            )
                        apply_grok_layout(fig, height=350)
                        render_plotly(
                            fig, use_container_width=True, key=f"interp_{pair_idx}"
                        )
                else:
                    st.info("No interpolation file found for selected epoch.")
    else:
        st.info("Interpolation artifacts not found.")


def render_neuron_responses(response_dir, response_epochs, global_epoch, class_names):
    st.subheader("Neuron Response Curves")
    if response_dir.exists():
        if not response_epochs:
            st.info("No response curve files yet.")
        else:
            response_epoch = epoch_selector(
                "Response epoch", response_epochs, global_epoch, "response"
            )
            if response_epoch is None:
                st.info("No response epochs available.")
            else:
                response_path = response_dir / f"epoch_{response_epoch:03d}.npz"
                if response_path.exists():
                    data = read_npz_fresh(response_path)
                    if data is None:
                        st.info("Response file not ready yet.")
                        return
                    pc1 = data["pc1"]
                    activations = data["activations"]
                    unit_indices = data["unit_indices"].astype(int)
                    labels = map_labels(data["labels"], class_names)
                    label_names = (
                        class_names if class_names else sorted(set(labels.tolist()))
                    )
                    class_color_map = build_discrete_color_map(
                        label_names, GROK_COLORWAY
                    )
                    unit_options = [int(u) for u in unit_indices]
                    view_mode = st.radio(
                        "View mode",
                        ["Single unit", "Small multiples"],
                        horizontal=True,
                        key="resp_view_mode",
                    )
                    if view_mode == "Single unit":
                        selected_unit = st.selectbox("Unit", unit_options, index=0)
                        unit_pos = unit_options.index(selected_unit)
                        fig = go.Figure()
                        compare_prev = st.checkbox(
                            "Ghost previous epoch",
                            value=False,
                            key="resp_ghost",
                        )
                        if compare_prev:
                            prev_epoch = previous_epoch(
                                response_epoch, response_epochs
                            )
                            if prev_epoch is None:
                                st.caption(
                                    "No previous epoch available for ghosting."
                                )
                            else:
                                prev_path = (
                                    response_dir / f"epoch_{prev_epoch:03d}.npz"
                                )
                                if prev_path.exists():
                                    prev_data = read_npz_fresh(prev_path)
                                    if prev_data is not None:
                                        prev_labels = map_labels(
                                            prev_data["labels"], class_names
                                        )
                                        ghost_fig = px.scatter(
                                            x=prev_data["pc1"],
                                            y=prev_data["activations"][:, unit_pos],
                                            color=prev_labels,
                                            color_discrete_map=class_color_map,
                                            opacity=0.15,
                                            render_mode="webgl",
                                        )
                                        for trace in ghost_fig.data:
                                            trace.showlegend = False
                                            fig.add_trace(trace)
                        current_fig = px.scatter(
                            x=pc1,
                            y=activations[:, unit_pos],
                            color=labels,
                            color_discrete_map=class_color_map,
                            opacity=0.85,
                            render_mode="webgl",
                        )
                        for trace in current_fig.data:
                            fig.add_trace(trace)
                        fig.update_traces(marker=dict(size=6, line=dict(width=0)))
                        apply_grok_layout(fig, height=350)
                        render_plotly(
                            fig,
                            use_container_width=True,
                            key=f"resp_unit_{selected_unit}",
                        )
                    else:
                        tracked_units = st.multiselect(
                            "Tracked units",
                            unit_options,
                            default=unit_options[: min(6, len(unit_options))],
                        )
                        if not tracked_units:
                            st.info("Select at least one unit to display.")
                        else:
                            columns = st.selectbox("Grid columns", [2, 3], index=1)
                            cols = st.columns(columns)
                            for idx, unit in enumerate(tracked_units):
                                unit_pos = unit_options.index(unit)
                                fig = px.scatter(
                                    x=pc1,
                                    y=activations[:, unit_pos],
                                    color=labels,
                                    color_discrete_map=class_color_map,
                                    opacity=0.85,
                                    render_mode="webgl",
                                )
                                fig.update_traces(
                                    marker=dict(size=5, line=dict(width=0))
                                )
                                showlegend = idx == 0
                                apply_grok_layout(
                                    fig,
                                    height=250,
                                    showlegend=showlegend,
                                    margin=dict(l=20, r=10, t=35, b=25),
                                )
                                fig.update_layout(title=f"Unit {unit}")
                                with cols[idx % columns]:
                                    render_plotly(
                                        fig,
                                        use_container_width=True,
                                        key=f"resp_grid_{unit}",
                                    )
                else:
                    st.info("No response file found for selected epoch.")
    else:
        st.info("Neuron response artifacts not found.")


def render_loss_landscape(land_dir, land_epochs, global_epoch):
    st.subheader("Loss Landscape")
    if land_dir.exists():
        if land_epochs:
            land_epoch = epoch_selector(
                "Landscape epoch", land_epochs, global_epoch, "land"
            )
            if land_epoch is None:
                st.info("No landscape epochs available.")
            else:
                land_path = land_dir / f"epoch_{land_epoch:03d}.npz"
                if land_path.exists():
                    land = read_npz_fresh(land_path)
                    if land is None:
                        st.info("Loss landscape file not ready yet.")
                        return
                    alphas = land["alphas"]
                    betas = land["betas"]
                    loss = land["loss"]
                    view = st.radio(
                        "Landscape view",
                        ["3D", "Heatmap"],
                        horizontal=True,
                        key="land_view",
                    )
                    if view == "3D":
                        x_grid, y_grid = np.meshgrid(betas, alphas)
                        surface = go.Figure(
                            data=[
                                go.Surface(
                                    x=x_grid,
                                    y=y_grid,
                                    z=loss,
                                    colorscale=GROK_SEQ_SCALE,
                                )
                            ]
                        )
                        surface.update_layout(
                            height=500,
                            margin=GROK_MARGIN,
                            scene=dict(
                                xaxis=dict(
                                    backgroundcolor=GROK_BG,
                                    gridcolor=GROK_GRID,
                                    color=GROK_TEXT,
                                ),
                                yaxis=dict(
                                    backgroundcolor=GROK_BG,
                                    gridcolor=GROK_GRID,
                                    color=GROK_TEXT,
                                ),
                                zaxis=dict(
                                    backgroundcolor=GROK_BG,
                                    gridcolor=GROK_GRID,
                                    color=GROK_TEXT,
                                ),
                            ),
                        )
                        render_plotly(
                            surface,
                            use_container_width=True,
                            key="loss_landscape_3d",
                        )
                    else:
                        heat = px.imshow(
                            loss,
                            x=betas,
                            y=alphas,
                            aspect="auto",
                            origin="lower",
                            color_continuous_scale=GROK_SEQ_SCALE,
                        )
                        apply_grok_layout(heat, height=500)
                        render_plotly(
                            heat, use_container_width=True, key="loss_landscape_heat"
                        )
        else:
            st.info("No loss landscape files yet.")
    else:
        st.info("Loss landscape artifacts not found.")


def render_filmstrip(
    surface_dir,
    surface_epochs,
    proj_dir,
    proj_epochs,
    probes_df,
    class_names,
):
    st.subheader("Filmstrip Mode")
    filmstrip_choice = st.selectbox(
        "Artifact",
        ["Decision surface", "Representation scatter", "Probe snapshot"],
        index=0,
        key="film_artifact",
    )
    default_cols = 2 if CHEAP_MODE else 3
    default_panels = 6 if CHEAP_MODE else 8
    panel_cols = st.slider("Columns", 2, 4, default_cols, key="film_cols")
    panel_count = st.slider("Panels", 4, 12, default_panels, key="film_panels")
    panel_stride = st.slider("Epoch stride", 1, 20, 5, key="film_stride")
    anchor = st.radio(
        "Anchor",
        ["Latest", "Start"],
        horizontal=True,
        key="film_anchor",
    )

    epochs = []
    if filmstrip_choice == "Decision surface":
        if not surface_epochs:
            st.info("No decision surface files available for filmstrip.")
            return
        metric_choice = st.selectbox(
            "Surface metric (filmstrip)",
            ["Confidence (max prob)", "Entropy", "Class probability"],
            index=0,
            key="film_surface_metric",
        )
        class_idx = None
        if metric_choice == "Class probability":
            latest_path = surface_dir / f"epoch_{surface_epochs[-1]:03d}.npz"
            latest_data = read_npz_fresh(latest_path)
            if latest_data is None:
                st.info("Surface file not ready yet.")
                return
            class_options = (
                list(class_names)
                if class_names
                else [str(i) for i in range(latest_data["proba"].shape[2])]
            )
            class_choice = st.selectbox(
                "Class (filmstrip)",
                class_options,
                index=0,
                key="film_surface_class",
            )
            class_idx = class_options.index(class_choice)
        epochs = select_filmstrip_epochs(
            surface_epochs, panel_count, panel_stride, anchor
        )
    elif filmstrip_choice == "Representation scatter":
        if not proj_epochs:
            st.info("No projection files available for filmstrip.")
            return
        default_sample = 400 if CHEAP_MODE else 800
        sample_size = st.slider(
            "Sample size", 200, 2000, default_sample, 100, key="film_sample_size"
        )
        available_layers = ["penultimate"]  # Default/Fallback
        proj_path = proj_dir / f"epoch_{proj_epochs[-1]:03d}.npz"
        if proj_path.exists():
            latest_proj = read_npz_fresh(proj_path)
            if latest_proj:
                available_layers = sorted(
                    {
                        k.split("_")[0]
                        for k in latest_proj
                        if k.endswith(("_pca", "_umap"))
                    }
                )
        if not available_layers:
            st.info("No projection layers available.")
            return
        layer = st.selectbox(
            "Layer (filmstrip)",
            available_layers,
            index=available_layers.index(
                "penultimate"
                if "penultimate" in available_layers
                else available_layers[-1]
            ),
            key="film_proj_layer",
        )
        method = st.radio(
            "Projection (filmstrip)",
            ["umap", "pca"],
            horizontal=True,
            key="film_proj_method",
        )
        epochs = select_filmstrip_epochs(
            proj_epochs, panel_count, panel_stride, anchor
        )
    else:  # Probe snapshot
        if probes_df is None:
            st.info("probes.csv not found for filmstrip.")
            return
        metric = st.selectbox(
            "Probe metric (filmstrip)",
            ["probe_acc", "probe_f1"],
            index=0,
            key="film_probe_metric",
        )
        epochs = select_filmstrip_epochs(
            sorted(probes_df["epoch"].unique().tolist()),
            panel_count,
            panel_stride,
            anchor,
        )

    if not epochs:
        st.info("No valid epochs found for filmstrip.")
        return

    rows = math.ceil(len(epochs) / panel_cols)
    fig = make_subplots(
        rows=rows,
        cols=panel_cols,
        subplot_titles=[f"Epoch {e}" for e in epochs],
        vertical_spacing=0.1 / rows if rows > 1 else 0,
        horizontal_spacing=0.05,
    )

    # Pre-calculate colors/metadata for loops
    # ... logic for each type ...

    for i, epoch in enumerate(epochs):
        row = (i // panel_cols) + 1
        col = (i % panel_cols) + 1
        
        if filmstrip_choice == "Decision surface":
            surface_path = surface_dir / f"epoch_{epoch:03d}.npz"
            if not surface_path.exists():
                continue
            data = read_npz_fresh(surface_path)
            if data is None:
                continue
            grid_x = data["grid_x"][0]
            grid_y = data["grid_y"][:, 0]
            proba = data["proba"].astype(np.float32)
            z = surface_metric(proba, metric_choice, class_idx)
            fig.add_trace(
                go.Heatmap(
                    z=z,
                    x=grid_x,
                    y=grid_y,
                    colorscale=GROK_SEQ_SCALE,
                    showscale=False,
                ),
                row=row,
                col=col,
            )

        elif filmstrip_choice == "Representation scatter":
            proj_path = proj_dir / f"epoch_{epoch:03d}.npz"
            if not proj_path.exists():
                continue
            data = read_npz_fresh(proj_path)
            key = f"{layer}_{method}"
            if data is None or key not in data:
                continue
            coords = data[key]
            labels = map_labels(data["labels"], class_names)
            label_names = (
                class_names if class_names else sorted(set(labels.tolist()))
            )
            class_color_map = build_discrete_color_map(label_names, GROK_COLORWAY)
            
            if coords.shape[0] > sample_size:
                rng = np.random.default_rng(0)
                idxs = rng.choice(
                    coords.shape[0], size=sample_size, replace=False
                )
                coords = coords[idxs]
                labels = labels[idxs]
            
            # Use px to generate traces easily, then copy to subplots
            fig_tmp = px.scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                color=labels,
                color_discrete_map=class_color_map,
                opacity=0.85,
            )
            for trace in fig_tmp.data:
                trace.showlegend = (i == 0) # Only show legend for first plot
                trace.legendgroup = trace.name
                trace.marker.size = 4
                trace.marker.line.width = 0
                fig.add_trace(trace, row=row, col=col)

        elif filmstrip_choice == "Probe snapshot":
            snapshot = probes_df[probes_df["epoch"] == epoch]
            if snapshot.empty:
                continue
            layers_sorted = sorted(probes_df["layer"].unique().tolist())
            layer_color_map = build_discrete_color_map(layers_sorted, GROK_COLORWAY)
            
            fig_tmp = px.bar(
                snapshot,
                x="layer",
                y=metric,
                color="layer",
                color_discrete_map=layer_color_map,
            )
            for trace in fig_tmp.data:
                trace.showlegend = (i == 0)
                trace.legendgroup = trace.name
                fig.add_trace(trace, row=row, col=col)

    # Cleanup axes after adding traces
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    
    # Global layout adjustments
    apply_grok_layout(
        fig,
        height=rows * 240,
        showlegend=(filmstrip_choice != "Decision surface"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    render_plotly(fig, use_container_width=True, key="filmstrip_main_plot")


def render_snapshots(frames_dir):
    st.subheader("Snapshots")
    if frames_dir.exists():
        frames = sorted(frames_dir.glob("epoch_*.png"))
        if frames:
            show_image(str(frames[-1]), caption=frames[-1].name)
        else:
            st.info("No snapshots yet.")
    else:
        st.info("Frames directory not found.")


st.set_page_config(page_title="Grokking Dashboard", layout="wide")
inject_grok_css()

st.title("Grokking Training Dashboard")
render_start = time.perf_counter()

runs_dir = Path(st.sidebar.text_input("Runs directory", "runs", on_change=mark_ui_event))
if not runs_dir.exists():
    st.info("Runs directory not found.")
    runs = []
else:
    runs = list_runs(runs_dir)
    if not runs:
        st.info("No runs found.")

run_names = [run.name for run in runs]
selected = (
    st.sidebar.selectbox(
        "Run",
        run_names,
        index=0,
        on_change=mark_ui_event,
    )
    if run_names
    else None
)
run_dir = runs_dir / selected if selected else None
config = load_config(run_dir)
class_names = config.get("classes")

auto_refresh = st.sidebar.checkbox(
    "Auto refresh",
    value=False,
    key="auto_refresh_enabled",
    on_change=mark_ui_event,
)
refresh = (
    st.sidebar.slider(
        "Auto refresh (sec)",
        1,
        30,
        max(1, AUTO_REFRESH_DEFAULT),
        key="auto_refresh_sec",
        on_change=mark_ui_event,
    )
    if auto_refresh
    else 0
)
page = st.sidebar.radio(
    "View",
    TAB_NAMES,
    index=0,
    on_change=mark_ui_event,
)

missing_dir = Path("__missing__do_not_create__")
base_dir = run_dir if run_dir is not None else missing_dir
metrics_path = base_dir / "metrics.csv"
probes_path = base_dir / "probes.csv"
proj_dir = base_dir / "projections"
land_dir = base_dir / "landscapes"
frames_dir = base_dir / "frames"
surface_dir = base_dir / "surfaces"
interp_dir = base_dir / "interpolations"
response_dir = base_dir / "responses"
rep_dir = base_dir / "diagnostics" / "rep"

metrics_df = read_csv_fresh(metrics_path)
probes_df = read_csv_fresh(probes_path)

metrics_epochs = (
    sorted(metrics_df["epoch"].dropna().astype(int).unique().tolist())
    if metrics_df is not None
    else []
)
proj_epochs = list_epochs(proj_dir)
surface_epochs = list_epochs(surface_dir)
interp_epochs = list_epochs(interp_dir)
response_epochs = list_epochs(response_dir)
land_epochs = list_epochs(land_dir)
rep_epochs = list_epochs(rep_dir)

all_epochs = sorted(
    set(
        metrics_epochs
        + proj_epochs
        + rep_epochs
        + surface_epochs
        + interp_epochs
        + response_epochs
        + land_epochs
    )
)

st.session_state.setdefault("global_play", False)
st.session_state.setdefault("global_speed", 0.4)
play = False

pending_epoch = (
    st.session_state.pop("pending_epoch")
    if "pending_epoch" in st.session_state
    else None
)
if all_epochs:
    if pending_epoch in all_epochs:
        st.session_state.global_epoch = pending_epoch
    elif (
        "global_epoch" not in st.session_state
        or st.session_state.global_epoch not in all_epochs
    ):
        st.session_state.global_epoch = all_epochs[-1]
else:
    st.session_state.global_epoch = 0

st.subheader("Epoch Controller")
ctrl_cols = st.columns([3, 1, 1, 1])
with ctrl_cols[0]:
    if all_epochs:
        st.select_slider(
            "Global epoch",
            options=all_epochs,
            value=st.session_state.global_epoch,
            key="global_epoch",
            on_change=mark_ui_event,
        )
    else:
        st.info("No epoch artifacts yet.")
with ctrl_cols[1]:
    play = st.toggle(
        "Play", key="global_play", disabled=not all_epochs, on_change=mark_ui_event
    )
with ctrl_cols[2]:
    st.slider(
        "Speed (sec/step)",
        0.05,
        2.0,
        value=st.session_state.global_speed,
        step=0.05,
        key="global_speed",
        disabled=not all_epochs,
        on_change=mark_ui_event,
    )
with ctrl_cols[3]:
    if st.button("Jump to latest", disabled=not all_epochs):
        st.session_state.pending_epoch = all_epochs[-1]
        mark_ui_event()
        st.rerun()

if not play:
    st.session_state.pop("pending_epoch", None)

prev_play = st.session_state.get("prev_play", False)
if play and not prev_play:
    st.session_state["rep_lock_axes"] = True
    st.session_state["proj_lock_axes"] = True
st.session_state["prev_play"] = play

scroll_token = None
if play or auto_refresh:
    scroll_token = watch_scroll_events()
    if scroll_token is not None:
        st.session_state["scroll_y"] = scroll_token
        st.session_state["last_ui_event"] = time.perf_counter()

global_epoch = st.session_state.global_epoch
CHEAP_MODE = play

if page == TAB_NAMES[0]:
    render_loss_curves(metrics_df)
elif page == TAB_NAMES[1]:
    render_metrics(metrics_df)
elif page == TAB_NAMES[2]:
    render_rep_grid(rep_dir, rep_epochs, global_epoch, class_names, metrics_df)
elif page == TAB_NAMES[3]:
    render_rep_evolution(proj_dir, proj_epochs, global_epoch, class_names)
elif page == TAB_NAMES[4]:
    render_decision_surface(surface_dir, surface_epochs, global_epoch, class_names)
elif page == TAB_NAMES[5]:
    render_linear_probes(probes_df)
elif page == TAB_NAMES[6]:
    render_interpolation(interp_dir, interp_epochs, global_epoch, class_names)
elif page == TAB_NAMES[7]:
    render_neuron_responses(response_dir, response_epochs, global_epoch, class_names)
elif page == TAB_NAMES[8]:
    render_loss_landscape(land_dir, land_epochs, global_epoch)
elif page == TAB_NAMES[9]:
    render_filmstrip(
        surface_dir,
        surface_epochs,
        proj_dir,
        proj_epochs,
        probes_df,
        class_names,
    )
elif page == TAB_NAMES[10]:
    render_snapshots(frames_dir)

if play or auto_refresh:
    restore_scroll_position()

advance_epoch = False
render_elapsed = time.perf_counter() - render_start
last_ui_event = st.session_state.get("last_ui_event", 0.0)
idle_for = time.perf_counter() - last_ui_event
idle_delay = max(0.0, AUTO_REFRESH_GRACE - idle_for)
if play and all_epochs:
    current_idx = all_epochs.index(st.session_state.global_epoch)
    next_idx = (current_idx + 1) % len(all_epochs)
    st.session_state.pending_epoch = all_epochs[next_idx]
    advance_epoch = True

if advance_epoch:
    target_interval = max(
        st.session_state.global_speed,
        MIN_PLAY_INTERVAL,
        render_elapsed + 0.05,
    )
    sleep_for = max(0.0, target_interval - render_elapsed, idle_delay)
    time.sleep(sleep_for)
    st.rerun()
elif refresh > 0 and not play:
    sleep_for = max(0.0, refresh - render_elapsed, idle_delay)
    time.sleep(sleep_for)
    st.rerun()
