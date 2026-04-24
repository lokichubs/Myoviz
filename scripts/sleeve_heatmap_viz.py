"""
sleeve_heatmap_viz.py
---------------------
Dual-forearm EMG heatmap animation overlaid on anatomical forearm templates.

Pipeline:
  1. Load data  (signal, ring_map, valid_mask, metadata)
  2. Compute sliding-window RMS
  3. Build strip grids  →  left_display_plot (A-E, mirrored) / right_display (F-J)
  4. Load calibration borders  (pre-fitted warp control points)
  5. Precompute RBF inverse-warp maps  (done once, reused every frame)
  6. Animate: apply_warp per frame  →  FuncAnimation  →  HTML or live window

Usage
-----
    # In a notebook:
    from sleeve_heatmap_viz import run
    display(run(data_dir="../data", assets_dir="../assets"))

    # As a script (opens live window):
    python sleeve_heatmap_viz.py
"""

from __future__ import annotations

import matplotlib
# Must be set before pyplot is imported.
# TkAgg opens a live interactive window when run as a script.
# Notebooks override this automatically — no effect there.
matplotlib.use("TkAgg")

from pathlib import Path
import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from IPython.display import HTML
from PIL import Image
from scipy.interpolate import RBFInterpolator


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRIP_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# Channel ranges for each of the 10 strips.
# E and F have only 12 electrodes (missing the first ring position).
STRIP_CHANNELS = [
    np.arange(0,   13),   # A  – 13 ch
    np.arange(13,  26),   # B  – 13 ch
    np.arange(26,  39),   # C  – 13 ch
    np.arange(39,  52),   # D  – 13 ch
    np.arange(52,  64),   # E  – 12 ch
    np.arange(64,  76),   # F  – 12 ch
    np.arange(76,  89),   # G  – 13 ch
    np.arange(89,  102),  # H  – 13 ch
    np.arange(102, 115),  # I  – 13 ch
    np.arange(115, 128),  # J  – 13 ch
]

CORNER_NAMES = ["TL", "TR", "BR", "BL"]

# electrode_map row/col indices corresponding to each corner name
CORNER_MAP_POSITIONS = [(0, 0), (0, -1), (-1, -1), (-1, 0)]

# text alignment per corner: (ha, va)
CORNER_HA_VA = [
    ("left",  "top"),
    ("right", "top"),
    ("right", "bottom"),
    ("left",  "bottom"),
]


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: str | Path) -> dict:
    """
    Load the EMG signal, ring map, valid mask, and metadata from *data_dir*.

    Returns a dict with keys:
        signal       – np.ndarray  [T, 128]  raw EMG samples
        ring_map     – np.ndarray  [26, 5]   channel-to-ring mapping
        valid_mask   – np.ndarray  [26, 5]   bool mask of active channels
        metadata     – dict                   JSON sidecar
    """
    d = Path(data_dir)
    paths = {
        "signal":     d / "synthetic_signal.npy",
        "ring_map":   d / "ring_map.npy",
        "valid_mask": d / "valid_mask.npy",
        "metadata":   d / "metadata.json",
    }
    for k, p in paths.items():
        assert p.exists(), f"Missing file: {p}"

    return {
        "signal":     np.load(paths["signal"]),
        "ring_map":   np.load(paths["ring_map"]),
        "valid_mask": np.load(paths["valid_mask"]),
        "metadata":   json.loads(paths["metadata"].read_text()),
    }


def load_templates(assets_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the inner (palm-side) and outer (dorsal-side) forearm JPEG templates.

    Returns
    -------
    bg_inner, bg_outer  – float32 RGB arrays normalised to [0, 1]
    """
    a = Path(assets_dir)
    bg_inner = np.array(Image.open(a / "forearm_in_template.jpeg").convert("RGB")).astype(np.float32) / 255.0
    bg_outer = np.array(Image.open(a / "forearm_out_template.jpeg").convert("RGB")).astype(np.float32) / 255.0
    return bg_inner, bg_outer


def load_calibration(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the pre-fitted warp border control points saved by the calibration
    notebook cells.

    Returns
    -------
    dst_border_inner, dst_border_outer  – float32 arrays of shape [N, 2]
        N border points in image-pixel coordinates (x, y).
    """
    cal = np.load(Path(data_dir) / "forearm_calibration.npz")
    return cal["inner"].astype(np.float32), cal["outer"].astype(np.float32)


# ---------------------------------------------------------------------------
# 2. RMS computation
# ---------------------------------------------------------------------------

def sliding_rms(x: np.ndarray, win: int = 50, hop: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute sliding-window RMS over a 2-D signal array.

    Parameters
    ----------
    x   : [T, C]  raw signal
    win : window length in samples
    hop : step between windows in samples

    Returns
    -------
    rms_frames : [F, C]  RMS values,  F = number of windows
    starts     : [F]     start sample index of each window
    """
    assert x.ndim == 2
    n, c = x.shape
    starts = np.arange(0, n - win + 1, hop)
    out = np.empty((len(starts), c), dtype=np.float32)
    sq = x.astype(np.float32) ** 2
    for i, s in enumerate(starts):
        out[i] = np.sqrt(np.mean(sq[s:s + win], axis=0))
    return out, starts


# ---------------------------------------------------------------------------
# 3. Strip-grid construction
# ---------------------------------------------------------------------------

def build_strip_grids(
    rms_frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Arrange the 128 RMS channels into two display grids:

      left_display_plot  – strips A-E, shape [F, 5, 13], mirrored H+V
      right_display      – strips F-J, shape [F, 5, 13]

    Also returns the corresponding static electrode-ID maps:

      left_electrode_display_plot  – [5, 13]  (mirrored)
      right_electrode_display      – [5, 13]

    Strip depth is 13 ring positions; strips E and F have no electrode at
    position 0, so that cell is NaN / -1.
    """
    strip_depth = 13
    n_frames = rms_frames.shape[0]

    strip_grid     = np.full((n_frames, strip_depth, 10), np.nan, dtype=np.float32)
    electrode_grid = np.full((strip_depth, 10), -1, dtype=np.int32)

    for s_idx, ch_idx in enumerate(STRIP_CHANNELS):
        if ch_idx.size == 13:
            strip_grid[:, :, s_idx]   = rms_frames[:, ch_idx]
            electrode_grid[:, s_idx]  = ch_idx + 1          # 1-based IDs
        elif ch_idx.size == 12:
            strip_grid[:, 1:, s_idx]  = rms_frames[:, ch_idx]
            electrode_grid[1:, s_idx] = ch_idx + 1
        else:
            raise RuntimeError(f"Unexpected strip length for {STRIP_NAMES[s_idx]}: {ch_idx.size}")

    # Split into A-E (left) and F-J (right) halves
    left_half  = strip_grid[:, :, 0:5]
    right_half = strip_grid[:, :, 5:10]
    left_elec  = electrode_grid[:, 0:5]
    right_elec = electrode_grid[:, 5:10]

    # Transpose to [F, 5-strips, 13-depth] for imshow
    left_display  = np.transpose(left_half,  (0, 2, 1))
    right_display = np.transpose(right_half, (0, 2, 1))
    left_elec_t   = left_elec.T
    right_elec_t  = right_elec.T

    # Mirror A-E both horizontally and vertically so it faces the right way
    left_display_plot      = np.flip(left_display, axis=(1, 2))
    left_elec_display_plot = np.flip(left_elec_t,  axis=(0, 1))

    return left_display_plot, right_display, left_elec_display_plot, right_elec_t


def compute_color_limits(
    left_display_plot: np.ndarray,
    right_display: np.ndarray,
    percentile: float = 99.0,
) -> tuple[float, float]:
    """
    Compute shared vmin / vmax across both halves using a robust percentile.

    Returns (vmin, vmax) where vmin is always 0.
    """
    all_vals = np.concatenate([
        left_display_plot[np.isfinite(left_display_plot)],
        right_display[np.isfinite(right_display)],
    ])
    return 0.0, float(np.percentile(all_vals, percentile))


# ---------------------------------------------------------------------------
# 4. Warp border source points  (must match calibration notebook exactly)
# ---------------------------------------------------------------------------

def make_source_border_points(nx: int = 7, ny: int = 2) -> tuple[np.ndarray, list[int]]:
    """
    Generate perimeter control points on the heatmap source rectangle
    (x: 0..12, y: 0..4) in clockwise order: top → right → bottom → left.

    Parameters
    ----------
    nx, ny : number of points along x-edge and y-edge (corners shared).
             Must match the values used during calibration.

    Returns
    -------
    src_border  : [N, 2]  source-space perimeter points
    corner_idx  : [4]     indices of TL, TR, BR, BL corners inside src_border
    """
    xs = np.linspace(0.0, 12.0, nx, dtype=np.float32)
    ys = np.linspace(0.0,  4.0, ny, dtype=np.float32)

    top    = np.column_stack([xs,                                            np.full_like(xs, ys[0])])
    right  = np.column_stack([np.full((ny - 1,), xs[-1], dtype=np.float32), ys[1:]])
    bottom = np.column_stack([xs[-2::-1],                                    np.full((nx - 1,), ys[-1], dtype=np.float32)])
    left   = np.column_stack([np.full((ny - 2,), xs[0],  dtype=np.float32), ys[-2:0:-1]])

    src_border = np.vstack([top, right, bottom, left]).astype(np.float32)
    corner_idx = [0, nx - 1, nx + (ny - 2), 2 * nx + ny - 3]   # TL TR BR BL
    return src_border, corner_idx


# ---------------------------------------------------------------------------
# 5. Warp precomputation  (the expensive step — done once)
# ---------------------------------------------------------------------------

def precompute_warp(
    dst_border: np.ndarray,
    src_border: np.ndarray,
    bg: np.ndarray,
) -> dict:
    """
    Fit a thin-plate-spline (TPS) inverse warp from image pixels → heatmap
    source coords, then rasterise the map over the bounding box of the warp
    region.

    The resulting map_x / map_y arrays are passed directly to cv2.remap every
    frame — no further interpolation setup needed.

    Parameters
    ----------
    dst_border : [N, 2]  border control points in image pixel space
    src_border : [N, 2]  corresponding source heatmap coordinates
    bg         : [H, W, 3] background image (used only for size clamping)

    Returns
    -------
    dict with keys:
        map_x, map_y – float32 remap arrays
        mask_f       – float32 [h, w, 1] polygon mask inside the warp region
        bbox         – (x_min, x_max, y_min, y_max) ROI in image coords
    """
    H, W = bg.shape[:2]

    rbf_x = RBFInterpolator(dst_border, src_border[:, 0], kernel="thin_plate_spline")
    rbf_y = RBFInterpolator(dst_border, src_border[:, 1], kernel="thin_plate_spline")

    # Bounding box with a small margin
    x_min = max(0,     int(np.floor(np.min(dst_border[:, 0])) - 8))
    x_max = min(W - 1, int(np.ceil(np.max(dst_border[:, 0])) + 8))
    y_min = max(0,     int(np.floor(np.min(dst_border[:, 1])) - 8))
    y_max = min(H - 1, int(np.ceil(np.max(dst_border[:, 1])) + 8))

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max + 1, dtype=np.float32),
        np.arange(y_min, y_max + 1, dtype=np.float32),
        indexing="xy",
    )
    q = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)

    map_x = rbf_x(q).reshape(xx.shape).astype(np.float32)
    map_y = rbf_y(q).reshape(yy.shape).astype(np.float32)

    # Polygon mask so the overlay is clipped to the warp boundary
    poly     = dst_border.astype(np.int32)
    poly_roi = poly - np.array([x_min, y_min], dtype=np.int32)
    mask     = np.zeros((yy.shape[0], xx.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_roi], 255)
    mask_f   = (mask.astype(np.float32) / 255.0)[:, :, None]

    return {
        "map_x":  map_x,
        "map_y":  map_y,
        "mask_f": mask_f,
        "bbox":   (x_min, x_max, y_min, y_max),
    }


# ---------------------------------------------------------------------------
# 6. Per-frame warp application
# ---------------------------------------------------------------------------

def apply_warp(
    frame_data: np.ndarray,
    warp: dict,
    bg: np.ndarray,
    cmap,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """
    Colour-map one frame of heatmap data and composite it onto the background
    image using the pre-computed warp maps.

    Parameters
    ----------
    frame_data : [rows, cols]  single-frame RMS grid (may contain NaN)
    warp       : dict returned by precompute_warp
    bg         : [H, W, 3]  background forearm image (not mutated)
    cmap       : matplotlib colormap
    vmin, vmax : colour scale limits

    Returns
    -------
    out : [H, W, 3] float32 composited image
    """
    norm = np.clip((frame_data - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    rgba = cmap(norm).astype(np.float32)
    rgba[~np.isfinite(frame_data), 3] = 0.0   # transparent for NaN cells

    warped = cv2.remap(
        rgba,
        warp["map_x"],
        warp["map_y"],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    x_min, x_max, y_min, y_max = warp["bbox"]
    out    = bg.copy()
    roi_bg = out[y_min:y_max + 1, x_min:x_max + 1]
    alpha  = warped[:, :, 3:4] * warp["mask_f"]
    out[y_min:y_max + 1, x_min:x_max + 1] = roi_bg * (1.0 - alpha) + warped[:, :, :3] * alpha
    return out


# ---------------------------------------------------------------------------
# 7. Corner-label annotation  (called once at figure setup)
# ---------------------------------------------------------------------------

def annotate_warp_corners(
    ax,
    dst_border: np.ndarray,
    corner_idx: list[int],
    electrode_map: np.ndarray,
    names: list[str] = CORNER_NAMES,
) -> None:
    """
    Draw four static electrode-ID labels at the corner border control points
    in image-pixel (data) coordinates.

    Must be called ONCE after ax.imshow() and BEFORE FuncAnimation starts.
    Labels are static artists — they are never recreated during animation.

    Parameters
    ----------
    ax            : matplotlib Axes
    dst_border    : [N, 2]  border control points in pixel space
    corner_idx    : [4]     indices of TL/TR/BR/BL inside dst_border
    electrode_map : [rows, cols]  int array of 1-based electrode IDs (-1 = absent)
    names         : corner label prefixes, default ["TL","TR","BR","BL"]
    """
    for k, ci in enumerate(corner_idx):
        px, py = dst_border[ci]
        r, c   = CORNER_MAP_POSITIONS[k]
        eid    = int(electrode_map[r, c])
        txt    = f"{names[k]}: E{eid}" if eid > 0 else f"{names[k]}: NA"
        ha, va = CORNER_HA_VA[k]
        ax.text(
            px, py, txt,
            ha=ha, va=va,
            fontsize=7, color="white",
            transform=ax.transData,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7, edgecolor="none"),
        )


# ---------------------------------------------------------------------------
# 8. Main animation builder
# ---------------------------------------------------------------------------

def build_animation(
    left_display_plot: np.ndarray,
    right_display: np.ndarray,
    left_electrode_display_plot: np.ndarray,
    right_electrode_display: np.ndarray,
    rms_starts: np.ndarray,
    warp_inner: dict,
    warp_outer: dict,
    dst_border_inner: np.ndarray,
    dst_border_outer: np.ndarray,
    corner_idx: list[int],
    bg_inner: np.ndarray,
    bg_outer: np.ndarray,
    cmap,
    vmin: float,
    vmax: float,
    interval: int = 60,
    close_fig: bool = True,
) -> FuncAnimation:
    """
    Build and return a FuncAnimation that renders both forearm views
    stacked vertically.

    Static elements (corner labels, titles) are created once.
    The update() closure only calls set_data() — no artists are recreated.

    Parameters
    ----------
    interval  : ms between frames (lower = faster playback)
    close_fig : True  → plt.close() after building  (notebook mode, avoids double render)
                False → leave figure open            (script mode, needed for plt.show())

    Returns
    -------
    ani : FuncAnimation
    """
    fig, axs = plt.subplots(2, 1, figsize=(8, 5))
    axs[0].set_title("Inner (A–E)")
    axs[1].set_title("Outer (F–J)")

    # Initial frame — background only, animation fills in from frame 0
    im_inner = axs[0].imshow(bg_inner.copy())
    im_outer = axs[1].imshow(bg_outer.copy())

    # Static corner labels — drawn once, never touched by update()
    annotate_warp_corners(axs[0], dst_border_inner, corner_idx, left_electrode_display_plot)
    annotate_warp_corners(axs[1], dst_border_outer, corner_idx, right_electrode_display)

    for ax in axs:
        ax.set_axis_off()

    title = fig.suptitle("Frame 0")
    fig.tight_layout()

    def update(i: int):
        """Update image data for frame i — no new artists created."""
        out_inner = apply_warp(left_display_plot[i], warp_inner, bg_inner, cmap, vmin, vmax)
        out_outer = apply_warp(right_display[i],      warp_outer, bg_outer, cmap, vmin, vmax)
        im_inner.set_data(out_inner)
        im_outer.set_data(out_outer)
        title.set_text(f"Frame {i}  (sample {int(rms_starts[i])})")
        return im_inner, im_outer, title

    ani = FuncAnimation(
        fig, update,
        frames=left_display_plot.shape[0],
        interval=interval,
        blit=False,
    )

    if close_fig:
        plt.close(fig)   # prevents double-render in notebooks

    return ani


# ---------------------------------------------------------------------------
# 9. Top-level pipeline
# ---------------------------------------------------------------------------

def run(
    data_dir:    str | Path = "../data",
    assets_dir:  str | Path = "../assets",
    rms_win:     int = 50,
    rms_hop:     int = 10,
    edge_nx:     int = 7,
    edge_ny:     int = 2,
    interval:    int = 60,
    return_html: bool = True,
) -> HTML | FuncAnimation:
    """
    Full pipeline from raw data files to animation.

    Parameters
    ----------
    data_dir    : directory containing .npy / .json / calibration files
    assets_dir  : directory containing forearm JPEG templates
    rms_win     : sliding-RMS window length (samples)
    rms_hop     : sliding-RMS hop size (samples)
    edge_nx     : border control points along the long axis  (must match calibration)
    edge_ny     : border control points along the short axis (must match calibration)
    interval    : animation frame interval in ms
    return_html : True  → returns IPython HTML object  (notebook mode)
                  False → returns raw FuncAnimation     (script / interactive mode)

    Returns
    -------
    HTML object (notebook) or FuncAnimation (script)
    """
    # -- Load ----------------------------------------------------------------
    data                               = load_data(data_dir)
    bg_inner, bg_outer                 = load_templates(assets_dir)
    dst_border_inner, dst_border_outer = load_calibration(data_dir)
    signal                             = data["signal"]

    # -- RMS -----------------------------------------------------------------
    rms_frames, rms_starts = sliding_rms(signal, win=rms_win, hop=rms_hop)

    # -- Strip grids ---------------------------------------------------------
    (
        left_display_plot,
        right_display,
        left_electrode_display_plot,
        right_electrode_display,
    ) = build_strip_grids(rms_frames)

    vmin, vmax = compute_color_limits(left_display_plot, right_display)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad((0, 0, 0, 0))

    # -- Border source points (must match what was used during calibration) --
    src_border, corner_idx = make_source_border_points(nx=edge_nx, ny=edge_ny)

    # -- Precompute warps (slow step, runs once) -----------------------------
    warp_inner = precompute_warp(dst_border_inner, src_border, bg_inner)
    warp_outer = precompute_warp(dst_border_outer, src_border, bg_outer)

    # -- Build animation -----------------------------------------------------
    ani = build_animation(
        left_display_plot           = left_display_plot,
        right_display               = right_display,
        left_electrode_display_plot = left_electrode_display_plot,
        right_electrode_display     = right_electrode_display,
        rms_starts                  = rms_starts,
        warp_inner                  = warp_inner,
        warp_outer                  = warp_outer,
        dst_border_inner            = dst_border_inner,
        dst_border_outer            = dst_border_outer,
        corner_idx                  = corner_idx,
        bg_inner                    = bg_inner,
        bg_outer                    = bg_outer,
        cmap                        = cmap,
        vmin                        = vmin,
        vmax                        = vmax,
        interval                    = interval,
        close_fig                   = return_html,   # keep fig open for interactive mode
    )

    if return_html:
        return HTML(ani.to_jshtml())
    return ani


# ---------------------------------------------------------------------------
# Script entry point  —  opens a live interactive window
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data and precomputing warps...")
    ani = run(return_html=False)

    print("Rendering — close the window to exit.")
    plt.show()   # blocks until window is closed