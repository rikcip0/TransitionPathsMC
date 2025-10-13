from __future__ import annotations
def _is_mu_sigma_line(ln):
    try:
        lab = (ln.get_label() or "").lower()
    except Exception:
        lab = ""
    # match common notations
    keys = ["mu", "μ", "sigma", "σ", "mu+sigma", "mu-sigma", "mu±sigma", "mu +/- sigma"]
    return any(k in lab for k in keys)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrapper PAPER â€” STRICT + GUIDES + NORMALIZATION + pass-through CDF refs
Aggiorna multipleCurvesAndHist per produrre pannelli con geometria e tipografia
coerenti con le altre figure del paper.
"""


from typing import List, Dict, Any, Optional, Sequence, Tuple, Union
import os
import sys
import re
import json
import inspect
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, MaxNLocator, ScalarFormatter

xTicks =None
xTicksLabels=None

def ref_from_tauAndT(tau,T):
    frac= tau/T
    return [(1.-frac)/2., (1.+frac)/2.]
    
# ======================
# LISTA PLOT (EDITA QUI)
# ======================
PLOTS: List[Dict[str, Any]] = [
    {
        "name": "Fig01_M_vs_t/ER_HT_m",
        "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\ER\p2C3\N200\structure388845\fPosJ1.00\graph8865\DataForPathsMC\PathsMCs\80_0.525_0_inf_120_inf_run7201",
        "outfile": "_figs/ChiVsT_allTrajs/ER_HT",
        "only_end": False,
        "y_key": "M",
        "N": 200,
        "curvesIndices": [8, 98, 55, 108,59],
        "x_label": "t",
        "y_label": "m",
        "show_yhist": True,
        "y_cdf": True,
        "yGuides": [[0.6, "m*", "red"]],
        "cdf_ref_probs_y": ref_from_tauAndT(30.,80),
        "y_log_density": False,
        "legend": False,
        "reserve_legend_space": False,
        "panel_width_in": 2.45,
        "outfile": "_figs/Fig01_m_vs_t/ER_HT_m",
                "fig_scale":1.5,
                        "xticks":[0,40,80],
    },
    {
        "name": "Fig01_M_vs_t/RRG_refnew_m",
        "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\RRG\p2C3\N160\structure140322\fPosJ1.00\graph340\DataForPathsMC\PathsMCs\80_1_0_inf_96_inf_run9157",
        "y_key": "M",
        "N": 160,
        "curvesIndices": [30, 60, 90, 120,150],
        "x_label": "t",
        "y_label": "m",
        "show_yhist": True,
        "y_cdf": True,
        "yGuides": [[0.6, "m*", "red"]],
        "cdf_ref_probs_y": ref_from_tauAndT(43.,80),
        "y_log_density": True,
        "legend": False,
        "reserve_legend_space": False,
        "panel_width_in": 2.45,
        "outfile": "_figs/Fig01_m_vs_t/RRG_refnew_m",
                "fig_scale":1.5,
                        "xticks":[0,40,80],
    },
    {
        "name": "Fig01_M_vs_t/ER_refnew_m",
        "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\ER\p2C3\N140\structure655712\fPosJ1.00\graph5735\DataForPathsMC\PathsMCs\70_1_0_inf_84_inf_run6751",
        "y_key": "M",
        "N": 140,
        "curvesIndices": [10, 14, 80, 150,123],
        "x_label": "t",
        "y_label": "m",
        "show_yhist": True,
        "y_cdf": True,
        "yGuides": [[0.6, "m*", "red"]],
        "cdf_ref_probs_y": ref_from_tauAndT(43.,70),
        "y_log_density": False,
        "legend": False,
        "reserve_legend_space": False,
        "panel_width_in": 2.45,
        "outfile": "_figs/Fig01_m_vs_t/ER_refnew_m",
        "fig_scale":1.5,
                "xticks":[0,35,70],
    },
    {
        "name": "Fig01_M_vs_t/ZKC40_0p55",
        "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.55_0_inf_20_inf_run2634",
        "y_key": "M",
        "N": 34,
        "curvesIndices": [9, 14, 36, 40, 53],
        "x_label": "t",
        "y_label": "m",
        "show_yhist": True,
        "y_cdf": True,
        "yGuides": [[20./34., "m*", "red"]],
        "cdf_ref_probs_y": [1/3, 2/3],
        "y_log_density": False,
        "legend": False,
        "reserve_legend_space": False,
        "panel_width_in": 2.45,
        "outfile": "_figs/Fig01_m_vs_t/ZKC40_0p55",
    },
    {
        "name": "Fig01_M_vs_t/ZKC40_0p35",
        "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.35_0_inf_20_inf_run2642",
        "y_key": "M",
        "N": 34,
        "curvesIndices": [10, 76, 30, 24, 53],
        "x_label": "t",
        "y_label": "m",
        "show_yhist": True,
        "y_cdf": True,
        "yGuides": [[20./34., "m*", "red"]],
        "cdf_ref_probs_y": [1/3, 2/3],
        "y_log_density": False,
        "legend": False,
        "reserve_legend_space": False,
        "panel_width_in": 2.45,
        "outfile": "_figs/Fig01_m_vs_t/ZKC40_0p35",
    },

    {
        "name": "Fig01_M_vs_t/ZKC40_0p9",
        "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.9_0_inf_20_inf_run2620",
        "y_key": "M",
        "N": 34,
        "curvesIndices": [0, 3, 10, 25, 130],
        "x_label": "t",
        "y_label": "m",
        "show_yhist": True,
        "y_cdf": True,
        "yGuides": [[20./34., "m*", "red"]],
        "cdf_ref_probs_y": [1/3, 2/3],
        "y_log_density": False,
        "legend": False,
        "reserve_legend_space": False,
        "panel_width_in": 2.1170826069,   # = 153.00156 pt / 72.27
        "panel_height_in": 1.12,

        # margini (come energy_vs_m)
        "left_in":   0.46,
        "right_in":  0.10,
        "bottom_in": 0.34,
        "top_in":    0.12,

        "outfile": "_figs/Fig01_m_vs_t/ZKC40_0p9",
    },
    {
        "name": "Fig01_M_vs_t/ZKC40_0p25",
        "run_path": r"C:\Users\ricca\Desktop\College\Codici\TransitionPathsMC\Data\Graphs\realGraphs\ZKC\DataForPathsMC\PathsMCs\40_0.25_0_inf_20_inf_run6871",
        "y_key": "M",
        "N": 34,
        "curvesIndices": [15, 21, 46, 91, 123],
        "x_label": "t",
        "y_label": "m",
        "show_yhist": True,
        "y_cdf": True,
        "yGuides": [[20./34., "m*", "red"]],
        "cdf_ref_probs_y": [1/3, 2/3],
        "y_log_density": False,
        "legend": False,
        "reserve_legend_space": False,
        "panel_width_in": 2.45,
        "outfile": "_figs/Fig01_m_vs_t/ZKC40_0p25",
        "xticks":[0,35,70],
    },

]

# =====================
# COSTANTI GEOMETRIA
# =====================
DEFAULT_PANEL_WIDTH_IN = 2.45
DEFAULT_PANEL_HEIGHT_IN = 1.580
LEFT_MARGIN_IN = 0.47
RIGHT_MARGIN_IN = 0.10
BOTTOM_MARGIN_IN = 0.45
TOP_MARGIN_IN = 0.14
Y_HIST_GAP_IN = 0.10
Y_HIST_WIDTH_IN = 0.32
TICK_LENGTH_PT = 3.2
TICK_WIDTH_PT = 0.85
TICK_PAD_PT = 2.0
DEFAULT_FONT_SCALE_HIST = 0.60
DENSITY_TICK_SCALE = 0.48
GUIDE_LINEWIDTH = 0.75
GUIDE_DASH_ON_PT  = 6.0   # guideline dash 'on' length in pt (long to avoid dotted look)
GUIDE_DASH_OFF_PT = 3.0   # guideline dash 'off' length in pt
MU_SIGMA_LINEWIDTH = 0.55
CURVE_LINEWIDTH = 1.80
GUIDE_LINEWIDTH = 0.70 * CURVE_LINEWIDTH  # proportional for consistency
CDF_LINEWIDTH = 2.20
HIST_BASE_LINEWIDTH = 0.60
DASH_PATTERN = (0.8, 0.8)


class FixedDecimalScalarFormatter(ScalarFormatter):
    """Scalar formatter with fixed number of decimals while keeping mathtext exponent."""

    def __init__(self, decimals: int = 2, **kwargs: Any) -> None:
        super().__init__(useMathText=True, **kwargs)
        self.decimals = decimals

    def _set_format(self) -> None:  # noqa: D401 - internal API override
        super()._set_format()
        self.format = f"%0.{self.decimals}f"


# ===============
# CONFIG AMBIENTE
# ===============
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.append('../')

from MyBasePlots.FigCore import utils_style as ustyle
from MyBasePlots.FigCore import utils_plot as uplot
from MyBasePlots.multipleCurvesAndHistcopy import multipleCurvesAndHist

ArrayLike = Union[np.ndarray, Sequence[float], List[float]]

_CANDIDATE_BASES = [
    Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent,
    REPO_ROOT, REPO_ROOT / "Data", REPO_ROOT / "Data" / "MultiPathsMC",
    REPO_ROOT / "Data" / "Graphs",
]


def _windows_to_wsl(p: str) -> str:
    m = re.match(r"^([a-zA-Z]):[\\/](.*)$", p)
    if m:
        drive = m.group(1).lower()
        rest = re.sub(r"[\\/]+", "/", m.group(2))
        return f"/mnt/{drive}/{rest}"
    return p




def _maybe_graphs_prefix(s: str) -> Optional[Path]:
    s_clean = re.sub(r'[\\]+', '/', s).lstrip('/')
    if s_clean.lower().startswith('realgraphs/'):
        return (REPO_ROOT / 'Data' / 'Graphs' / s_clean).resolve()
    return None




def _normalize_run_path(raw: str) -> Path:
    s = os.path.expanduser(str(raw)).strip()
    if not s:
        raise ValueError("run_path vuoto")
    cand = _maybe_graphs_prefix(s)
    if cand and cand.exists():
        return cand
    s = _windows_to_wsl(s)
    cand = Path(s)
    if cand.is_absolute() and cand.exists():
        return cand
    s2 = s.lstrip('/')
    for base in _CANDIDATE_BASES:
        cand = (base / s2).resolve()
        if cand.exists():
            return cand
    return Path(s if s.startswith('/') else s2)


def get_file_with_prefix(parent_dir: str, prefix: str) -> Optional[str]:
    if not os.path.isdir(parent_dir):
        return None
    for fname in os.listdir(parent_dir):
        fpath = os.path.join(parent_dir, fname)
        if os.path.isfile(fpath) and fname.startswith(prefix):
            return fpath
    return None


def arraysFromBlockFile(filename: str) -> np.ndarray:
    blocks: List[List[List[float]]] = []
    current: List[List[float]] = []
    with open(filename, 'r', encoding='utf-8') as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped:
                if current:
                    blocks.append(current)
                    current = []
                continue
            if stripped.startswith('#'):
                continue
            cols = [float(x) for x in stripped.split()]
            current.append(cols)
    if current:
        blocks.append(current)
    arr = np.asarray([np.array(block) for block in blocks], dtype=float)
    arr = np.transpose(arr, (2, 0, 1))
    return arr




_YKEY_TO_INDEX = {"qin": 1, "qout": 2, "m": 3, "energy": 4}


def _slug(s: str) -> str:
    return ''.join(c if c.isalnum() or c in '-_.' else '_' for c in s).strip('_')


def _assert_figsize_supported() -> None:
    sig = inspect.signature(multipleCurvesAndHist)
    if 'figsize' not in sig.parameters:
        raise RuntimeError("La tua multipleCurvesAndHist NON accetta 'figsize'. Aggiorna il file per supportarlo.")


def _ensure_math_label(label: str | None) -> str | None:
    if not label:
        return label
    stripped = label.strip()
    if stripped.startswith('$') and stripped.endswith('$'):
        return label
    if len(stripped) == 1 and stripped.isalpha():
        return f"${stripped}$"
    return label


def _find_hist_axes(fig: plt.Figure, ax_main: plt.Axes) -> Tuple[Optional[plt.Axes], Optional[plt.Axes]]:
    ax_xhist: Optional[plt.Axes] = None
    ax_yhist: Optional[plt.Axes] = None
    main_bbox = ax_main.get_position().bounds  # (x0, y0, width, height)
    for ax in fig.axes:
        if ax is ax_main:
            continue
        bbox = ax.get_position().bounds
        same_height = abs(bbox[1] - main_bbox[1]) < 1e-6 and abs(bbox[3] - main_bbox[3]) < 1e-6
        same_width = abs(bbox[0] - main_bbox[0]) < 1e-6 and abs(bbox[2] - main_bbox[2]) < 1e-6
        if same_height and bbox[0] > main_bbox[0]:
            ax_yhist = ax
        elif same_width and bbox[1] > main_bbox[1]:
            ax_xhist = ax
    return ax_xhist, ax_yhist


def _apply_panel_layout(fig: plt.Figure, ax_main: plt.Axes, ax_yhist: Optional[plt.Axes], *,
                        show_yhist: bool,
                        panel_width_in: float,
                        panel_height_in: float) -> None:
    target_w = float(panel_width_in or DEFAULT_PANEL_WIDTH_IN)
    target_h = float(panel_height_in or DEFAULT_PANEL_HEIGHT_IN)
    fig.set_size_inches(target_w, target_h, forward=True)
    fig.canvas.draw()

    yhist_gap = Y_HIST_GAP_IN if show_yhist and ax_yhist is not None else 0.0
    yhist_width = Y_HIST_WIDTH_IN if show_yhist and ax_yhist is not None else 0.0

    data_w = target_w - LEFT_MARGIN_IN - RIGHT_MARGIN_IN - yhist_gap - yhist_width
    data_h = target_h - BOTTOM_MARGIN_IN - TOP_MARGIN_IN
    if data_w <= 0 or data_h <= 0:
        raise RuntimeError('Panel layout constants leave no room for data area; adjust margins.')

    left = LEFT_MARGIN_IN / target_w
    bottom = BOTTOM_MARGIN_IN / target_h
    width = data_w / target_w
    height = data_h / target_h
    ax_main.set_position([left, bottom, width, height])

    if show_yhist and ax_yhist is not None:
        y_left = (LEFT_MARGIN_IN + data_w + yhist_gap) / target_w
        y_width = yhist_width / target_w
        ax_yhist.set_position([y_left, bottom, y_width, height])

    fig.canvas.draw_idle()


def _style_main_axis(ax: plt.Axes) -> None:
    ax.set_xlabel(_ensure_math_label(ax.get_xlabel()))
    ax.set_ylabel(_ensure_math_label(ax.get_ylabel()))

    ax.tick_params(axis='both', which='both', direction='out',
                    length=TICK_LENGTH_PT, width=TICK_WIDTH_PT, pad=TICK_PAD_PT)

    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))

    if len(ax.get_xticks()) < 3:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    if len(ax.get_yticks()) < 3:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))

    formatter = FixedDecimalScalarFormatter(decimals=2)
    formatter.set_powerlimits((-2, 2))
    formatter.set_scientific(True)
    formatter.set_useOffset(True)
    ax.yaxis.set_major_formatter(formatter)
    offset = ax.yaxis.get_offset_text()
    offset.set_fontsize(offset.get_fontsize() * 0.65)

    # Fixed ticks required by paper layout
    ax.set_xticks(xTicks)
    ax.set_xticklabels(xTicksLabels)
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.set_yticklabels(['-1', '0', '1'])

    xmin, xmax = ax.get_xlim()
    if xmin > 0.0 or xmax < 40.0:
        ax.set_xlim(min(0.0, xmin), max(40.0, xmax))
    ymin, ymax = ax.get_ylim()
    if ymin > -1.0 or ymax < 1.0:
        ax.set_ylim(min(-1.0, ymin), max(1.0, ymax))

    if ax.get_xscale() == 'log':
        ax.minorticks_off()
    if ax.get_yscale() == 'log':
        ax.minorticks_off()

    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def _style_hist_axis(ax: plt.Axes, orientation: str) -> None:
    ax.tick_params(axis='both', which='both', direction='out',
                    length=TICK_LENGTH_PT * 0.7,
                    width=TICK_WIDTH_PT * 0.9,
                    pad=TICK_PAD_PT * 0.8)

    locator = MaxNLocator(nbins=4, min_n_ticks=3)
    if orientation == 'x':
        ax.xaxis.set_major_locator(locator)
    else:
        ax.yaxis.set_major_locator(locator)

    if ax.get_xscale() == 'log' or ax.get_yscale() == 'log':
        ax.minorticks_off()

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    if orientation == 'y':
        for label in ax.get_yticklabels():
            label.set_fontsize(label.get_fontsize() * DENSITY_TICK_SCALE)
        for label in ax.get_xticklabels():
            label.set_fontsize(label.get_fontsize() * DEFAULT_FONT_SCALE_HIST)
    else:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(label.get_fontsize() * DEFAULT_FONT_SCALE_HIST)


def _normalize_series(Y2D: np.ndarray, key: str, spec: Dict[str, Any]) -> np.ndarray:
    key_l = key.lower()
    if key_l in ("m", "energy"):
        if "N" not in spec or spec["N"] in (None, 0):
            raise ValueError(f"Per y_key='{key}' devi specificare N (>0) per la normalizzazione.")
        N = float(spec["N"])
        return np.asarray(Y2D, dtype=float) / N
    return Y2D


def _adjust_xticks(ax: plt.Axes) -> None:
    ax.set_xticks([0.0, 20.0, 40.0])
    xmin, xmax = ax.get_xlim()
    if xmin > 0.0 or xmax < 40.0:
        ax.set_xlim(min(0.0, xmin), max(40.0, xmax))





def _tune_linewidths(ax_main: plt.Axes, ax_yhist: Optional[plt.Axes]) -> None:
    def _is_mu_sigma(label: str) -> bool:
        return any(token in label for token in ('mu', 'σ', 'sigma', 'μ'))

    def _apply_dash_style(line: plt.Line2D) -> None:
        try:
            line.set_dashes(DASH_PATTERN)
        except Exception:
            pass

    for ln in getattr(ax_main, 'lines', []):
        try:
            label = (ln.get_label() or '').lower()
            if ln.get_linestyle() == '--' or _is_mu_sigma_line(ln):
                if _is_mu_sigma(label):
                    ln.set_linewidth(MU_SIGMA_LINEWIDTH)
                    _apply_dash_style(ln)  # keeps current μ/σ dash pattern
                else:
                    # GUIDELINE in MAIN panel: special dash (long) + capstyle butt
                    ln.set_linewidth(GUIDE_LINEWIDTH)
                    try:
                        ln.set_dashes((GUIDE_DASH_ON_PT * GUIDE_LINEWIDTH,
                                       GUIDE_DASH_OFF_PT * GUIDE_LINEWIDTH))
                        ln.set_dash_capstyle('butt')
                    except Exception:
                        pass
            else:
                lw = CURVE_LINEWIDTH
                if 'cdf' in label:
                    lw = CDF_LINEWIDTH
                ln.set_linewidth(lw)
        except Exception:
            pass

    if ax_yhist is not None:
        for ln in getattr(ax_yhist, 'lines', []):
            try:
                if ln.get_linestyle() == '--' or _is_mu_sigma_line(ln):
                    ln.set_linewidth(GUIDE_LINEWIDTH)
                    _apply_dash_style(ln)
                else:
                    ln.set_linewidth(HIST_BASE_LINEWIDTH)
            except Exception:
                pass
        for coll in getattr(ax_yhist, 'collections', []):
            try:
                coll.set_linewidth(0.40)
            except Exception:
                try:
                    coll.set_linewidths([0.40])
                except Exception:
                    pass



def run_from_specs(specs: List[Dict[str, Any]]) -> None:
    _assert_figsize_supported()

    with ustyle.auto_style(
        mode="latex",
        base=str(Path(ustyle.__file__).resolve().parent / "styles" / "paper_base.mplstyle"),
        overlay=str(Path(ustyle.__file__).resolve().parent / "styles" / "overlay_latex.mplstyle"),
    ):
        for spec in specs:
            name = spec.get("name", "Figure")
            original_run_path = spec["run_path"]

            # --- normalizzazione path run (immutata) ---
            candidates: List[Path] = []
            if isinstance(original_run_path, str):
                candidates.append(Path(original_run_path))
                candidates.append(Path(_windows_to_wsl(original_run_path)))

            run_path: Optional[Path] = None
            for cand in candidates:
                try:
                    if cand.exists():
                        run_path = cand.resolve()
                        break
                except Exception:
                    pass

            if run_path is None:
                run_path = _normalize_run_path(original_run_path)
                run_path_str = str(run_path)
                if ':' in run_path_str:
                    candidate = Path(_windows_to_wsl(run_path_str))
                    if candidate.exists():
                        run_path = candidate.resolve()

            if run_path is None or not run_path.exists():
                bases = '\n  - ' + '\n  - '.join(str(b) for b in _CANDIDATE_BASES)
                norm_path = _windows_to_wsl(str(original_run_path)) if isinstance(original_run_path, str) else str(run_path)
                raise FileNotFoundError(
                    f"Cartella run non trovata:\n  {original_run_path}\n"
                    f"Normalizzato a:\n  {norm_path}\n"
                    f"Cercati anche i prefissi:\n{bases}"
                )

            # --- dati (immutati) ---
            y_key = str(spec.get("y_key", "M")).lower()
            if y_key not in _YKEY_TO_INDEX:
                raise ValueError(f"y_key non valido: {y_key} (ammessi: {list(_YKEY_TO_INDEX.keys())})")

            story_path = get_file_with_prefix(str(run_path), "story")
            if story_path is None:
                raise FileNotFoundError(f"Nessun file 'story*' in {run_path}")

            arrays = arraysFromBlockFile(story_path)
            times_2d = arrays[0, :, :]
            yfull_2d = arrays[_YKEY_TO_INDEX[y_key], :, :]

            if times_2d.shape[0] == 0:
                raise RuntimeError(f"Nessuna traiettoria in {story_path}")

            indices = spec.get("curvesIndices") or spec.get("curvesIndeces")
            if indices is not None and len(indices) == 0:
                indices = None

            yfull_2d = _normalize_series(yfull_2d, y_key, spec)
            yGuides = spec.get("yGuides")
            xGuides = spec.get("xGuides")

            # === GEOMETRIA DETERMINISTICA (identica agli altri pannelli) ===
            DATA_W_IN   = 1.60
            DATA_H_IN   = 1.10
            LEFT_IN     = 0.40
            RIGHT_FRAME = 0.08
            BOTTOM_IN   = 0.36   # +0.02 per evitare tagli xlabel
            TOP_IN      = 0.16
            FIG_SCALE   = float(spec.get("fig_scale", 1.00))  # per 2 colonne: ~1.45

            # Gutter Y-hist più largo per leggibilità
            Y_HIST_GAP_IN = 0.06
            Y_HIST_W_IN   = 0.40
            show_yhist    = bool(spec.get("show_yhist", True))

            # Applica scala (non usiamo panel_width/height degli spec)
            L  = LEFT_IN   * FIG_SCALE
            R0 = RIGHT_FRAME * FIG_SCALE
            B  = BOTTOM_IN * FIG_SCALE
            T  = TOP_IN    * FIG_SCALE
            DW = DATA_W_IN * FIG_SCALE
            DH = DATA_H_IN * FIG_SCALE

            gap     = (Y_HIST_GAP_IN * FIG_SCALE) if show_yhist else 0.0
            yhist_w = (Y_HIST_W_IN   * FIG_SCALE) if show_yhist else 0.0
            R_total = R0 + gap + yhist_w

            # Figura con area-dati ESATTA (main)
            fig, ax_main, _meta = uplot.figure_single_fixed(
                data_w_in=DW, data_h_in=DH,
                left_in=L, right_in=R_total,
                bottom_in=B, top_in=T
            )
            fig.set_constrained_layout(False)
            try:
                fig.tight_layout = lambda *a, **k: None
            except Exception:
                pass

            # Axes Y-hist nel gutter destro (stessa altezza del main)
            ax_yhist = None
            if show_yhist:
                FW, FH = fig.get_size_inches()
                ax_yhist = fig.add_axes([
                    (L + DW + gap) / FW,
                    B / FH,
                    (yhist_w) / FW,
                    DH / FH
                ], sharey=ax_main)

            # === chiamata alla funzione (external_layout) ===
            mc_kwargs = dict(
                name=name,
                title="",
                x_list=times_2d,
                x_label=spec.get("x_label", "t"),
                y_list=yfull_2d,
                y_label=spec.get("y_label", y_key),
                curvesIndices=indices,
                namesForCurves=spec.get("namesForCurves"),
                show_xhist=bool(spec.get("show_xhist", False)),
                show_yhist=show_yhist,
                x_cdf=bool(spec.get("x_cdf", False)),
                y_cdf=bool(spec.get("y_cdf", True if show_yhist else True)),
                density=bool(spec.get("density", True)),
                y_log_density=bool(spec.get("y_log_density", False)),
                x_log_density=bool(spec.get("x_log_density", False)),
                y_bins='magnetizationComplete',
                x_bins=spec.get("x_bins", "auto"),
                legend=bool(spec.get("legend", False)),
                reserve_legend_space=bool(spec.get("reserve_legend_space", False)),
                manage_style=False,                         # stile lo imponiamo noi
                palette=spec.get("palette", "cb_safe"),
                yGuides=yGuides,
                xGuides=xGuides,
                cdf_ref_probs_y=spec.get("cdf_ref_probs_y"),
                cdf_ref_probs_x=spec.get("cdf_ref_probs_x"),
                # external layout hooks
                external_layout=True,
                fig=fig,
                ax_main=ax_main,
                ax_yhist=ax_yhist
            )

            fig, ax_main, meta = multipleCurvesAndHist(**mc_kwargs)
            ax_main.xaxis.label.set_fontsize(ax_main.xaxis.label.get_size() * 0.90)
            ax_main.yaxis.label.set_fontsize(ax_main.yaxis.label.get_size() * 0.90)
            # === stile coerente “paper” ===
            # linee nel main più visibili
            for ln in list(ax_main.lines):
                try:
                    ln.set_linewidth(1.4)
                except Exception:
                    pass
            global xTicks, xTicksLabels
            xTicks=spec.get("xticks",[0,20,40])
            xTicksLabels=[str(x) for x in xTicks]
            # spessori/ticks base + labelpad contenuto
            _style_main_axis(ax_main)
            ax_main.tick_params(direction='out', length=3.2, width=0.8, pad=2.0)
            ax_main.xaxis.labelpad = 1.4
            ax_main.yaxis.labelpad = 1.2

            if ax_yhist is not None:
                _style_hist_axis(ax_yhist, 'y')
                ax_yhist.tick_params(direction='out', length=3.2, width=0.8, pad=2.0)
                ax_yhist.tick_params(axis='y', which='both', length=0, width=0,
                     labelleft=False, labelright=False)
                # opzionale: nascondi le spine Y se non ti servono (estetica più pulita)
                for side in ('left', 'right'):
                    try:
                        ax_yhist.spines[side].set_visible(False)
                    except Exception:
                        pass

            # ticks fissi richiesti

            ax_main.set_xticks(xTicks)
            
            ax_main.set_yticks([-1, 0, 1])

            # guideline rossa nel pannello istogramma (se yGuides presente)
            if ax_yhist is not None and yGuides:
                try:
                    y_val = float(yGuides[-1][0])
                    # rimuovi eventuali duplicati “orizzontali”
                    for ln in list(ax_yhist.lines):
                        try:
                            yd = ln.get_ydata()
                            if len(yd) == 2 and np.isclose(yd[0], yd[1]) and np.isclose(yd[0], y_val):
                                ln.remove()
                        except Exception:
                            pass
                    # disegna la guideline rossa
                    g = ax_yhist.axhline(y_val, ls='--', color='red', linewidth=1.6)
                    g.set_dashes((3.3, 3.3))
                    g.set_dash_capstyle('butt')
                except Exception:
                    pass

            # === LOG ===
            W, H = fig.get_size_inches()
            bbox = ax_main.get_position()
            data_w = bbox.width * W
            data_h = bbox.height * H
            print(f"[FIG SIZE] {name}: {W:.3f} × {H:.3f} in")
            print(f"[AX BOX]   data: {data_w:.3f} × {data_h:.3f} in; margins L/R/B/T={L:.2f}/{R0:.2f}/{B:.2f}/{T:.2f} (+ gutter {gap:.2f}+{yhist_w:.2f})")

            # === export rigoroso ===
            out = spec.get("outfile") or _slug(name)
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            fmts = tuple(spec.get("formats", ["pdf", "png"]))
            dpi = int(spec.get("dpi", 300))
            uplot.export_figure_strict(fig, out, formats=fmts, dpi=dpi)
            plt.close(fig)
            print(f"Saved figure '{out}'.")


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        specs = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    else:
        specs = PLOTS
    run_from_specs(specs)
    print('Done.')


if __name__ == '__main__':
    main()