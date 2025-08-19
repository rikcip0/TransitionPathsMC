# templates.py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Tuple, Optional

from MyBasePlots.plot_cfg import PANEL, GAPS, MARGINS, HIST, COLORBAR, LEGEND
from MyBasePlots.plot_cfg import figsize_single_panel, figsize_with_side_hist, figsize_with_colorbar
from MyBasePlots.myColors import apply_cycle

from contextlib import contextmanager
from MyBasePlots.utils_plot import exportFigure, standardize_ticks
from MyBasePlots.plot_cfg import TICKS
# import già esistenti dei tuoi create_* ...

@contextmanager
def standard_plot_session(name, outdir, to_pdf=False,
                          title=None, title_lines=1,
                          palette="okabe_ito_noy",
                          xbins=TICKS.X_MAJ, ybins=TICKS.Y_MAJ, minor=TICKS.MINOR):
    fig, ax = create_standard_figure(name=name, title_lines=title_lines, palette=None)
    if palette: apply_cycle(ax, palette)
    if title: ax.set_title(title)
    standardize_ticks(ax, xbins=xbins, ybins=ybins, minor=minor)
    try:
        yield fig, ax
    finally:
        exportFigure(fig, name, outdir, to_pdf=to_pdf)
        plt.close(fig)

@contextmanager
def sidehist_plot_session(name, outdir, to_pdf=False,
                          n_cols_hist=1, title=None, title_lines=1,
                          use_cdf_padding=False,
                          palette="okabe_ito_noy",
                          xbins=TICKS.X_MAJ, ybins=TICKS.Y_MAJ, minor=TICKS.MINOR):
    fig, ax_main, hist_axes, ax_leg = create_figure_with_side_hist(
        name=name, n_cols_hist=n_cols_hist, title_lines=title_lines,
        use_cdf_padding=use_cdf_padding, palette=None, xbins=xbins, ybins=ybins, minor=minor
    )
    if palette: apply_cycle(ax_main, palette)
    if title: ax_main.set_title(title)
    try:
        yield fig, (ax_main, hist_axes, ax_leg)
    finally:
        exportFigure(fig, name, outdir, to_pdf=to_pdf)
        plt.close(fig)

@contextmanager
def colorbar_plot_session(name, outdir, to_pdf=False,
                          vertical=True, title=None, title_lines=1,
                          palette="okabe_ito_noy",
                          xbins=TICKS.X_MAJ, ybins=TICKS.Y_MAJ, minor=TICKS.MINOR):
    fig, ax, cax = create_figure_with_colorbar(
        name=name, vertical=vertical, title_lines=title_lines,
        palette=None, xbins=xbins, ybins=ybins, minor=minor
    )
    if palette: apply_cycle(ax, palette)
    if title: ax.set_title(title)
    try:
        yield fig, (ax, cax)
    finally:
        exportFigure(fig, name, outdir, to_pdf=to_pdf)
        plt.close(fig)


# ---- helper per convertire inches -> frazioni figura ----
def _frac(total: float, inches: float) -> float:
    return inches / total

# ---- SINGLE PANEL ----
def create_standard_figure(
    name: Optional[str] = None,
    title_lines: int = 1,
    palette: Optional[str] = None,
    xbins: int = 5, ybins: int = 5, minor: bool = False,
):
    fig_w, fig_h = figsize_single_panel(title_lines=title_lines)
    fig = plt.figure(name, figsize=(fig_w, fig_h))

    # costruiamo righe/colonne in inches e convertiamo in frazioni
    left_frac = MARGINS.LEFT_FRAC
    right_frac = 1.0 - _frac(fig_w, MARGINS.RIGHT_PAD)
    bottom_frac = MARGINS.BOTTOM_FRAC
    top_pad = MARGINS.TOP_PAD_BASE + max(0, title_lines - 1) * MARGINS.TOP_PAD_PER_LINE
    top_frac = 1.0 - _frac(fig_h, top_pad)

    fig.subplots_adjust(left=left_frac, right=right_frac, bottom=bottom_frac, top=top_frac)
    ax = fig.add_subplot(1, 1, 1)

    if palette:
        apply_cycle(ax, palette)

    standardize_ticks(ax, xbins=xbins, ybins=ybins, minor=minor)
    return fig, ax

# ---- SIDE HISTOGRAM ----
def create_figure_with_side_hist(
    name: Optional[str] = None,
    n_cols_hist: int = 1,
    title_lines: int = 1,
    use_cdf_padding: bool = False,
    palette: Optional[str] = None,
    xbins: int = 5, ybins: int = 5, minor: bool = False,
):
    """
    Layout: [LEFT_MARGIN][MAIN][GAP][HIST...][LEG_PAD][LEGEND][RIGHT_PAD]
            con due righe: [TOP_PAD][MAIN_ROW][BOTTOM_MARGIN]
    """
    fig_w, fig_h = figsize_with_side_hist(n_cols_hist=n_cols_hist, title_lines=title_lines, use_cdf_padding=use_cdf_padding)
    fig = plt.figure(name, figsize=(fig_w, fig_h))

    # Blocchi in inches
    w_left   = fig_w * MARGINS.LEFT_FRAC
    w_main   = PANEL.W_MAIN
    parts = [w_left, w_main, GAPS.GAP_W]
    for _ in range(n_cols_hist):
        parts += [HIST.W_COL, GAPS.GAP_W]
    leg_pad = LEGEND.PAD_W_CDF if use_cdf_padding else LEGEND.PAD_W
    leg_w   = 0.5 * (LEGEND.W_MIN + LEGEND.W_MAX)
    parts += [leg_pad, leg_w, MARGINS.RIGHT_PAD]
    width_ratios = [p / fig_w for p in parts]

    top_pad = MARGINS.TOP_PAD_BASE + max(0, title_lines - 1) * MARGINS.TOP_PAD_PER_LINE
    h_top   = top_pad
    h_main  = PANEL.H_MAIN
    h_bot   = fig_h * MARGINS.BOTTOM_FRAC
    height_ratios = [h_top/fig_h, h_main/fig_h, h_bot/fig_h]

    gs = GridSpec(nrows=3, ncols=len(width_ratios), width_ratios=width_ratios, height_ratios=height_ratios, figure=fig)

    # asse principale: riga 1 (centrale), colonna 1 (dopo il margine sinistro)
    ax_main = fig.add_subplot(gs[1, 1])

    # assi istogramma nelle colonne dedicate (saltando i gap)
    hist_axes = []
    col = 3  # 0: left, 1: main, 2: gap, 3: primo hist
    for i in range(n_cols_hist):
        axh = fig.add_subplot(gs[1, col])
        hist_axes.append(axh)
        col += 2  # dopo ogni hist c'è un gap

    # asse legenda (ultima colonna prima di RIGHT_PAD)
    ax_leg = fig.add_subplot(gs[1, -2])
    ax_leg.axis("off")  # ci disegnerai la legend con fig.legend(...) o ax_leg.legend(...)

    if palette:
        apply_cycle(ax_main, palette)

    standardize_ticks(ax_main, xbins=xbins, ybins=ybins, minor=minor)
    for axh in hist_axes:
        # ticks meno invasivi sugli istogrammi
        standardize_ticks(axh, xbins=xbins, ybins=xbins, minor=False)
        # riduci font tick (gestito da mplstyle + scaling nei label se vuoi fine-tuning)
        for lbl in axh.get_xticklabels()+axh.get_yticklabels():
            lbl.set_fontsize(lbl.get_fontsize()*HIST.TICK_SCALE)

    return fig, ax_main, hist_axes, ax_leg

# ---- COLORBAR ----
def create_figure_with_colorbar(
    name: Optional[str] = None,
    vertical: bool = True,
    title_lines: int = 1,
    palette: Optional[str] = None,
    xbins: int = 5, ybins: int = 5, minor: bool = False,
):
    """
    Se vertical=True: colorbar a destra del main (pannello main costante, figura più larga).
    Se vertical=False: colorbar sotto (figura più alta).
    """
    fig_w, fig_h = figsize_with_colorbar(vertical=vertical, title_lines=title_lines)
    fig = plt.figure(name, figsize=(fig_w, fig_h))

    if vertical:
        w_left = fig_w * MARGINS.LEFT_FRAC
        w_main = PANEL.W_MAIN
        parts = [w_left, w_main, COLORBAR.GAP, COLORBAR.W_VERT, MARGINS.RIGHT_PAD]
        width_ratios = [p/fig_w for p in parts]

        top_pad = MARGINS.TOP_PAD_BASE + max(0, title_lines - 1) * MARGINS.TOP_PAD_PER_LINE
        h_top   = top_pad
        h_main  = PANEL.H_MAIN
        h_bot   = fig_h * MARGINS.BOTTOM_FRAC
        height_ratios = [h_top/fig_h, h_main/fig_h, h_bot/fig_h]

        gs = GridSpec(nrows=3, ncols=len(width_ratios), width_ratios=width_ratios, height_ratios=height_ratios, figure=fig)
        ax  = fig.add_subplot(gs[1, 1])
        cax = fig.add_subplot(gs[1, 3])
    else:
        # larghezza come single panel, altezza con CB orizzontale
        w_left = fig_w * MARGINS.LEFT_FRAC
        w_main = PANEL.W_MAIN
        w_right_pad = MARGINS.RIGHT_PAD
        width_ratios = [w_left/fig_w, w_main/fig_w, w_right_pad/fig_w]

        top_pad = MARGINS.TOP_PAD_BASE + max(0, title_lines - 1) * MARGINS.TOP_PAD_PER_LINE
        h_top   = top_pad
        h_main  = PANEL.H_MAIN
        parts_h = [h_top, h_main, COLORBAR.GAP, COLORBAR.H_HORZ, fig_h * MARGINS.BOTTOM_FRAC]
        height_ratios = [p/fig_h for p in parts_h]

        gs = GridSpec(nrows=len(height_ratios), ncols=3, width_ratios=width_ratios, height_ratios=height_ratios, figure=fig)
        ax  = fig.add_subplot(gs[1, 1])
        cax = fig.add_subplot(gs[3, 1])  # sotto il main

    if palette:
        apply_cycle(ax, palette)

    standardize_ticks(ax, xbins=xbins, ybins=ybins, minor=minor)
    return fig, ax, cax
