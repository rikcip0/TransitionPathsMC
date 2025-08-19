"""
myTemplates.py
===================
Template due pannelli *deterministico* con supporto:
- inner_labels: 'outer' (default) oppure 'both'
- gap_w: 'cfg' (default) | 'auto' | float (pollici)
  * 'cfg'  -> usa cfg.GAPS.GAP_W
  * float  -> usa quel valore
  * 'auto' -> misura i ticklabels interni dopo il draw e **se serve** espande la figura
             aumentando SOLO il gap centrale; la larghezza dei pannelli (W_MAIN) resta invariata.

Niente tight_layout, niente correzioni runtime sulla zona-dati (si possono reintrodurre dopo).
"""
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Tuple

import plot_cfg as cfg
from utils_plot import outerize_two_columns, recommend_gap_between

# --- stile opzionale ---
try:
    from utils_style import auto_style
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def auto_style(): yield

# ------------------ SINGLE (come v6_2) ------------------
def single_panel(title: str | None = None, legend: str | None = None, title_lines: int | None = None):
    if title_lines is None: title_lines = cfg.DEFAULTS.TITLE_LINES
    if legend is None: legend = cfg.DEFAULTS.LEGEND
    fig_w, fig_h = cfg.figsize_single_panel(title_lines=title_lines)
    right_extra = cfg.legend_strip_width() if legend == "outside" else 0.0
    if right_extra > 0:
        fig_w = (cfg.PANEL.W_MAIN + cfg.MARGINS.RIGHT_PAD + right_extra) / (1.0 - cfg.MARGINS.LEFT_FRAC)
    fig = plt.figure(figsize=(fig_w, fig_h))
    left = cfg.MARGINS.LEFT_FRAC
    bottom = cfg.MARGINS.BOTTOM_FRAC
    right  = 1.0 - (cfg.MARGINS.RIGHT_PAD + right_extra) / fig_w
    top    = 1.0 - (cfg.MARGINS.TOP_PAD_BASE + max(0, title_lines-1)*cfg.MARGINS.TOP_PAD_PER_LINE) / fig_h
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    ax = fig.add_subplot(1,1,1)
    if title: ax.set_title(title)
    fig._tpl = {"legend": legend}
    return fig, ax

def finalize_single(fig, ax, legend: str | None = None, loc_inside: str = "upper right"):
    if legend is None: legend = fig._tpl.get("legend", cfg.DEFAULTS.LEGEND)
    if legend == "outside":
        right_frac = ax.get_position().x1
        fw = fig.get_size_inches()[0]
        x = right_frac + cfg.LEGEND.PAD_W/fw + 1e-3
        h, l = ax.get_legend_handles_labels()
        if h:
            fig.legend(h, l, loc="upper left", bbox_to_anchor=(x,1.0), frameon=False)
    else:
        ax.legend(loc=loc_inside, frameon=False)
    return fig, ax

# ------------------ TWO PANELS (auto-gap) ------------------
def _parts_inches(fig_w: float, fig_h: float, gap_w_in: float, title_lines:int, legend:str) -> Tuple[list, list, float]:
    right_extra = cfg.legend_strip_width() if legend == "outside" else 0.0
    parts_w = [fig_w*cfg.MARGINS.LEFT_FRAC, cfg.PANEL.W_MAIN, gap_w_in, cfg.PANEL.W_MAIN]
    if right_extra > 0:
        parts_w += [right_extra, cfg.MARGINS.RIGHT_PAD]
    else:
        parts_w += [cfg.MARGINS.RIGHT_PAD]
    parts_h = [cfg.MARGINS.TOP_PAD_BASE + max(0, title_lines-1)*cfg.MARGINS.TOP_PAD_PER_LINE,
               cfg.PANEL.H_MAIN, fig_h*cfg.MARGINS.BOTTOM_FRAC]
    return parts_w, parts_h, right_extra

def two_panels(title: str | None = None, *, legend: str | None = None, labels: bool = True,
               title_lines: int | None = None, inner_labels: str = "outer", gap_w='cfg'):
    if title_lines is None: title_lines = cfg.DEFAULTS.TITLE_LINES
    if legend is None: legend = cfg.DEFAULTS.LEGEND
    # gap iniziale
    if gap_w == 'cfg':
        gap_in = cfg.GAPS.GAP_W
    elif gap_w == 'auto':
        gap_in = cfg.GAPS.GAP_W  # provvisorio, poi auto-espandiamo se serve
    elif isinstance(gap_w, (int, float)):
        gap_in = float(gap_w)
    else:
        raise ValueError("gap_w must be 'cfg'|'auto'|float")

    # dimensioni figura da cfg (con gap_in iniziale)
    fig_w, fig_h, _re = cfg.figsize_two_panels(legend=legend, title_lines=title_lines)
    # NOTA: la formula di cfg usa GAPS.GAP_W; se gap_w è float diverso, adeguiamo fig_w
    base_gap = cfg.GAPS.GAP_W
    if abs(gap_in - base_gap) > 1e-6:
        fig_w += (gap_in - base_gap)

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)  # niente doppio margine
    parts_w, parts_h, right_extra = _parts_inches(fig_w, fig_h, gap_in, title_lines, legend)
    gs = GridSpec(nrows=3, ncols=len(parts_w),
                  width_ratios=parts_w, height_ratios=parts_h, figure=fig,
                  wspace=0.0, hspace=0.0)
    a1 = fig.add_subplot(gs[1,1])
    a2 = fig.add_subplot(gs[1,3])
    if title: a1.set_title(title)
    if labels:
        try:
            from utils_style import apply_panel_labels
            apply_panel_labels(fig, [a1, a2], labels=["a","b"])
        except Exception:
            pass
    # policy label
    if inner_labels == "outer":
        outerize_two_columns(a1, a2, right_ticks_out=True)
    elif inner_labels != "both":
        raise ValueError("inner_labels must be 'outer' or 'both'")

    fig._tpl = {"legend": legend, "gap_in": gap_in, "title_lines": title_lines}
    return fig, (a1, a2)

def finalize_multi(fig, axes, legend: str | None = None, auto_gap_action: str = "expand"):
    """
    Se fig._tpl['gap_in'] == 'auto' (gestito a monte come valore cfg), misura i ticklabels interni e,
    se servono più pollici, **espande** solo la larghezza figura di delta_gap mantenendo W_MAIN invariato.
    auto_gap_action: 'expand'|'suppress'|'warn'
    - 'expand'   -> aumenta fig_w e riposiziona gli assi con nuove frazioni
    - 'suppress' -> spegne i label interni (equivalente a inner_labels='outer')
    - 'warn'     -> non tocca la figura, stampa solo un avviso
    """
    if legend is None: legend = fig._tpl.get("legend", cfg.DEFAULTS.LEGEND)
    a1, a2 = axes
    used_gap = fig._tpl.get("gap_in", cfg.GAPS.GAP_W)
    title_lines = fig._tpl.get("title_lines", cfg.DEFAULTS.TITLE_LINES)

    # --- misurazione gap necessario (solo se i label interni sono attivi) ---
    need_gap = recommend_gap_between(a1, a2, min_gap_in=used_gap)

    if need_gap > used_gap + 1e-3 and auto_gap_action in ("expand", "suppress"):
        if auto_gap_action == "suppress":
            outerize_two_columns(a1, a2, right_ticks_out=True)
        else:
            # EXPAND: aumenta fig_w e riposiziona assi mantenendo W_MAIN invariato
            fig_w, fig_h = fig.get_size_inches()
            delta = need_gap - used_gap
            fig_w_new = fig_w + delta
            fig.set_size_inches(fig_w_new, fig_h, forward=True)

            # ricalcola posizioni in frazioni figura con il nuovo fig_w
            L_in = fig_w_new * cfg.MARGINS.LEFT_FRAC
            H_in = cfg.PANEL.H_MAIN
            W_in = cfg.PANEL.W_MAIN
            top_pad = cfg.MARGINS.TOP_PAD_BASE + max(0, title_lines-1)*cfg.MARGINS.TOP_PAD_PER_LINE
            y0 = (fig_h * cfg.MARGINS.BOTTOM_FRAC) / fig_h
            height_frac = H_in / fig_h

            # colonne in inches: [L, W, need_gap, W, (strip?), RIGHT_PAD]
            # calcola x frazionari
            x0_left  = L_in / fig_w_new
            x0_right = (L_in + W_in + need_gap) / fig_w_new
            width_frac = W_in / fig_w_new

            a1.set_position([x0_left, y0, width_frac, height_frac])
            a2.set_position([x0_right, y0, width_frac, height_frac])

    # --- legenda ---
    if legend == "outside":
        right_frac = a2.get_position().x1
        fw = fig.get_size_inches()[0]
        x = right_frac + cfg.LEGEND.PAD_W/fw + 1e-3
        h, l = a2.get_legend_handles_labels()
        if h:
            fig.legend(h, l, loc="upper left", bbox_to_anchor=(x,1.0), frameon=False)
    else:
        axes[0].legend(loc="upper right", frameon=False)
    return fig, axes
