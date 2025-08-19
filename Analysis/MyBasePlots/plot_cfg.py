"""
plot_cfg_v6.py
==============
Configurazione *gold* (sola geometria) per figure paper‑ready.

Obiettivo: separare **completamente** la logica di dimensionamento dal resto.
Qui non si fa nessun "tight layout" o correzione runtime: forniamo solo
le **formule chiuse** per ricavare dimensioni figura e margini coerenti.
I template useranno questi numeri senza introdurre effetti collaterali.

Convenzioni:
- `PANEL.W_MAIN`, `PANEL.H_MAIN` sono le dimensioni TARGET (in pollici) del
  pannello *logico*. Nei template potrà significare "bbox Axes" (policy 'axes')
  oppure "zona dati" (policy 'data'), ma **qui** restiamo agnostici.
- Margini: `LEFT_FRAC` e `BOTTOM_FRAC` sono frazioni della figura;
  `RIGHT_PAD` e `TOP_PAD_*` sono *in pollici*.
- `legend_strip_width()` restituisce la larghezza (in pollici) da riservare
  a destra quando la legenda è "outside".

API minimale esposta da questo file (da usare nei template):
- `figsize_single_panel(title_lines=1) -> (fig_w, fig_h)`
- `figsize_two_panels(legend="outside", title_lines=1) -> (fig_w, fig_h, right_extra_in)`
- `axes_rect(fig_w, fig_h, right_extra_in=0.0, title_lines=1) -> (left,bottom,right,top)`
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

# ------------------------------------------------------------
# Scala globale (se vuoi scalare tutte le misure insieme)
# ------------------------------------------------------------
SCALE: float = 1.00
def sc(x: float) -> float: return x * SCALE

# ------------------------------------------------------------
# Pannello target (dimensioni logiche del pannello)
# ------------------------------------------------------------
@dataclass(frozen=True)
class Panel:
    W_MAIN: float = sc(4.2)   # [in]
    H_MAIN: float = sc(3.2)   # [in]
PANEL = Panel()

# ------------------------------------------------------------
# Spaziature & margini
# ------------------------------------------------------------
@dataclass(frozen=True)
class Gaps:
    GAP_W: float = sc(0.90)   # gap orizzontale tra due pannelli [in]
    GAP_H: float = sc(0.10)   # (non usato qui, utile per griglie più complesse)
GAPS = Gaps()

@dataclass(frozen=True)
class Margins:
    LEFT_FRAC: float        = 0.16   # frazione della figura
    BOTTOM_FRAC: float      = 0.18   # frazione della figura
    RIGHT_PAD: float        = sc(0.16)  # [in]
    TOP_PAD_BASE: float     = sc(0.52)  # [in]
    TOP_PAD_PER_LINE: float = sc(0.30)  # [in] aggiunta per ogni riga di titolo oltre la prima
MARGINS = Margins()

# ------------------------------------------------------------
# Legenda
# ------------------------------------------------------------
@dataclass(frozen=True)
class LegendCfg:
    PAD_W: float = sc(0.14)           # [in] distanza pannello -> legend
    W_MIN: float = sc(1.00)           # [in] stima larghezza legenda
    W_MAX: float = sc(2.20)           # [in]
LEGEND = LegendCfg()

def legend_strip_width() -> float:
    """Larghezza in pollici da riservare a destra quando la legenda è outside."""
    return LEGEND.PAD_W + 0.5*(LEGEND.W_MIN + LEGEND.W_MAX)

# ------------------------------------------------------------
# Ticks / label / linee (placeholders utili per altri moduli)
# ------------------------------------------------------------
@dataclass(frozen=True)
class TickCfg:
    X_MAJ: int = 5
    Y_MAJ: int = 5
    PRUNE: str = "both"   # {'lower','upper','both',None}
    MINOR: bool = False
TICKS = TickCfg()

@dataclass(frozen=True)
class LineScales:
    THIN: float = 0.60
    MEDIUM: float = 1.00
    STRONG: float = 1.15
    STRONG_MIN_PT: float = 1.20
LINES = LineScales()

# ------------------------------------------------------------
# DEFAULTS (usati dai template; qui solo per completezza)
# ------------------------------------------------------------
@dataclass(frozen=True)
class Defaults:
    AREA_POLICY: str = "data"     # 'data' (gold) o 'axes'
    LEGEND: str      = "outside"  # 'outside' o 'inside'
    TITLE_LINES: int = 1
DEFAULTS = Defaults()

# ------------------------------------------------------------
# Helpers interni
# ------------------------------------------------------------
def _top_pad(title_lines:int) -> float:
    """Padding superiore in pollici, in funzione del numero di righe di titolo."""
    return MARGINS.TOP_PAD_BASE + max(0, title_lines - 1)*MARGINS.TOP_PAD_PER_LINE

# ------------------------------------------------------------
# Figure sizes (FORMULE CHIUSE, nessuna approssimazione)
# ------------------------------------------------------------
def figsize_single_panel(title_lines:int=DEFAULTS.TITLE_LINES) -> Tuple[float,float]:
    """
    Formule esatte (margini misti):
      axes_w = fig_w*(1-LEFT_FRAC)  - RIGHT_PAD
      axes_h = fig_h*(1-BOTTOM_FRAC)- TOP_PAD
      -> fig_w = (W_MAIN + RIGHT_PAD)/(1-LEFT_FRAC)
      -> fig_h = (H_MAIN + TOP_PAD)/(1-BOTTOM_FRAC)
    """
    top = _top_pad(title_lines)
    w = (PANEL.W_MAIN + MARGINS.RIGHT_PAD) / (1.0 - MARGINS.LEFT_FRAC)
    h = (PANEL.H_MAIN + top) / (1.0 - MARGINS.BOTTOM_FRAC)
    return (w, h)

def figsize_two_panels(legend:str=DEFAULTS.LEGEND, title_lines:int=DEFAULTS.TITLE_LINES) -> Tuple[float,float,float]:
    """
    DUE pannelli affiancati (formule esatte).
      total_axes_w = 2*W_MAIN + GAP_W
      fig_w = (total_axes_w + RIGHT_PAD + right_extra) / (1-LEFT_FRAC)
      fig_h = come single
    Ritorna (fig_w, fig_h, right_extra_in).
    """
    right_extra = legend_strip_width() if legend == "outside" else 0.0
    total_axes_w = 2.0*PANEL.W_MAIN + GAPS.GAP_W
    w = (total_axes_w + MARGINS.RIGHT_PAD + right_extra) / (1.0 - MARGINS.LEFT_FRAC)
    top = _top_pad(title_lines)
    h = (PANEL.H_MAIN + top) / (1.0 - MARGINS.BOTTOM_FRAC)
    return (w, h, right_extra)

# ------------------------------------------------------------
# Rettangolo degli Axes in FRAZIONI figura
# ------------------------------------------------------------
def axes_rect(fig_w: float, fig_h: float, *, right_extra_in: float = 0.0, title_lines:int=DEFAULTS.TITLE_LINES) -> Tuple[float,float,float,float]:
    """
    Converte i margini misti in rettangolo (left,bottom,right,top) in FRAZIONI figura.
    Usare direttamente in `fig.subplots_adjust(*axes_rect(...))`.
    """
    left   = MARGINS.LEFT_FRAC
    bottom = MARGINS.BOTTOM_FRAC
    right  = 1.0 - (MARGINS.RIGHT_PAD + right_extra_in) / fig_w
    top    = 1.0 - (_top_pad(title_lines) / fig_h)
    return (left, bottom, right, top)
