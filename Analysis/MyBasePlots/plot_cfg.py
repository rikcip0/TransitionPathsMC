"""
plot_cfg.py
Parametri geometrici e di scaling per figure "paper-ready".

Tutto è espresso in pollici (Matplotlib usa inches).
Usiamo una SCALA UNIVERSALE (SCALE): variando SCALE, ridimensioni
tutte le lunghezze mantenendo i rapporti.

Struttura:
- BASE: scala e utility
- PANEL: dimensioni pannello principale
- GAPS: spaziature tra pannelli
- MARGINS: margini esterni (in frac o inches)
- LEGEND: spazio e limiti legenda
- HIST: pannelli istogramma marginali
- COLORBAR: dimensioni CB verticale/orizzontale
- LINES: fattori di spessore coerenti per elementi "sottili/medi/forti"
- TICKS: regole standard per bins/pruning/minor
- COLOR/CMAP: registri di palette e colormap (nomi simbolici)
- HELPERS: funzioni per calcolare figsize totali in layout comuni
"""

from dataclasses import dataclass
from typing import Tuple

# =========================
# BASE
# =========================
SCALE: float = 1.00
# Modifica SCALE per fare "zoom" globale dell'intera geometria:
# ad es. 1.10 = +10% su TUTTE le lunghezze fisiche.

def sc(x: float) -> float:
    """Applica la scala universale alle lunghezze in inches."""
    return x * SCALE

# =========================
# PANEL (main panel size)
# =========================
@dataclass(frozen=True)
class Panel:
    W_MAIN: float = sc(4.2)   # larghezza pannello principale
    H_MAIN: float = sc(3.2)   # altezza pannello principale

PANEL = Panel()

# =========================
# GAPS (spazi tra pannelli)
# =========================
@dataclass(frozen=True)
class Gaps:
    GAP_W: float = sc(0.10)   # gap orizzontale tra main e colonna ausiliaria
    GAP_H: float = sc(0.10)   # gap verticale tra main e riga ausiliaria

GAPS = Gaps()

# =========================
# MARGINS (margini esterni)
# =========================
@dataclass(frozen=True)
class Margins:
    LEFT_FRAC: float      = 0.16  # frazione della larghezza figura (per y-label + ticks)
    BOTTOM_FRAC: float    = 0.18  # frazione dell'altezza figura (per x-label + ticks)
    RIGHT_PAD: float      = sc(0.16)  # bordo destro fisso in inches
    TOP_PAD_BASE: float   = sc(0.52)  # bordo alto base in inches
    TOP_PAD_PER_LINE: float = sc(0.30) # incremento per linea extra di titolo

MARGINS = Margins()

# =========================
# LEGEND (spazio legenda)
# =========================
@dataclass(frozen=True)
class LegendCfg:
    PAD_W: float       = sc(0.12)  # spazio tra main e legenda (caso base)
    PAD_W_CDF: float   = sc(0.16)  # spazio se presenti CDF/bande, un po' più largo
    W_MIN: float       = sc(1.00)  # min width legenda
    W_MAX: float       = sc(2.20)  # max width legenda
    # Posizionamento standard (per "plot sciolti" senza asse legenda dedicato):
    # anchor rispetto alla FIGURA (non all'axes), upper-right
    ANCHOR_FIG: Tuple[float, float] = (0.965, 0.98)

LEGEND = LegendCfg()

# =========================
# HIST (istogrammi marginali)
# =========================
@dataclass(frozen=True)
class HistCfg:
    W_COL: float         = sc(0.90)   # larghezza colonna istogramma a destra
    H_ROW: float         = sc(0.90)   # altezza riga istogramma in alto
    TICK_SCALE: float    = 0.88       # scala font tick per pannelli istogramma

HIST = HistCfg()

# =========================
# COLORBAR
# =========================
@dataclass(frozen=True)
class ColorbarCfg:
    W_VERT: float       = sc(0.30)  # larghezza CB verticale
    H_HORZ: float       = sc(0.20)  # altezza CB orizzontale
    GAP: float          = sc(0.16)  # gap tra main e CB

COLORBAR = ColorbarCfg()

# =========================
# LINES (rapporti coerenti)
# Questi sono fattori *relativi* a lines.linewidth del .mplstyle
# =========================
@dataclass(frozen=True)
class LineScales:
    THIN: float        = 0.60   # linee sottili (guide/reference) = 60% dello spessore base
    MEDIUM: float      = 1.00   # linee standard = 100%
    STRONG: float      = 1.15   # p.es. curve CDF o media un po' più marcate
    STRONG_MIN_PT: float = 1.20 # minimo assoluto in pt per le "strong" (applicalo in codice se necessario)

LINES = LineScales()

# =========================
# TICKS (regole standard)
# =========================
@dataclass(frozen=True)
class TickCfg:
    X_MAJ: int      = 5     # numero target di tick major su X
    Y_MAJ: int      = 5     # idem Y
    PRUNE: str      = "both" # rimuovi tick di bordo se affollati
    MINOR: bool     = False  # minor tick off by default

TICKS = TickCfg()

# =========================
# COLOR / CMAP registri simbolici
# (il ciclo colori principale è nel .mplstyle; qui alias/nomi)
# =========================
@dataclass(frozen=True)
class PaletteRegistry:
    DEFAULT: str = "cb_safe"   # Okabe–Ito
    ALT: str     = "tab10"     # alternativa compatibile matplotlib

PALETTES = PaletteRegistry()

@dataclass(frozen=True)
class CmapRegistry:
    DEFAULT: str   = "viridis"   # percettivamente uniforme
    SEQUENTIAL: str = "viridis"
    DIVERGING: str  = "coolwarm" # o 'RdBu_r' se preferisci
    CYCLIC: str     = "twilight"

CMAPS = CmapRegistry()

# =========================
# HELPERS (calcolo figsize)
# =========================
def figsize_single_panel(title_lines: int = 1) -> Tuple[float, float]:
    """
    Figure con solo pannello principale.
    Restituisce (fig_w, fig_h) in inches.
    """
    Wm, Hm = PANEL.W_MAIN, PANEL.H_MAIN
    # larghezza figura: main / (1 - LEFT_FRAC) + RIGHT_PAD
    fig_w = Wm / (1.0 - MARGINS.LEFT_FRAC) + MARGINS.RIGHT_PAD
    # altezza figura: main / (1 - BOTTOM_FRAC) + top pads
    top_pad = MARGINS.TOP_PAD_BASE + max(0, title_lines - 1) * MARGINS.TOP_PAD_PER_LINE
    fig_h = Hm / (1.0 - MARGINS.BOTTOM_FRAC) + top_pad
    return (fig_w, fig_h)

def figsize_with_side_hist(n_cols_hist: int = 1, title_lines: int = 1, use_cdf_padding: bool = False) -> Tuple[float, float]:
    """
    Figure con istogrammo/i laterale/i (colonne a destra del main).
    Calcola width aggiungendo n_cols_hist*(W_COL + GAP_W) e spazio legenda (min..max).
    L'altezza è come single panel (istogrammi laterali non cambiano l'altezza).
    """
    base_w, base_h = figsize_single_panel(title_lines=title_lines)
    # Rimuovi RIGHT_PAD aggiunto dalla single_panel (verrà riaggiunto dopo)
    base_w_no_right = base_w - MARGINS.RIGHT_PAD
    # spazio per istogrammi e gap
    extra_w = 0.0
    if n_cols_hist > 0:
        extra_w += n_cols_hist * (HIST.W_COL + GAPS.GAP_W)
    # spazio per legenda (stimato in mezzo tra min e max)
    leg_pad = LEGEND.PAD_W_CDF if use_cdf_padding else LEGEND.PAD_W
    leg_w   = 0.5 * (LEGEND.W_MIN + LEGEND.W_MAX)
    total_w = base_w_no_right + extra_w + leg_pad + leg_w + MARGINS.RIGHT_PAD
    return (total_w, base_h)

def figsize_with_colorbar(vertical: bool = True, title_lines: int = 1) -> Tuple[float, float]:
    """
    Figure con colorbar a lato (vertical=True) oppure sotto (vertical=False).
    """
    base_w, base_h = figsize_single_panel(title_lines=title_lines)
    if vertical:
        # aggiungi CB verticale (larghezza + gap); altezza invariata
        w = base_w - MARGINS.RIGHT_PAD + COLORBAR.GAP + COLORBAR.W_VERT + MARGINS.RIGHT_PAD
        h = base_h
    else:
        # aggiungi CB orizzontale sotto (altezza + gap); larghezza invariata
        w = base_w
        h = base_h + COLORBAR.GAP + COLORBAR.H_HORZ
    return (w, h)
