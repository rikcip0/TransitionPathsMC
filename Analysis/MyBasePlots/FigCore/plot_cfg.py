
"""
plot_cfg_v2.py — GOLD base di configurazione (versionato)
---------------------------------------------------------
- Unità di misura robuste (mm/pt/in).
- Target editoriali (Nature/APS/Springer) per larghezze colonna.
- Minimi tipografici consigliati (font/line/tick) per stampa.
- Dimensioni PANEL DATA (area assi) e MARGINI in inches, deterministici.
- Helper per: figsize single/doppia, rettangoli axes in frazione figura,
  e griglie N×M con gap e slot opzionali per colorbar/legend esterne.
- Nessun bbox tight implicito.

Nota: non gestisce palette/marker (delegati altrove).
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Optional
import math

# ======================= UNIT CONVERSIONS =======================

class UNITS:
    MM_PER_IN = 25.4
    PT_PER_IN = 72.27
    @staticmethod
    def mm_to_in(mm: float) -> float: return mm / UNITS.MM_PER_IN
    @staticmethod
    def in_to_mm(inch: float) -> float: return inch * UNITS.MM_PER_IN
    @staticmethod
    def pt_to_in(pt: float) -> float: return pt / UNITS.PT_PER_IN
    @staticmethod
    def in_to_pt(inch: float) -> float: return inch * UNITS.PT_PER_IN
    @staticmethod
    def mm_to_pt(mm: float) -> float: return UNITS.in_to_pt(UNITS.mm_to_in(mm))
    @staticmethod
    def pt_to_mm(pt: float) -> float: return UNITS.in_to_mm(UNITS.pt_to_in(pt))

# ======================= JOURNAL TARGETS ========================

class JOURNAL:
    # Nature (Research Articles)
    NATURE_1COL_MM = 89.0
    NATURE_1P5COL_MM = 133.0   # opzione intermedia
    NATURE_2COL_MM = 183.0
    NATURE_MAX_H_MM = 247.0

    # APS (Physical Review)
    APS_1COL_MM = 86.0
    APS_2COL_MM = 178.0
    APS_MAX_H_MM = 247.0

    # Springer (tipico, es. Oecologia)
    SPR_1COL_MM = 84.0
    SPR_2COL_MM = 174.0
    SPR_MAX_H_MM = 234.0

# ======================= MINIMI TIPOGRAFICI =====================

FONT_MIN_PRINT_PT   = 6.0   # minimo assoluto leggibile
FONT_TARGET_PRINT_PT= 8.0   # target per etichette/annotazioni
LINE_MIN_PT         = 0.5   # spessore minimo curve/assi (≈0.18 mm)
TICK_MAJOR_LEN_PT   = 2.0   # lunghezza minima tacche maggiori

# ======================= PANEL / MARGINS ========================

class PANEL_DATA:
    """Dimensioni della DATA BOX (area assi) in inches."""
    WIDTH_IN  = UNITS.mm_to_in(JOURNAL.NATURE_1COL_MM)  # 89 mm ≈ 3.504"
    HEIGHT_IN = 2.40                                    # altezza dati equilibrata

class PANEL_MARGINS_IN:
    """Margini tra figura e DATA BOX (inches)."""
    LEFT   = 0.42
    RIGHT  = 0.08
    TOP    = 0.20
    BOTTOM = 0.34

class GAPS:
    """Spazio tra DATA BOX adiacenti (inches)."""
    W_IN = 0.10     # orizzontale
    H_IN = 0.10     # verticale

class LEGEND_SLOT:
    """Slot dedicato (inches) se si riserva una colonna/rigaper legenda esterna."""
    WIDTH_IN  = 0.18
    HEIGHT_IN = 0.24
    PAD_IN    = 0.06

class CBAR:
    """Barra colore (inches)."""
    WIDTH_IN  = 0.18
    PAD_IN    = 0.06

# ======================= EXPORT POLICY ==========================

class EXPORT:
    DPI_RASTER = 300
    INCLUDE_METADATA = True

# ======================= FIGSIZE PREDEFINITE ====================

class DEFAULTS:
    """Figura PANEL BOX (PANEL = DATA + MARGINI) in inches."""
    PANEL_W_IN = PANEL_MARGINS_IN.LEFT + PANEL_DATA.WIDTH_IN + PANEL_MARGINS_IN.RIGHT
    PANEL_H_IN = PANEL_MARGINS_IN.BOTTOM + PANEL_DATA.HEIGHT_IN + PANEL_MARGINS_IN.TOP

    FIGSIZE_SINGLE = (PANEL_W_IN, PANEL_H_IN)

    FIGSIZE_DOUBLE = (
        PANEL_MARGINS_IN.LEFT + PANEL_DATA.WIDTH_IN + GAPS.W_IN
        + PANEL_DATA.WIDTH_IN + PANEL_MARGINS_IN.RIGHT,
        PANEL_H_IN
    )

# ======================= HELPERS DI DIMENSIONAMENTO =============

def panel_box_figsize_single(height_in: Optional[float] = None) -> Tuple[float, float]:
    """Figura a 1 pannello: (W,H) del PANEL BOX in inches."""
    h_data = height_in if height_in is not None else PANEL_DATA.HEIGHT_IN
    W = PANEL_MARGINS_IN.LEFT + PANEL_DATA.WIDTH_IN + PANEL_MARGINS_IN.RIGHT
    H = PANEL_MARGINS_IN.BOTTOM + h_data + PANEL_MARGINS_IN.TOP
    return (W, H)

def panel_box_figsize_two_panels(height_in: Optional[float] = None,
                                 gap_in: Optional[float] = None) -> Tuple[float, float]:
    """Figura a 2 pannelli affiancati: (W,H) del PANEL BOX in inches."""
    h_data = height_in if height_in is not None else PANEL_DATA.HEIGHT_IN
    gap = gap_in if gap_in is not None else GAPS.W_IN
    W = (PANEL_MARGINS_IN.LEFT + PANEL_DATA.WIDTH_IN + gap
         + PANEL_DATA.WIDTH_IN + PANEL_MARGINS_IN.RIGHT)
    H = PANEL_MARGINS_IN.BOTTOM + h_data + PANEL_MARGINS_IN.TOP
    return (W, H)

def axes_rect_from_cfg(height_in: Optional[float] = None,
                       fig_w_in: Optional[float] = None,
                       fig_h_in: Optional[float] = None) -> List[float]:
    """
    Ritorna [x0,y0,w,h] in FRAZIONE FIGURA per posizionare l'Axes in modo che:
    - DATA BOX abbia dimens. PANEL_DATA.(WIDTH_IN, HEIGHT_IN o height_in)
    - i margini PANEL_MARGINS_IN siano rispettati in inches.
    """
    # Dimensioni figura (inches)
    if fig_w_in is None or fig_h_in is None:
        W, H = panel_box_figsize_single(height_in=height_in)
    else:
        W, H = fig_w_in, fig_h_in
    # DATA BOX desiderato
    w_data = PANEL_DATA.WIDTH_IN
    h_data = height_in if height_in is not None else PANEL_DATA.HEIGHT_IN
    # Margini assoluti
    L, R, T, B = (PANEL_MARGINS_IN.LEFT, PANEL_MARGINS_IN.RIGHT,
                  PANEL_MARGINS_IN.TOP, PANEL_MARGINS_IN.BOTTOM)
    # Converti in frazioni figura
    x0 = L / W
    y0 = B / H
    w  = w_data / W
    h  = h_data / H
    return [x0, y0, w, h]

# ======================= GRIGLIE AVANZATE =======================

def grid_figsize(nrows: int, ncols: int,
                 data_w_in: Optional[float] = None,
                 data_h_in: Optional[float] = None,
                 gaps: Tuple[float, float] = (GAPS.W_IN, GAPS.H_IN),
                 reserve_cbar_right: bool = False,
                 reserve_legend_right: bool = False) -> Tuple[float, float]:
    """
    Calcola la figsize per una griglia nrows×ncols di pannelli dati.
    Opzioni: riserva una colonna per colorbar o legenda a destra.
    """
    dw = data_w_in if data_w_in is not None else PANEL_DATA.WIDTH_IN
    dh = data_h_in if data_h_in is not None else PANEL_DATA.HEIGHT_IN
    gw, gh = gaps

    extra_w = 0.0
    if reserve_cbar_right:
        extra_w += CBAR.PAD_IN + CBAR.WIDTH_IN
    if reserve_legend_right:
        extra_w += LEGEND_SLOT.PAD_IN + LEGEND_SLOT.WIDTH_IN

    W = PANEL_MARGINS_IN.LEFT + ncols * dw + (ncols - 1) * gw + extra_w + PANEL_MARGINS_IN.RIGHT
    H = PANEL_MARGINS_IN.BOTTOM + nrows * dh + (nrows - 1) * gh + PANEL_MARGINS_IN.TOP
    return (W, H)

def grid_axes_rects(nrows: int, ncols: int,
                    fig_w_in: float, fig_h_in: float,
                    data_w_in: Optional[float] = None,
                    data_h_in: Optional[float] = None,
                    gaps: Tuple[float, float] = (GAPS.W_IN, GAPS.H_IN),
                    reserve_cbar_right: bool = False,
                    reserve_legend_right: bool = False) -> List[List[float]]:
    """
    Ritorna una lista di rettangoli [x0,y0,w,h] (frazione figura) per ogni cella dati della griglia.
    Le eventuali colonne riservate per cbar/legend NON sono incluse nei rettangoli dati.
    Ordinamento: riga 0 in alto, colonna 0 a sinistra (convenzione spettrale).
    """
    dw = data_w_in if data_w_in is not None else PANEL_DATA.WIDTH_IN
    dh = data_h_in if data_h_in is not None else PANEL_DATA.HEIGHT_IN
    gw, gh = gaps

    # Calcola offset sinistra/alto in inches
    left_in = PANEL_MARGINS_IN.LEFT
    bottom_in = PANEL_MARGINS_IN.BOTTOM
    # Se si riserva a destra, i pannelli non cambiano x0; lo spazio si aggiunge a destra
    rects: List[List[float]] = []
    total_w_in = fig_w_in
    total_h_in = fig_h_in

    for r in range(nrows):
        for c in range(ncols):
            x_in = left_in + c * (dw + gw)
            y_in = bottom_in + (nrows - 1 - r) * (dh + gh)
            rects.append([
                x_in / total_w_in,
                y_in / total_h_in,
                dw / total_w_in,
                dh / total_h_in
            ])
    return rects

def right_cbar_rect(fig_w_in: float, fig_h_in: float,
                    nrows: int, ncols: int,
                    data_h_in: Optional[float] = None,
                    total_data_h_in: Optional[float] = None) -> List[float]:
    """
    Rettangolo (frazione figura) per una colorbar verticale a destra che
    copra l'altezza totale della griglia dati.
    """
    dh = data_h_in if data_h_in is not None else PANEL_DATA.HEIGHT_IN
    total_h = total_data_h_in if total_data_h_in is not None else (
        nrows * dh + (nrows - 1) * GAPS.H_IN
    )
    x0_in = fig_w_in - PANEL_MARGINS_IN.RIGHT - CBAR.WIDTH_IN
    y0_in = PANEL_MARGINS_IN.BOTTOM + 0.5 * ( (PANEL_DATA.HEIGHT_IN * nrows + (nrows - 1)*GAPS.H_IN) - total_h )
    return [x0_in/fig_w_in, y0_in/fig_h_in, CBAR.WIDTH_IN/fig_w_in, total_h/fig_h_in]
