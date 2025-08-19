"""
utils_style_v3.py
=================
Back-compat + modalità chiare per lo stile:
- auto_style(mode='auto'|'base'|'overlay'|'latex', base='paper.mplstyle', overlay='paper_latex_overlay.mplstyle')
  * base    -> solo paper.mplstyle
  * overlay -> paper + overlay LaTeX (senza forzare usetex)
  * latex   -> prova text.usetex=True sopra overlay; se fallisce, fallback a overlay
  * auto    -> 'latex' se disponibile, altrimenti 'overlay'

Altri helper (immutati nell'API):
- apply_panel_labels(fig, axes, labels=None)
- derive_linewidths(base=None)

Note: non crea side-effects globali; usa solo contesti (rc_context/style.context).
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Iterable, Sequence, Optional
import matplotlib as mpl
import matplotlib.pyplot as plt

# cfg opzionale per default "paper-ready"
try:
    import plot_cfg as cfg
except Exception:
    cfg = None

# ---------------- auto_style ----------------
def _try_latex():
    """Ritorna True se un draw con usetex=True non solleva errori, False altrimenti."""
    try:
        with mpl.rc_context({'text.usetex': True}):
            with mpl.style.context(['paper.mplstyle','paper_latex_overlay.mplstyle']):
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(1,1,1)
                ax.set_title(r"$E=mc^2$")  # forza il pass TeX
                fig.canvas.draw()
                plt.close(fig)
        return True
    except Exception:
        return False

@contextmanager
def auto_style(mode: str = 'auto', base: str = 'paper.mplstyle', overlay: str = 'paper_latex_overlay.mplstyle'):
    """
    Contesto stile con 4 modalità:
      - 'base'    : [base]
      - 'overlay' : [base, overlay]
      - 'latex'   : rc(text.usetex=True) + [base, overlay]; fallback a 'overlay' se LaTeX manca
      - 'auto'    : 'latex' se disponibile, altrimenti 'overlay'
    """
    if mode not in {'auto','base','overlay','latex'}:
        raise ValueError("mode must be one of: 'auto','base','overlay','latex'")
    styles_base = [base]
    styles_ov   = [base, overlay]

    if mode == 'base':
        with mpl.style.context(styles_base):
            yield
        return

    if mode == 'overlay':
        with mpl.style.context(styles_ov):
            yield
        return

    if mode == 'latex':
        if _try_latex():
            with mpl.rc_context({'text.usetex': True}):
                with mpl.style.context(styles_ov):
                    yield
        else:
            with mpl.style.context(styles_ov):
                yield
        return

    # mode == 'auto'
    if _try_latex():
        with mpl.rc_context({'text.usetex': True}):
            with mpl.style.context(styles_ov):
                yield
    else:
        with mpl.style.context(styles_ov):
            yield

# ---------------- panel labels ----------------
def apply_panel_labels(fig, axes: Sequence, labels: Optional[Sequence[str]] = None):
    """
    Applica etichette di pannello (a, b, c, ...).
    Usa config da plot_cfg_v6 se presente, altrimenti default robusti.
    """
    if labels is None:
        labels = [chr(ord('a') + i) for i in range(len(axes))]

    if cfg is not None and hasattr(cfg, "PANEL_LABEL"):
        pos = getattr(cfg.PANEL_LABEL, "POS_AX", (0.02, 0.98))
        ha = getattr(cfg.PANEL_LABEL, "HA", "left")
        va = getattr(cfg.PANEL_LABEL, "VA", "top")
        weight = getattr(cfg.PANEL_LABEL, "WEIGHT", "bold")
        size_rel = getattr(cfg.PANEL_LABEL, "SIZE_REL", 1.00)
        fmt = getattr(cfg.PANEL_LABEL, "FORMAT", "({letter})")
        # dimensiona il font rispetto a rcParams
        size_pt = mpl.rcParams.get('font.size', 10.0) * size_rel
    else:
        pos, ha, va, weight, size_pt, fmt = (0.02, 0.98), "left", "top", "bold", 10.0, "({letter})"

    for ax, lab in zip(axes, labels):
        ax.text(pos[0], pos[1], fmt.format(letter=lab), transform=ax.transAxes,
                ha=ha, va=va, weight=weight, fontsize=size_pt)

# ---------------- linewidths ----------------
def derive_linewidths(base: Optional[float] = None):
    """
    Ritorna un dict con spessori coerenti. Se plot_cfg_v6 è presente, scala da lì.
    """
    if base is None:
        base = 1.0
    if cfg is not None and hasattr(cfg, "LINES"):
        return {
            "thin":   base * getattr(cfg.LINES, "THIN", 0.60),
            "medium": base * getattr(cfg.LINES, "MEDIUM", 1.00),
            "strong": base * getattr(cfg.LINES, "STRONG", 1.15),
            "strong_min_pt": getattr(cfg.LINES, "STRONG_MIN_PT", 1.20)
        }
    else:
        return {"thin": 0.6*base, "medium": 1.0*base, "strong": 1.15*base, "strong_min_pt": 1.20}
