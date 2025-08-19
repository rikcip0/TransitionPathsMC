# utils_plot.py
from pathlib import Path
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------- Export ----------
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def exportFigure(fig, basename: str, outdir, to_pdf: bool = False, dpi: int = 300):
    """
    Salva una figura in PNG (default) o PDF (to_pdf=True).
    - bbox deterministico (no tight di default).
    - font embedding (pdf/ps fonttype = 42) già in myStyle/myLatexStyle.
    """
    outdir = Path(outdir)
    _ensure_dir(outdir)
    ext = "pdf" if to_pdf else "png"
    fname = outdir / f"{basename}.{ext}"
    kwargs = {"dpi": dpi} if not to_pdf else {"dpi": dpi}  # lascia dpi anche su PDF per raster secondari
    fig.savefig(fname, **kwargs)
    return str(fname)

def exportAllOpenFigures(outdir, to_pdf: bool = False, dpi: int = 300):
    paths = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        name = fig.get_label() or f"Figure_{num}"
        paths.append(exportFigure(fig, name, outdir, to_pdf=to_pdf, dpi=dpi))
    plt.close("all")
    return paths

# ---------- Legend ----------
def place_legend_outside(ax, anchor=(0.965, 0.98), loc="upper right"):
    """Sposta la legenda fuori dall'axes ma dentro la figura, in posizione coerente."""
    leg = ax.get_legend()
    if leg is None:
        return
    leg.set_bbox_to_anchor(anchor, transform=ax.figure.transFigure)
    leg._loc = loc  # forza loc interno all'area di anchor
    # nessun frame: demandato allo stile (legend.frameon=False)

# ---------- Ticks ----------
def standardize_ticks(ax, xbins=5, ybins=5, prune="both", minor=False):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xbins, prune=prune))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=ybins, prune=prune))
    ax.minorticks_on() if minor else ax.minorticks_off()
