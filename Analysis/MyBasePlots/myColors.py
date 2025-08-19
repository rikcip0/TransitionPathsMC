# plot_colors.py
"""
Registro centralizzato di palette (liste di colori) e colormap per i plot.
- Non toccare rcParams globali qui: fornisci funzioni/contesti.
- Usa in codice: apply_cycle(ax, 'okabe_ito'), get_cmap('viridis'), ecc.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Optional
import matplotlib as mpl
from matplotlib import cycler
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# =========================
# Palette discrete (liste di colori HEX)
# =========================

OKABE_ITO = [
    "#0072B2", "#D55E00", "#CC79A7", "#E69F00",
    "#56B4E9", "#009E73", "#F0E442", "#000000"
]

# Variante senza giallo (utile per linee sottili su bianco):
OKABE_ITO_NOY = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#009E73", "#000000"]

# Esempi di palette aggiuntive
MUTED = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]  # tipo 'muted'
GRAYS = ["#000000", "#444444", "#777777", "#AAAAAA", "#CCCCCC"]

PALETTES = {
    "okabe_ito": OKABE_ITO,
    "okabe_ito_noy": OKABE_ITO_NOY,
    "muted": MUTED,
    "grays": GRAYS,
    # alias utili
    "cb_safe": OKABE_ITO_NOY,
    "tab10": list(mpl.rcParamsDefault.get('axes.prop_cycle', cycler(color=[])).by_key().get('color', [])) or OKABE_ITO_NOY,
}

def get_palette(name: str, n: Optional[int] = None) -> List[str]:
    """Ritorna una lista di colori. Se n è dato e la palette ha < n colori, cicla."""
    if name not in PALETTES:
        raise KeyError(f"Palette '{name}' non registrata")
    pal = PALETTES[name]
    if n is None or len(pal) >= n:
        return pal[:n] if n else pal
    # cicla i colori per raggiungere n
    out = []
    k = len(pal)
    for i in range(n):
        out.append(pal[i % k])
    return out

def get_cycler(name: str, n: Optional[int] = None) -> cycler:
    return cycler('color', get_palette(name, n))

def apply_cycle(ax, name: str, n: Optional[int] = None) -> None:
    """Imposta il ciclo colori su un Axes."""
    ax.set_prop_cycle(get_cycler(name, n))

# Context manager per applicare una palette a livello rcParams, localmente
from contextlib import contextmanager
@contextmanager
def palette_context(name: str, n: Optional[int] = None):
    cyc = {'axes.prop_cycle': get_cycler(name, n)}
    with mpl.rc_context(rc=cyc):
        yield

# =========================
# Colormap continue / discrete
# =========================

# Registriamo qui cmaps personalizzate se servono
def register_listed_cmap(name: str, colors: Sequence[str], overwrite: bool = False) -> ListedColormap:
    cmap = ListedColormap(colors, name=name)
    _register_cmap(cmap, name, overwrite=overwrite)
    return cmap

def register_linear_cmap(name: str, colors: Sequence[str], overwrite: bool = False) -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(name, colors)
    _register_cmap(cmap, name, overwrite=overwrite)
    return cmap

def _register_cmap(cmap, name: str, overwrite: bool = False) -> None:
    try:
        mpl.colormaps.register(cmap, name=name, override=overwrite)
    except ValueError:
        if overwrite:
            # rimuovi e re-registra
            mpl.colormaps.unregister(name)
            mpl.colormaps.register(cmap, name=name, override=True)

def get_cmap(name: str, reverse: bool = False, discrete: Optional[int] = None):
    """
    Ritorna una colormap Matplotlib.
    - reverse: usa la versione _r
    - discrete: se dato, restituisce una ListedColormap con N campioni discreti
    """
    base_name = name + ("_r" if reverse and not name.endswith("_r") else "")
    if base_name not in mpl.colormaps:
        raise KeyError(f"Colormap '{base_name}' non registrata")
    base = mpl.colormaps[base_name]
    if discrete is None:
        return base
    # campiona N colori dalla colormap continua
    colors = [base(i) for i in [i/(discrete-1) for i in range(discrete)]]
    return ListedColormap(colors, name=f"{base_name}_{discrete}")

# Esempi di registrazione all'import (opzionali)
# register_linear_cmap("blue_orange", ["#313695","#74add1","#e0f3f8","#fdae61","#a50026"])
# register_listed_cmap("grays5", GRAYS[:5])
