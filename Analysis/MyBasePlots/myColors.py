"""
myColors_v3.py
==============
Estensioni *gold* e stateless per palette/colormap in contesto scientifico.

Novità vs v2
------------
- Palette aggiuntive e "soft" (versione schiarita per aree/fill).
- Costruttori rapidi:
    • make_monochrome_ramp(base, n)      -> cmap sequenziale partendo da un colore base
    • diverging_from(base_neg, base_pos) -> cmap divergente (chiara al centro)
- Norm helpers (RobustNorm):
    • vlim_symmetric(data, method='max'|'percentile', p=99) -> limiti simmetrici
    • norm_centered(data, center=0, method='max'|'percentile', p=99) -> colors.CenteredNorm
    • norm_linear / norm_log / norm_symlog -> normalizzazioni pronte
- Categoriale deterministico:
    • colors_for_categories(keys, palette='okabe_ito') -> mappa categoria->colore stabile

Resta invariato rispetto a v2:
- get_palette / get_cycler / apply_cycle / palette_context
- get_cmap (reverse, discrete), register_listed_cmap / register_linear_cmap
- lighten, darken, with_alpha, sample_cmap

Tutto senza effetti globali a rcParams.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Optional, Iterable, Tuple, Callable, Dict
from contextlib import contextmanager
import hashlib
import numpy as np

import matplotlib as mpl
from matplotlib import cycler
from matplotlib.colors import (ListedColormap, LinearSegmentedColormap, to_rgb, to_hex,
                               Normalize, LogNorm, SymLogNorm, CenteredNorm)

# =========================
# Palette qualitative (liste di HEX) — color-blind safe
# =========================

OKABE_ITO = [
    "#0072B2", "#D55E00", "#CC79A7", "#E69F00",
    "#56B4E9", "#009E73", "#F0E442", "#000000"
]
OKABE_ITO_NOY = ["#0072B2", "#D55E00", "#CC79A7", "#56B4E9", "#009E73", "#000000"]

def _lighten_many(cols, amt=0.35):  # utile per versioni "soft"
    out = []
    for c in cols:
        r,g,b = mpl.colors.to_rgb(c)
        r = r + (1.0-r)*amt
        g = g + (1.0-g)*amt
        b = b + (1.0-b)*amt
        out.append(mpl.colors.to_hex((r,g,b)))
    return out

OKABE_ITO_SOFT = _lighten_many(OKABE_ITO_NOY, 0.40)  # no-yellow + schiarita per aree/fill

MUTED = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]
GRAYS = ["#000000", "#444444", "#777777", "#AAAAAA", "#CCCCCC"]

_TAB10 = list(mpl.rcParamsDefault.get('axes.prop_cycle', cycler(color=[])).by_key().get('color', [])) or OKABE_ITO_NOY

PALETTES: Dict[str, List[str]] = {
    "okabe_ito": OKABE_ITO,
    "okabe_ito_noy": OKABE_ITO_NOY,
    "okabe_ito_soft": OKABE_ITO_SOFT,
    "cb_safe": OKABE_ITO_NOY,    # alias
    "muted": MUTED,
    "grays": GRAYS,
    "tab10": _TAB10,
}

def get_palette(name: str, n: Optional[int] = None) -> List[str]:
    if name not in PALETTES:
        raise KeyError(f"Palette '{name}' non registrata")
    base = list(PALETTES[name])
    if n is None or len(base) >= n:
        return base[:n] if n else base
    out = []
    k = len(base)
    for i in range(n):
        out.append(base[i % k])
    return out

def get_cycler(name: str, n: Optional[int] = None) -> cycler:
    return cycler('color', get_palette(name, n))

def apply_cycle(ax, name: str, n: Optional[int] = None) -> None:
    ax.set_prop_cycle(get_cycler(name, n))

@contextmanager
def palette_context(name: str, n: Optional[int] = None):
    cyc = {'axes.prop_cycle': get_cycler(name, n)}
    with mpl.rc_context(rc=cyc):
        yield

# =========================
# Colormap: registrazione e accesso
# =========================

def _register_cmap_safe(cmap, name: str, overwrite: bool = False) -> None:
    try:
        mpl.colormaps.register(cmap, name=name, override=overwrite)
    except Exception:
        import matplotlib.cm as cm
        if overwrite:
            try:
                cm.unregister_cmap(name)
            except Exception:
                pass
        cm.register_cmap(name=name, cmap=cmap)

def register_listed_cmap(name: str, colors: Sequence[str], overwrite: bool = False) -> ListedColormap:
    cmap = ListedColormap(colors, name=name)
    _register_cmap_safe(cmap, name, overwrite=overwrite)
    return cmap

def register_linear_cmap(name: str, colors: Sequence[str], overwrite: bool = False) -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(name, colors)
    _register_cmap_safe(cmap, name, overwrite=overwrite)
    return cmap

def get_cmap(name: str, *, reverse: bool = False, discrete: Optional[int] = None):
    base_name = name + ("_r" if reverse and not name.endswith("_r") else "")
    if base_name not in mpl.colormaps:
        raise KeyError(f"Colormap '{base_name}' non registrata")
    base = mpl.colormaps[base_name]
    if discrete is None:
        return base
    if discrete <= 1:
        raise ValueError("discrete deve essere >= 2")
    vals = np.linspace(0, 1, int(discrete))
    colors = [base(v) for v in vals]
    return ListedColormap(colors, name=f"{base_name}_{discrete}")

def sample_cmap(name: str, stops: Sequence[float]):
    if name not in mpl.colormaps:
        raise KeyError(f"Colormap '{name}' non registrata")
    cm = mpl.colormaps[name]
    return [cm(float(s)) for s in stops]

# =========================
# Costruttori rapidi
# =========================

def lighten(color: str, amount: float) -> str:
    r, g, b = mpl.colors.to_rgb(color)
    r = r + (1.0 - r)*amount
    g = g + (1.0 - g)*amount
    b = b + (1.0 - b)*amount
    return mpl.colors.to_hex((r, g, b))

def darken(color: str, amount: float) -> str:
    r, g, b = mpl.colors.to_rgb(color)
    r = r*(1.0 - amount)
    g = g*(1.0 - amount)
    b = b*(1.0 - amount)
    return mpl.colors.to_hex((r, g, b))

def with_alpha(color: str, alpha: float):
    r, g, b = mpl.colors.to_rgb(color)
    return (r, g, b, float(alpha))

def make_monochrome_ramp(base: str, n: int = 256, *, low: float = 0.05, high: float = 0.95):
    """
    Crea una cmap sequenziale dal chiaro (low) -> base -> scuro (high).
    low/high controllano l'intensità di schiarimento/scurimento (0..1).
    """
    base_rgb = np.array(mpl.colors.to_rgb(base))
    white = np.ones(3)
    black = np.zeros(3)
    mid = base_rgb
    c_lo = white*(1-low) + base_rgb*low
    c_hi = black*(high) + base_rgb*(1-high)
    colors = np.vstack([c_lo, mid, c_hi])
    return LinearSegmentedColormap.from_list(f"mono_{base.strip('#')}", colors, N=n)

def diverging_from(base_neg: str, base_pos: str, n: int = 256, *, mid: Tuple[float,float,float] = (1,1,1)):
    """
    Crea una cmap divergente simmetrica, chiara al centro.
    """
    c_neg = np.array(mpl.colors.to_rgb(base_neg))
    c_pos = np.array(mpl.colors.to_rgb(base_pos))
    c_mid = np.array(mid)
    colors = np.vstack([c_neg, c_mid, c_pos])
    return LinearSegmentedColormap.from_list(f"div_{base_neg.strip('#')}_{base_pos.strip('#')}", colors, N=n)

# =========================
# Norm (robuste per dati scientifici)
# =========================

def vlim_symmetric(data, *, method: str = "max", p: float = 99.0):
    """
    Ritorna (vmin, vmax) con |vmin|=|vmax|.
    - method='max'        -> vmax = max(|data|)
    - method='percentile' -> vmax = percentile(|data|, p)
    """
    arr = np.asarray(data)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0: return (-1.0, 1.0)
    if method == "max":
        vmax = float(np.max(np.abs(arr)))
    elif method == "percentile":
        vmax = float(np.percentile(np.abs(arr), p))
    else:
        raise ValueError("method must be 'max'|'percentile'")
    return (-vmax, vmax) if vmax > 0 else (-1.0, 1.0)

def norm_centered(data, *, center: float = 0.0, method: str = "max", p: float = 99.0):
    vmin, vmax = vlim_symmetric(np.asarray(data) - center, method=method, p=p)
    halfrange = max(abs(vmin), abs(vmax))
    return CenteredNorm(vcenter=center, halfrange=halfrange if halfrange > 0 else 1.0)

def norm_linear(vmin=None, vmax=None, clip=False):
    return Normalize(vmin=vmin, vmax=vmax, clip=clip)

def norm_log(vmin=None, vmax=None, clip=False):
    return LogNorm(vmin=max(vmin, np.finfo(float).eps) if vmin is not None else None, vmax=vmax, clip=clip)

def norm_symlog(vmin=None, vmax=None, linthresh=1e-6, linscale=1.0, base=10.0, clip=False):
    return SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin, vmax=vmax, base=base, clip=clip)

# =========================
# Categoriale deterministico
# =========================

def _stable_hash(s: str) -> int:
    # sha1 stabile su stringa → int
    h = hashlib.sha1(s.encode('utf-8')).hexdigest()
    return int(h, 16)

def colors_for_categories(keys: Sequence, palette: str = "okabe_ito") -> Dict:
    pal = get_palette(palette)
    k = len(pal)
    out = {}
    for key in keys:
        idx = _stable_hash(str(key)) % k
        out[key] = pal[idx]
    return out
