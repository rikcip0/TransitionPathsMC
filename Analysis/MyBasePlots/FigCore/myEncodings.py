
"""
myEncodings_v4.py — GOLD (drop-in superset)
-------------------------------------------
- Mantiene API esistenti: get_palette, get_marker, get_linestyle, get_encoding(idx, use_color=True, palette="okabe_ito"),
  color_for_category(cat, palette="okabe_ito"), lighten/darken/alpha, norms helper.
- Aggiunge: palette OI_noYellow / OI_yellowLast, lighten_rgba/darken_rgba (clamp, alpha-preserving),
  color hash più robusto (multi-byte + salt), check_grayscale_contrast, safe_norm, style_for_role,
  register_default_cmaps (seq/div percettive). Nessun side-effect su rcParams.

Nota sul giallo:
- "okabe_ito" (strict): include il giallo standard #F0E442 (CUD).
- "okabe_ito_no_yellow": escluso il giallo → consigliato per linee su sfondo bianco / B/N.
- "okabe_ito_yellow_last": giallo spostato in coda (prima del nero).

"""

from __future__ import annotations
import itertools, math, colorsys, hashlib
from typing import Iterable, Sequence, Tuple, Dict, Optional, List, Any
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler as _cycler
# ======================= PALETTE =======================

# Okabe–Ito strict (8 colori + nero già incluso come ultimo)
_OI_STRICT = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow (standard)
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

# Varianti consigliate
_OI_NO_YELLOW = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"
]
_OI_YELLOW_LAST = [
    "#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#000000"
]

_MUTED = ["#4878CF","#6ACC65","#D65F5F","#B47CC7","#C4AD66","#77BEDB"]
_GRAYS = ["#000000","#333333","#666666","#999999","#CCCCCC","#E0E0E0"]
_GRAYSCALE = ["#000000","#3A3A3A","#6B6B6B","#9C9C9C","#CDCDCD"]

_PALETTES: Dict[str, List[str]] = {
    "okabe_ito"            : _OI_STRICT,
    "okabe-ito"            : _OI_STRICT,
    "oi"                   : _OI_STRICT,
    "okabe_ito_no_yellow"  : _OI_NO_YELLOW,
    "oi_no_yellow"         : _OI_NO_YELLOW,
    "okabe_ito_yellow_last": _OI_YELLOW_LAST,
    "oi_yellow_last"       : _OI_YELLOW_LAST,
    "muted"                : _MUTED,
    "grays"                : _GRAYS,
    "grayscale"            : _GRAYSCALE,
    "gray"                 : _GRAYSCALE,
    "tab10"                : [mpl.colors.to_hex(c) for c in plt.get_cmap("tab10").colors],
}

def get_palette(palette: str|Iterable[str]="okabe_ito") -> List[str]:
    """Ritorna una lista di colori hex. Accetta nome o lista custom."""
    if isinstance(palette, str):
        key = palette.lower()
        return list(_PALETTES.get(key, _PALETTES["okabe_ito"]))
    # copia della lista
    return list(palette)

def apply_cycle(ax, palette: str="okabe_ito_no_yellow") -> None:
    """Setta il prop_cycle dell'Axes con la palette scelta (colori solo)."""
    colors = get_palette(palette)
    ax.set_prop_cycle(_cycler(color=colors))

# ======================= MARKER / LINESTYLE =======================

_MARKERS = ["o","s","^","D","v","P","X","<",">"]
_LINESTYLES = ["-","--","-.",":"]

def get_marker(idx: int) -> str:
    return _MARKERS[idx % len(_MARKERS)]

def get_linestyle(idx: int) -> str:
    return _LINESTYLES[(idx // len(_MARKERS)) % len(_LINESTYLES)]

# ======================= COMBINED ENCODING =======================

def get_encoding(idx: int, use_color: bool=True, palette: str|Iterable[str]="okabe_ito", **kwargs) -> Tuple[str,str,str]:
    """
    Ritorna (color, marker, linestyle) per la serie i-esima.
    - use_color=False => BN (grayscale + marker/linestyle differenzianti)
    - palette: nome o lista di colori
    kwargs accetta eventuali arg legacy (es. grayscale) senza rompere la firma.
    """
    if use_color and not kwargs.get("grayscale", False):
        colors = get_palette(palette)
        c = colors[idx % len(colors)]
    else:
        c = _GRAYSCALE[idx % len(_GRAYSCALE)]
    m = get_marker(idx)
    ls = get_linestyle(idx)
    return (c, m, ls)

# ======================= CATEGORY MAPPING (color) =======================

def _hash_to_index(cat: str, modulo: int, salt: Optional[str]=None, nbytes: int=2) -> int:
    """Usa i primi nbytes dell'hash per mappare la categoria in [0, modulo)."""
    h = hashlib.sha1((str(cat) + (salt or "")).encode("utf-8")).digest()
    val = 0
    for i in range(nbytes):
        val = (val << 8) | h[i]
    return val % modulo

def color_for_category(cat: str, palette: str|Iterable[str]="okabe_ito", *, salt: Optional[str]=None, nbytes: int=2) -> str:
    colors = get_palette(palette)
    idx = _hash_to_index(cat, len(colors), salt=salt, nbytes=max(1, min(4, nbytes)))
    return colors[idx]

def apply_color_linestyle_cycle(ax, palette="okabe_ito_no_yellow",
                                linestyles=('-', '--', '-.', ':')):
    colors = get_palette(palette)
    n = min(len(colors), len(linestyles))
    ax.set_prop_cycle(_cycler(color=colors[:n]) + _cycler(linestyle=list(linestyles)[:n]))

# ======================= COLOR UTILS =======================

def _to_rgba(c) -> Tuple[float,float,float,float]:
    r,g,b,a = mpl.colors.to_rgba(c)
    return float(r), float(g), float(b), float(a)

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def lighten_rgba(color, factor: float=0.2):
    """Schiarisce preservando alpha. factor>0 avvicina a 1.0."""
    r,g,b,a = _to_rgba(color)
    r = _clamp01(r + (1-r)*factor)
    g = _clamp01(g + (1-g)*factor)
    b = _clamp01(b + (1-b)*factor)
    return (r,g,b,a)

def darken_rgba(color, factor: float=0.2):
    """Scurisce preservando alpha. factor>0 avvicina a 0.0."""
    r,g,b,a = _to_rgba(color)
    r = _clamp01(r*(1-factor))
    g = _clamp01(g*(1-factor))
    b = _clamp01(b*(1-factor))
    return (r,g,b,a)

def lighten(color, factor: float=0.2):
    r,g,b,a = lighten_rgba(color, factor)
    return mpl.colors.to_hex((r,g,b,a))

def darken(color, factor: float=0.2):
    r,g,b,a = darken_rgba(color, factor)
    return mpl.colors.to_hex((r,g,b,a))

def alpha(color, a: float):
    r,g,b,_ = _to_rgba(color)
    return (r,g,b,_clamp01(a))


# ======================= CONTRAST CHECK (B/N) =======================

def _luminance_rgba(color) -> float:
    r,g,b,_ = _to_rgba(color)
    # sRGB luminance (approx)
    return 0.2126*r + 0.7152*g + 0.0722*b

def check_grayscale_contrast(colors: Sequence[str], min_delta: float=0.08) -> Dict[str, Any]:
    """
    Ritorna un report con luminanze e coppie potenzialmente problematiche in B/N.
    min_delta ~ 0.08 è una soglia prudente per linee sottili.
    """
    cols = list(colors)
    L = [_luminance_rgba(c) for c in cols]
    issues = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if abs(L[i]-L[j]) < min_delta:
                issues.append((i,j,cols[i],cols[j],abs(L[i]-L[j])))
    return {"colors": cols, "luminance": L, "too_close_pairs": issues, "min_delta": min_delta}

# ======================= NORMS (safe) =======================

def safe_norm(mode: str, data: np.ndarray, **kw):
    """
    Restituisce una Norm Matplotlib robusta:
    - 'linear'/'none' => Normalize
    - 'log'           => LogNorm se data>0, altrimenti Normalize
    - 'symlog'        => SymLogNorm (kw: linthresh=...)
    - 'centered'      => TwoSlopeNorm centered a 0
    """
    mode = (mode or "linear").lower()
    data = np.asarray(data)
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)
    if mode in {"linear","none"}:
        return mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=kw.get("clip", False))
    if mode == "log":
        if vmin <= 0:
            return mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=kw.get("clip", False))
        return mpl.colors.LogNorm(vmin=max(vmin, kw.get("vmin", vmin)), vmax=kw.get("vmax", vmax), clip=kw.get("clip", False))
    if mode == "symlog":
        linthresh = kw.get("linthresh", max(1e-6, 0.01*(vmax - vmin)))
        return mpl.colors.SymLogNorm(linthresh=linthresh, vmin=kw.get("vmin", vmin), vmax=kw.get("vmax", vmax))
    if mode == "centered":
        vcenter = kw.get("vcenter", 0.0)
        return mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    # default fallback
    return mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=kw.get("clip", False))

# ======================= ROLE-AWARE STYLE =======================

def style_for_role(role: str="main", base: Tuple[str,str,str]|None=None):
    """
    Ritorna (color, marker, linestyle, alpha, lw_scale) a partire da un encoding base.
    role in {'main','fit','diagnostic','unused'}.
    - fit: linestyle='--', lw_scale=1.15, alpha=1.0
    - diagnostic: alpha=0.5, lw_scale=0.9
    - unused: alpha=0.35, lw_scale=0.9
    """
    if base is None:
        base = ("#000000","o","-")
    c,m,ls = base
    role = (role or "main").lower()
    alpha_v = 1.0; lw = 1.0
    if role == "fit":
        ls = "--"; lw = 1.15; alpha_v = 1.0
    elif role == "diagnostic":
        alpha_v = 0.5; lw = 0.9
    elif role == "unused":
        alpha_v = 0.35; lw = 0.9
    return (c,m,ls,alpha_v,lw)

# ======================= CMAPS REGISTRATION =======================

def register_default_cmaps(prefix: str="enc_"):
    """
    Registra 2–3 colormap percettivamente sensate (seq/div) derivate da palette sicure.
    """

    # diverging blu ↔ arancio
    div = mpl.colors.LinearSegmentedColormap.from_list(prefix+"div_blue_orange", ["#0072B2","#F0F0F0","#E69F00"])
    mpl.colormaps.register(div, name=prefix+"div_blue_orange", override_builtin=True)
    # diverging blu ↔ rosso
    div2 = mpl.colors.LinearSegmentedColormap.from_list(prefix+"div_blue_red", ["#0072B2","#F0F0F0","#D55E00"])
    mpl.colormaps.register(div2, name=prefix+"div_blue_red", override_builtin=True)
    
def register_gold_standard_cmaps(prefix: str = "my_"):
    """
    Registra colormap "gold standard" per la visualizzazione scientifica.
    - Cividis: sequenziale, percettivamente uniforme, massima robustezza per daltonismo.
    - Divergente Blu-Giallo con centro scuro: per dati dove il valore centrale è
      importante quanto gli estremi.
    """
    # 1. Sequenziale: Cividis (la scelta più robusta per pubblicazioni)
    try:
        cividis_map = mpl.colormaps['cividis']
        mpl.colormaps.register(cividis_map, name=prefix + "seq_cividis", override_builtin=True)
    except KeyError:
        warnings.warn("Colormap 'cividis' non trovata in Matplotlib. Impossibile registrarla.")

    # 2. Divergente con centro scuro (dai colori Okabe-Ito)
    #    Blu (#0072B2) -> Nero (#000000) -> Giallo (#F0E442)
    div_dark_center = mpl.colors.LinearSegmentedColormap.from_list(
        prefix + "div_BuYr_dark",
        ["#0072B2", "#000000", "#F0E442"]
    )
    mpl.colormaps.register(div_dark_center, name=prefix + "div_BuYr_dark", override_builtin=True)

def register_all_cmaps(prefix: str = "my_"):
    """
    Chiama tutte le funzioni di registrazione per avere a disposizione l'intero set di colormap.
    Questa è la funzione consigliata da chiamare all'inizio di uno script di plotting.
    """
    register_default_cmaps(prefix)
    register_gold_standard_cmaps(prefix)

# ======================= PREVIEW UTILS (facoltative) =======================

def preview_palette(palette="okabe_ito"):
    colors = get_palette(palette)
    fig,ax = plt.subplots(figsize=(max(6, len(colors)*0.6), 1.0))
    for i,c in enumerate(colors):
        ax.add_patch(mpl.patches.Rectangle((i,0),1,1,color=c))
    ax.set_xlim(0,len(colors)); ax.set_ylim(0,1)
    ax.axis("off"); plt.show()

def preview_encodings(n=8, use_color=True, palette="okabe_ito"):
    fig,ax = plt.subplots(figsize=(6, max(2, n*0.4)))
    for i in range(n):
        c,m,ls = get_encoding(i, use_color=use_color, palette=palette)
        ax.plot([0,1],[i,i], ls=ls, color=c, marker=m, label=f"series {i}")
    ax.set_yticks([]); ax.set_xlim(0,1)
    ax.legend(ncol=2, fontsize=8, frameon=True)
    plt.show()
