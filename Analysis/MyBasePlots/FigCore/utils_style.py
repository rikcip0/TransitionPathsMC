
"""
utils_style
---------------------------------------------
- Context di stile componibile: base + overlay (+ LaTeX opzionale) senza tight/constrained impliciti.
- Cache LaTeX (probe una sola volta), override via env MYBP_DISABLE_LATEX=1.
- Minimi editoriali "non-invasivi": enforce solo se sotto soglia (font, linewidth, ticklength).
- API compat: auto_style, apply_panel_labels, derive_linewidths, apply_ticks_from_cfg, style_debug_report.
- Nuove utility: get_style_stack(...), ensure_min_rc(...).

Dipendenze: matplotlib, numpy (opzionale per flatten axes).
"""

from __future__ import annotations
from contextlib import contextmanager
from typing import Optional, Sequence, Tuple, Iterable, List
import os
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from .latex_env import require_tex as _require_tex

try:
    import numpy as _np
except Exception:  # numpy non è obbligatorio
    _np = None

# plot_cfg opzionale
try:
    from . import plot_cfg as cfg
except Exception:
    cfg = None

# =============================================================================
# Path resolution helpers for .mplstyle
# =============================================================================

def _resolve_style(s):
    if isinstance(s, (list, tuple)):
        return [_resolve_style(x) for x in s]
    p = Path(str(s))
    if p.is_file():
        return str(p)
    # prova relativo alla cartella del package
    pkg = Path(__file__).resolve().parent / "styles"
    cand = pkg / p.name
    return str(cand if cand.is_file() else p)

def _candidate_paths(name: str) -> Sequence[str]:
    cand: List[str] = []
    if os.path.isabs(name) and os.path.exists(name):
        cand.append(name); return cand
    cand.append(os.path.join(os.getcwd(), name))              # cwd
    here = os.path.dirname(__file__)
    cand.append(os.path.join(here, name))                     # alongside module
    cand.append(name)                                         # let mpl search path
    return cand

def _resolve_style(name: str) -> str:
    for p in _candidate_paths(name):
        if os.path.exists(p):
            return p
    return name  # allow Matplotlib search; may fail later inside style.context

def style_paths(base: str='styles/paper_base.mplstyle',
                overlay: str='styles/overlay_latex.mplstyle') -> Tuple[str, str]:
    """Risolvi i percorsi dei file di stile (base, overlay)."""
    return _resolve_style(base), _resolve_style(overlay)

def _overlay_for_profile(default_overlay: str, profile: Optional[str]) -> str:
    """
    Se profile ('aps','acm','thesis',...) è fornito, prova 'styles/paper_<profile>_overlay.mplstyle'.
    Se non esiste, ritorna default_overlay e avvisa.
    """
    if not profile:
        return default_overlay
    cand = f"styles/paper_{profile}_overlay.mplstyle"
    resolved = _resolve_style(cand)
    if os.path.exists(resolved):
        return cand
    warnings.warn(f"[utils_style] overlay per profilo '{profile}' non trovato "
                  f"(atteso: {cand}); uso overlay di default: {default_overlay}.",
                  RuntimeWarning)
    return default_overlay

# =============================================================================
# LaTeX probing (cached)
# =============================================================================

_LATEX_AVAILABLE: Optional[bool] = None

def _try_latex() -> bool:
    """Probe leggero: usetex=True + draw di 1 figura. True se ok, False se fail. Cache globale."""
    global _LATEX_AVAILABLE
    if os.environ.get("MYBP_DISABLE_LATEX", "0") == "1":
        _LATEX_AVAILABLE = False
        return False
    if _LATEX_AVAILABLE is not None:
        return _LATEX_AVAILABLE
    try:
        with mpl.rc_context({'text.usetex': True, 'figure.autolayout': False}):
            b, o = style_paths()
            with mpl.style.context([b, o]):
                fig = plt.figure(figsize=(1,1))
                ax = fig.add_subplot(1,1,1)
                ax.set_title(r"$E=mc^2$")
                fig.canvas.draw()
                plt.close(fig)
        _LATEX_AVAILABLE = True
    except Exception:
        _LATEX_AVAILABLE = False
    return _LATEX_AVAILABLE

# =============================================================================
# RC minima editoriali (enforcement non-invasivo)
# =============================================================================

def ensure_min_rc():
    """
    Impone minimi editoriali *solo se* i valori correnti sono sotto soglia.
    Soglie conservative (post-ridimensionamento):
      - font.size >= 6 pt
      - axes.linewidth >= 0.5 pt
      - lines.linewidth >= 0.6 pt
      - xtick.major.size, ytick.major.size >= 2.0 pt
    Non tocca nulla se già sopra.
    """
    mins = {
        ("font.size", 6.0),
        ("axes.linewidth", 0.5),
        ("lines.linewidth", 0.6),
        ("xtick.major.size", 2.0),
        ("ytick.major.size", 2.0),
    }
    for key, val in mins:
        cur = mpl.rcParams.get(key, None)
        try:
            if cur is None or float(cur) < val:
                mpl.rcParams[key] = val
        except Exception:
            # se non è castabile, meglio non forzare
            pass

# =============================================================================
# Main style context
# =============================================================================

def _texsystem():
    # pdflatex (default) | xelatex | lualatex (puoi cambiarlo via env)
    return os.getenv("MYBP_TEXSYSTEM", "pdflatex")

def _latex_tools_available():
    # per PGF basta il motore LaTeX scelto
    return bool(shutil.which(_texsystem()))

@contextmanager
def _switch_backend(name: str):
    prev = mpl.get_backend()
    try:
        plt.switch_backend(name)
        yield
    finally:
        try:
            plt.switch_backend(prev)
        except Exception:
            pass

def get_style_stack(mode: str='auto',
                    base: str='styles/paper_base.mplstyle',
                    overlay: str='styles/overlay_latex.mplstyle',
                    profile: Optional[str]=None) -> Sequence[str]:
    """Restituisce la lista degli style file che verrebbero applicati (risolti)."""
    ov = _overlay_for_profile(overlay, profile)
    b, o = style_paths(base, ov)
    if mode == 'base':
        return [b]
    return [b, o]

def _stack(base: str, overlay: str | Sequence[str]):
    return [base, overlay] if isinstance(overlay, str) else [base, *overlay]

@contextmanager
def auto_style(mode: str='latex',
               base: str='styles/paper_base.mplstyle',
               overlay: str='styles/overlay_latex.mplstyle',
               profile: str | None = None):
    """
    overlay     : base+overlay, TeX OFF
    latex_text  : text.usetex=True (TeX per testo)  — fail-fast se TeX/pacchetti mancanti
    latex       : PGF heavy (figura tramite LaTeX) — fail-fast; backend 'pgf'
    auto        : prova 'latex' (PGF), altrimenti ricade a overlay
    """
    if profile and isinstance(overlay, str) and "{profile}" in overlay:
        overlay = overlay.format(profile=profile)

    rc_old = mpl.rcParams.copy()
    try:
        # baseline minima e pulita (no tight implicito; font editabili)
        mpl.rcParams["savefig.bbox"] = None
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"]  = 42
        mpl.rcParams["svg.fonttype"] = "none"

        base = _resolve_style(base)
        overlay = _resolve_style(overlay)
        stack = [base, overlay] if isinstance(overlay, str) else [base, *overlay]
        
        stack = _stack(base, overlay)

        if mode == "overlay":
            with mpl.style.context(stack):
                mpl.rcParams["text.usetex"] = False
                yield
            return

        if mode == "latex_text":
            rc = _require_tex(backend="usetex")
            with mpl.style.context(stack):
                mpl.rcParams.update(rc)
                # blindatura (ridondante, ma difensiva)
                mpl.rcParams["text.usetex"] = True
                yield
            return

        if mode in {"latex", "auto"}:
            try:
                rc = _require_tex(backend="pgf")  # può sollevare
            except Exception:
                if mode == "auto":
                    with mpl.style.context(stack):
                        mpl.rcParams["text.usetex"] = False
                        yield
                    return
                raise
            # Heavy PGF: serve backend 'pgf' per coerenza completa
            with _switch_backend("pgf"):
                with mpl.style.context(stack):
                    mpl.rcParams.update(rc)
                    yield
            return

        # fallback prudente
        with mpl.style.context(stack):
            mpl.rcParams["text.usetex"] = False
            yield
    finally:
        mpl.rcParams.update(rc_old)
        
@contextmanager
def style_profile(profile: str = "paper", mode: str = "auto"):
    """Alias compat: usa auto_style con profilo e modalità."""
    with auto_style(profile=profile, mode=mode):
        yield

# =============================================================================
# Utilities
# =============================================================================

def _flatten_axes(axes) -> list:
    """Rende piatta una collezione di axes (list/tuple/np.ndarray) in lista semplice."""
    if axes is None:
        return []
    if _np is not None and isinstance(axes, _np.ndarray):
        return [a for a in axes.ravel().tolist() if a is not None]
    if isinstance(axes, (list, tuple)):
        out = []
        for a in axes:
            if _np is not None and isinstance(a, _np.ndarray):
                out.extend([x for x in a.ravel().tolist() if x is not None])
            else:
                out.append(a)
        return out
    return [axes]

def apply_panel_labels(fig, axes, labels=None):
    """
    Applica etichette (a,b,c,...) sugli axes.
    Se cfg.PANEL_LABEL esiste, usa la policy definita lì.
    Gold: font minimo 8 pt.
    """
    ax_list = _flatten_axes(axes)
    if not ax_list:
        return
    if labels is None:
        labels = [chr(ord('a') + i) for i in range(len(ax_list))]
    # Se labels meno di axes, cicla; se più, ignora l'eccesso
    for i, ax in enumerate(ax_list):
        lab = labels[i % len(labels)]
        if cfg is not None and hasattr(cfg, "PANEL_LABEL"):
            pos = getattr(cfg.PANEL_LABEL, "POS_AX", (0.02, 0.98))
            ha = getattr(cfg.PANEL_LABEL, "HA", "left")
            va = getattr(cfg.PANEL_LABEL, "VA", "top")
            weight = getattr(cfg.PANEL_LABEL, "WEIGHT", "bold")
            size_rel = getattr(cfg.PANEL_LABEL, "SIZE_REL", 1.00)
            fmt = getattr(cfg.PANEL_LABEL, "FORMAT", "({letter})")
            size_pt = mpl.rcParams.get('font.size', 10.0) * size_rel
        else:
            pos, ha, va, weight, size_pt, fmt = (0.02, 0.98), "left", "top", "bold", 10.0, "({letter})"
        size_pt = max(size_pt, 8.0)  # enforce minimum
        ax.text(pos[0], pos[1], fmt.format(letter=lab), transform=ax.transAxes,
                ha=ha, va=va, weight=weight, fontsize=size_pt)

def derive_linewidths(base: float|None=None):
    """Deriva spessori coerenti con eventuale cfg.LINES; altrimenti default robusti."""
    if base is None:
        base = 1.0
    if cfg is not None and hasattr(cfg, "LINES"):
        return {
            "thin":   max(base * getattr(cfg.LINES, "THIN", 0.60), 0.5),
            "medium": max(base * getattr(cfg.LINES, "MEDIUM", 1.00), 0.5),
            "strong": max(base * getattr(cfg.LINES, "STRONG", 1.15), 0.5),
            "strong_min_pt": max(getattr(cfg.LINES, "STRONG_MIN_PT", 1.20), 0.5),
        }
    return {"thin":max(0.6*base,0.5), "medium":max(1.0*base,0.5), "strong":max(1.15*base,0.5), "strong_min_pt":1.20}

def apply_ticks_from_cfg(ax):
    """
    Applica la policy dei tick da cfg.TICKS se presente (no-op se assente).
    """
    if cfg is None or not hasattr(cfg, "TICKS"):
        return
    try:
        xmaj = getattr(cfg.TICKS, "X_MAJ", None)
        ymaj = getattr(cfg.TICKS, "Y_MAJ", None)
        if isinstance(xmaj, int) and xmaj > 0:
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(xmaj, prune=getattr(cfg.TICKS, "PRUNE", "both")))
        if isinstance(ymaj, int) and ymaj > 0:
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(ymaj, prune=getattr(cfg.TICKS, "PRUNE", "both")))
        if getattr(cfg.TICKS, "MINOR", False):
            ax.minorticks_on()
    except Exception as e:
        # Mai rompere il plot per ragioni di stile, ma avvisare.
        warnings.warn(
            f"[MyBasePlots] Fallita applicazione dei tick da plot_cfg. "
            f"Causa: {e.__class__.__name__}: {e}. Controlla la configurazione.",
            UserWarning
        )

def style_debug_report(details: bool=True):
    """Stampa diagnostica minima su rc chiave; se details=True elenca stack stile e stato LaTeX."""
    keys = ["text.usetex", "figure.autolayout", "font.size", "axes.linewidth",
            "lines.linewidth", "xtick.major.size", "ytick.major.size",
            "pdf.fonttype","ps.fonttype","svg.fonttype"]
    out = {k: mpl.rcParams.get(k) for k in keys}
    msg = "[style_debug] " + str(out)
    if details:
        try:
            # Non sappiamo gli ultimi argomenti usati; forniamo info LaTeX e cache
            msg += f" | latex_available_cache={_LATEX_AVAILABLE} | MYBP_DISABLE_LATEX={os.environ.get('MYBP_DISABLE_LATEX','0')}"
        except Exception:
            pass
    print(msg)
