# latex_env.py — v2 (strict core, optional extras)
from __future__ import annotations
import shutil, subprocess, os
from typing import Dict, List

def _texsystem() -> str:
    return os.getenv("MYBP_TEXSYSTEM", "pdflatex")  # o xelatex/lualatex

def _which(cmd: str) -> bool:
    return shutil.which(cmd) is not None

def _kpsewhich(pkgfile: str) -> bool:
    try:
        if not _which("kpsewhich"):
            return False
        p = subprocess.run(["kpsewhich", pkgfile], capture_output=True, text=True, check=False)
        return bool(p.stdout.strip())
    except Exception:
        return False

def _need_pkgs(pkgs: List[str]) -> List[str]:
    missing = []
    for p in pkgs:
        probe = p if p.endswith(".sty") else f"{p}.sty"
        if not _kpsewhich(probe):
            missing.append(p)
    return missing

def diag_report() -> Dict[str, object]:
    core = ["amsmath","amssymb","lmodern","fontenc"]
    extra = ["siunitx"]
    return {
        "texsystem": _texsystem(),
        "tools_ok": _which(_texsystem()),
        "kpsewhich_ok": _which("kpsewhich"),
        "missing_core": _need_pkgs(core),
        "missing_extra": _need_pkgs(extra),
    }

def require_tex(backend: str="pgf",
                pkgs: List[str] | None = None,
                strict_extra: bool | None = None) -> Dict[str, object]:
    """
    backend='pgf'   -> figura via LaTeX/PGF (heavy)
    backend='usetex'-> LaTeX solo per il testo
    Core richiesti (sempre): amsmath, amssymb, lmodern, fontenc
    Extra raccomandati: siunitx (opzionale se strict_extra=0 o MYBP_LATEX_STRICT_EXTRA=0)
    Ritorna un dict di rcParams da applicare; solleva se requisiti core mancano.
    """
    if backend not in {"pgf","usetex"}:
        raise ValueError("backend must be 'pgf' or 'usetex'")

    # toolchain
    if not _which(_texsystem()):
        raise RuntimeError(f"LaTeX engine '{_texsystem()}' not found on PATH")
    if not _which("kpsewhich"):
        raise RuntimeError("kpsewhich not found; cannot probe LaTeX packages")

    core = ["amsmath","amssymb","lmodern","fontenc"]
    extra = ["siunitx"]
    if pkgs:
        for p in pkgs:
            if p not in core and p not in extra:
                extra.append(p)

    miss_core = _need_pkgs(core)
    miss_extra = _need_pkgs(extra)

    if miss_core:
        raise RuntimeError(f"Missing LaTeX core packages: {miss_core}")

    if strict_extra is None:
        strict_extra = os.getenv("MYBP_LATEX_STRICT_EXTRA", "1") != "0"

    # preambolo: sempre core; aggiungi extras solo se presenti o se strict (ma allora avresti già fallito)
    preamble = r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{lmodern}\usepackage[T1]{fontenc}"
    if not miss_extra:
        preamble += r"\usepackage{siunitx}"

    if backend == "pgf":
        return {
            "text.usetex": True,
            "pgf.texsystem": _texsystem(),
            "pgf.rcfonts": False,
            "font.family": "serif",
            "pgf.preamble": preamble,
            "text.latex.preamble": preamble,
        }
    else:  # usetex
        return {
            "text.usetex": True,
            "font.family": "serif",
            "text.latex.preamble": preamble,
        }
