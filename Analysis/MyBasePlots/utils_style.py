# utils_style.py
from contextlib import contextmanager
from pathlib import Path
import matplotlib as mpl

# Percorsi risolti rispetto a questo file
_THIS_DIR = Path(__file__).resolve().parent
STYLE_PATH = _THIS_DIR / "myStyle.mplstyle"
LATEX_STYLE_PATH = _THIS_DIR / "myLatexStyle.mplstyle"

@contextmanager
def paper_style():
    """Applica lo stile base (no-LaTeX) localmente."""
    with mpl.rc_context(fname=str(STYLE_PATH)):
        yield

@contextmanager
def latex_style():
    """Applica lo stile LaTeX (font identici al documento) localmente."""
    with mpl.rc_context(fname=str(LATEX_STYLE_PATH)):
        yield

def use_paper_style(func):
    """Decorator: esegue func dentro paper_style()."""
    def _wrapped(*args, **kwargs):
        with paper_style():
            return func(*args, **kwargs)
    _wrapped.__name__ = func.__name__
    return _wrapped

def use_latex_style(func):
    """Decorator: esegue func dentro latex_style()."""
    def _wrapped(*args, **kwargs):
        with latex_style():
            return func(*args, **kwargs)
    _wrapped.__name__ = func.__name__
    return _wrapped
