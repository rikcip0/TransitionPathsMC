from pathlib import Path
import matplotlib as mpl
from contextlib import contextmanager
from functools import wraps

# Path to the shared .mplstyle
STYLE_PATH = Path(__file__).with_name("style.mplstyle")

@contextmanager
def paper_style():
    """Context manager that applies the shared paper style (.mplstyle).
    If the file is missing, it yields without raising (no-op fall-back).
    """
    if STYLE_PATH.is_file():
        with mpl.rc_context(fname=str(STYLE_PATH)):
            yield
    else:
        # fall-back: proceed without style (explicit to avoid silent inconsistencies)
        yield

def use_paper_style(func):
    """Decorator that executes a plotting function inside paper_style()."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        with paper_style():
            return func(*args, **kwargs)
    return wrapped

# ---- Shared geometry/styling helpers ----------------------------------------

def get_main_panel_size():
    """Canonical main panel size (inches), kept consistent across plots."""
    # Keep in sync with W_MAIN, H_MAIN used in plotting modules.
    return 4.2, 3.2

def theme_linewidth():
    """Default line width from rcParams (float). Fallback = 1.5."""
    try:
        return float(mpl.rcParams.get("lines.linewidth", 1.5))
    except Exception:
        return 1.5

def theme_fontsizes():
    """Snapshot of common font sizes from rcParams (ints/floats)."""
    rc = mpl.rcParams
    return {
        "label":  rc.get("axes.labelsize",  rc.get("font.size", 10)),
        "tick":   rc.get("xtick.labelsize", rc.get("font.size", 10)),
        "legend": rc.get("legend.fontsize", rc.get("font.size", 10)),
        "title":  rc.get("axes.titlesize",  rc.get("font.size", 10)),
    }

# (Optional utility) Footer helper for info lines in inches
def add_footer(fig, left_text: str = "", right_text: str = "",
               height_in: float = 0.32, pad_in: float = 0.06,
               fontsize=None):
    """Reserve a fixed-height footer (inches) at the bottom of the figure and
    render left/right info texts. This function shrinks all existing axes
    vertically to make room for the footer, preserving the main panel size in inches.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Target figure.
    left_text, right_text : str
        Text rendered aligned left/right.
    height_in : float
        Footer height in inches.
    pad_in : float
        Vertical padding between footer text and the figure edge (inches).
    fontsize : float or None
        If None, uses rcParams['axes.labelsize'].

    Notes
    -----
    - Works best when called after axes creation but before `tight_layout`
      (if any) to keep layout deterministic.
    - Does not modify the width; only vertical positions are adjusted.
    """
    import matplotlib.pyplot as plt

    W, H = fig.get_size_inches()
    if H <= 0:
        return  # degenerate figure

    footer_frac = max(0.0, min(1.0, height_in / H))
    pad_frac    = max(0.0, min(1.0, pad_in    / H))

    # Shift all existing axes upward to free space for the footer
    for ax in fig.axes:
        # Skip colorbars/legends that are outside normalized [0,1] box if any
        try:
            box = ax.get_position()
        except Exception:
            continue
        new_y0 = box.y0 + footer_frac
        new_y1 = box.y1 + footer_frac
        # Keep inside [0,1]
        if new_y1 > 1.0:
            # Clip (rare): stack just below the top
            dy = new_y1 - 1.0
            new_y0 -= dy
            new_y1 = 1.0
        ax.set_position([box.x0, new_y0, box.width, new_y1 - new_y0])

    # Add the footer axes across the full width
    ax_footer = fig.add_axes([0.0, 0.0, 1.0, footer_frac], frame_on=False)
    ax_footer.set_axis_off()

    fs = fontsize if fontsize is not None else mpl.rcParams.get("axes.labelsize", mpl.rcParams.get("font.size", 10))
    # Left text (baseline aligned)
    if left_text:
        ax_footer.text(0.0, pad_frac / max(footer_frac, 1e-6), left_text,
                       ha="left", va="bottom", fontsize=fs)
    # Right text
    if right_text:
        ax_footer.text(1.0, pad_frac / max(footer_frac, 1e-6), right_text,
                       ha="right", va="bottom", fontsize=fs)

    return ax_footer
