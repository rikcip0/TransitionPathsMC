
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pytest

from MyBasePlots.FigCore import utils_style as ustyle
from MyBasePlots.FigCore import myTemplates as tmpl
from MyBasePlots.FigCore import report_config as rep

PKG_DIR = Path(ustyle.__file__).resolve().parent
STYLES_DIR = PKG_DIR / "styles"
BASE = STYLES_DIR / "paper_base.mplstyle"
OVERLAY_LATEX = STYLES_DIR / "overlay_latex.mplstyle"
OVERLAY_SCREEN = STYLES_DIR / "overlay_screen.mplstyle"
OVERLAY_FALLBACK = OVERLAY_SCREEN if OVERLAY_SCREEN.exists() else OVERLAY_LATEX

MODES = [
    ("overlay", OVERLAY_FALLBACK, ("pdf","png")),
    ("latex_text", OVERLAY_LATEX, ("pdf","png")),
    ("latex", OVERLAY_LATEX, ("pdf",)),
]

@pytest.mark.parametrize("mode,overlay_path,formats", MODES)
def test_style_profiles(outdir: Path, mode: str, overlay_path: Path, formats):
    assert BASE.exists(), f"Missing base style: {BASE}"
    assert overlay_path.exists(), f"Missing overlay style: {overlay_path}"

    with ustyle.auto_style(mode=mode, base=str(BASE), overlay=str(overlay_path)):
        fig, ax, meta = tmpl.new_single_panel()
        x = np.linspace(0, 2*np.pi, 400)
        ax.plot(x, np.sin(x), label=r"$\sin x$")
        ax.set_xlabel(r"$t$ [s]"); ax.set_ylabel(r"$A$ [a.u.]")
        ax.set_title(rf"mode={mode}: $E=mc^2$; $\int_0^\pi \sin x\,dx=2$")
        ax.legend(loc="best", frameon=True)

        root = outdir / f"style_{mode}_demo"
        written = tmpl.finalize_and_export(meta, str(root), formats=formats)
        print(f"[WRITTEN {mode}] ", written)

        # optional .pgf for 'latex'
        if mode == "latex":
            try:
                pgf_path = outdir / f"style_{mode}_demo.pgf"
                fig.savefig(str(pgf_path))
                print(f"[WRITTEN {mode}] {pgf_path}")
            except Exception as e:
                print(f"[PGF SAVE FAILED] {e}")

        rep.write_json(rep.report(fig, check_panel_size=False), str(outdir / f"rc_{mode}.json"))

        assert (root.with_suffix(".pdf")).exists(), f"PDF non trovato per mode={mode}"
        if "png" in formats:
            assert (root.with_suffix(".png")).exists(), f"PNG non trovato per mode={mode}"
        plt.close(fig)
