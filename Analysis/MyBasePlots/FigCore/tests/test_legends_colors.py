
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from MyBasePlots.FigCore import myEncodings as enc
from MyBasePlots.FigCore import utils_plot as uplot
from MyBasePlots.FigCore import utils_style as ustyle

def _include_main(handle, label):
    return "diag" not in (label or "").lower()

def test_legend_filter_encodings_and_save(outdir: Path):
    with ustyle.auto_style(mode="overlay"):
        fig, axes, layout = uplot.figure_grid(1, 2, reserve_legend_right=True)
        ax = axes[0]
        enc.apply_cycle(ax, "okabe_ito_no_yellow")
        x = np.linspace(0,1,50)
        for i, lab in enumerate(["main1","main2","diag1","diag2"]):
            c,m,ls = enc.get_encoding(i, use_color=True, palette="okabe_ito_no_yellow")
            ax.plot(x, x + 0.1*i, color=c, marker=m, linestyle=ls, label=lab, linewidth=1.2)
        # filtered legend on right strip
        handles, labels = uplot.get_legend_handles_labels(ax, include=_include_main)
        lg = uplot.add_legend_outside_right(fig, ax, layout, handles=handles, labels=labels, ncol=1, frameon=True)
        # basic sanity: legend exists and has items filtered
        assert lg is not None and all("diag" not in t.get_text().lower() for t in lg.get_texts())
        uplot.export_figure(fig, str(outdir / "legend_right_filtered"))
        plt.close(fig)
