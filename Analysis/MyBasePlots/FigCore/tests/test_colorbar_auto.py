
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from MyBasePlots.FigCore import utils_plot as uplot
from MyBasePlots.FigCore import utils_style as ustyle

def test_colorbar_auto_label_and_save(outdir: Path):
    with ustyle.auto_style(mode="overlay"):
        fig, axes, layout = uplot.figure_grid(1, 1, reserve_cbar_right=True)
        ax = axes[0]
        data = np.linspace(0,1,100).reshape(10,10)
        im = ax.imshow(data, origin="lower")
        cb = uplot.add_colorbar_right_auto(fig, ax, layout)
        cb.set_label("Intensity (a.u.)")
        # sanity: label exists
        assert cb.ax.get_ylabel() or cb.ax.get_xlabel()
        uplot.export_figure(fig, str(outdir / "cbar_right_auto"))
        plt.close(fig)
