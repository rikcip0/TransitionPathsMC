
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from MyBasePlots.FigCore import utils_style as ustyle
from MyBasePlots.FigCore import plot_cfg as cfg
from MyBasePlots.FigCore import myTemplates as tmpl
from MyBasePlots.FigCore import myEncodings as enc

def test_two_panels_with_strips_and_save(outdir: Path):
    with ustyle.auto_style(mode="overlay"):
        fig, axes, meta = tmpl.new_two_panels_h(reserve_cbar_right=True, reserve_legend_right=True, gaps=(1.2*cfg.GAPS.W_IN, cfg.GAPS.H_IN))
        x = np.linspace(0,1,100)
        enc.apply_cycle(axes[0], "okabe_ito_no_yellow")
        axes[0].plot(x, np.sin(2*np.pi*x), label="sine")
        axes[0].plot(x, np.cos(2*np.pi*x), label="cosine")
        im = axes[1].imshow(np.outer(x,x), origin="lower")
        tmpl.attach_legend_right(fig, axes[0], meta, ncol=1)
        cb = tmpl.attach_colorbar_right(fig, im, meta, orientation="vertical")
        cb.set_label("arb. units")
        tmpl.apply_panel_labels(meta, labels=["a","b"])
        written = tmpl.finalize_and_export(meta, str(outdir / "template_2panels"))
        assert all(Path(p).exists() for p in written)
        plt.close(fig)

def test_grid_template_export_and_save(outdir: Path):
    with ustyle.auto_style(mode="overlay"):
        fig, axes, meta = tmpl.new_grid(2, 2, reserve_cbar_right=True, reserve_legend_right=True, gaps=(1.2*cfg.GAPS.W_IN, cfg.GAPS.H_IN))
        data = np.random.RandomState(0).randn(30,30)
        axes[0].imshow(data, origin="lower")
        enc.apply_cycle(axes[1], "okabe_ito_no_yellow")
        axes[1].plot([0,1],[1,0], label="a")
        axes[1].plot([0,1],[0.5,0.2], label="b")
        tmpl.attach_legend_right(fig, axes[1], meta, ncol=1)
        cb = tmpl.attach_colorbar_right(fig, axes[0].images[0], meta)
        cb.set_label("Z")
        tmpl.apply_panel_labels(meta, labels=["a","b","c","d"])
        written = tmpl.finalize_and_export(meta, str(outdir / "template_grid_2x2"))
        assert all(Path(p).exists() for p in written)
        plt.close(fig)
