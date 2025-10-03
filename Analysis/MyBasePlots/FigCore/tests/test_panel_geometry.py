
from pathlib import Path
import matplotlib.pyplot as plt
from MyBasePlots.FigCore import plot_cfg as cfg
from MyBasePlots.FigCore import utils_plot as uplot

TOL = 0.02  # inches

def _data_box_size(fig, ax):
    fig.canvas.draw()
    patch = ax.patch
    bb = patch.get_window_extent(renderer=fig.canvas.get_renderer())
    dpi = fig.dpi
    return bb.width/dpi, bb.height/dpi

def test_single_panel_size_and_save(outdir: Path):
    fig, ax, layout = uplot.figure_single()
    w, h = _data_box_size(fig, ax)
    assert abs(w - cfg.PANEL_DATA.WIDTH_IN) <= TOL
    assert abs(h - cfg.PANEL_DATA.HEIGHT_IN) <= TOL
    # export
    uplot.export_figure(fig, str(outdir / "single_panel"))
    plt.close(fig)

def test_grid_panel_sizes_and_save(outdir: Path):
    fig, axes, layout = uplot.figure_grid(1, 2)
    for ax in axes:
        w, h = _data_box_size(fig, ax)
        assert abs(w - cfg.PANEL_DATA.WIDTH_IN) <= TOL
        assert abs(h - cfg.PANEL_DATA.HEIGHT_IN) <= TOL
    uplot.export_figure(fig, str(outdir / "grid_1x2"))
    plt.close(fig)
