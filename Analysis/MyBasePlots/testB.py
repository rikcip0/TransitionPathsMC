"""
test_suite_B_grid_2x2_colorbar_v1.py
====================================
- Costruisce un layout 2×2 *deterministico* direttamente da plot_cfg (senza tight_layout).
- Heatmap divergente condividendo **una sola colorbar verticale** nella strip destra.
- Etichette solo ai bordi esterni (outerize_grid).

Esempio:
  python3 test_suite_B_grid_2x2_colorbar_v1.py --mode overlay
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

import plot_cfg as cfg
from utils_style import auto_style
from utils_plot import outerize_grid
from myColors import diverging_from, norm_centered

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base","overlay","latex","auto"], default="overlay")
    args = ap.parse_args()

    # --- figure size dedotta da cfg ---
    right_extra = cfg.legend_strip_width()
    fig_w = (2*cfg.PANEL.W_MAIN + cfg.GAPS.GAP_W + cfg.MARGINS.RIGHT_PAD + right_extra) / (1.0 - cfg.MARGINS.LEFT_FRAC)
    top_pad = cfg.MARGINS.TOP_PAD_BASE
    fig_h = (2*cfg.PANEL.H_MAIN + cfg.GAPS.GAP_H + top_pad) / (1.0 - cfg.MARGINS.BOTTOM_FRAC)

    with auto_style(mode=args.mode):
        fig = plt.figure(figsize=(fig_w, fig_h))
        # Ricaviamo frazioni figura
        fw, fh = fig.get_size_inches()
        left_frac = cfg.MARGINS.LEFT_FRAC
        bottom_frac = cfg.MARGINS.BOTTOM_FRAC
        width_frac  = cfg.PANEL.W_MAIN / fw
        height_frac = cfg.PANEL.H_MAIN / fh
        gap_w_frac  = cfg.GAPS.GAP_W / fw
        gap_h_frac  = cfg.GAPS.GAP_H / fh
        # x0 per colonna sinistra e destra
        x0L = left_frac
        x0R = left_frac + width_frac + gap_w_frac
        # y0 per riga bassa e alta
        y0B = bottom_frac
        y0T = bottom_frac + height_frac + gap_h_frac

        ax = []
        ax.append(fig.add_axes([x0L, y0T, width_frac, height_frac]))  # (0,0) top-left
        ax.append(fig.add_axes([x0R, y0T, width_frac, height_frac]))  # (0,1) top-right
        ax.append(fig.add_axes([x0L, y0B, width_frac, height_frac]))  # (1,0) bottom-left
        ax.append(fig.add_axes([x0R, y0B, width_frac, height_frac]))  # (1,1) bottom-right

        # Dati e cmap
        x = np.linspace(-3,3,240); y = np.linspace(-3,3,240)
        X, Y = np.meshgrid(x,y)
        Z = np.sin(X)*np.cos(Y) + 0.15*np.random.RandomState(0).randn(*X.shape)
        cmap = diverging_from("#3B4CC0", "#B40426", n=256)
        norm = norm_centered(Z, center=0.0, method="percentile", p=99)

        ims = []
        for a in ax:
            im = a.imshow(Z, cmap=cmap, norm=norm, origin='lower')
            ims.append(im)

        # Solo label esterne
        outerize_grid(ax, nrows=2, ncols=2)

        # Colorbar nella strip destra (deterministica)
        # ricaviamo la posizione a destra del pannello destro
        x1_right = x0R + width_frac
        pad_frac = cfg.LEGEND.PAD_W / fw
        strip_w = cfg.legend_strip_width() / fw
        cbar_w = min(0.05, strip_w - pad_frac - 0.01)
        cax = fig.add_axes([x1_right + pad_frac, y0B, cbar_w, height_frac*2 + gap_h_frac])
        cax.grid(False)  # << evita il warning
        fig.colorbar(ims[0], cax=cax)

        fig.savefig(f"B_grid2x2_{args.mode}.png", dpi=300)
        fig.savefig(f"B_grid2x2_{args.mode}.pdf", dpi=300)

if __name__ == "__main__":
    main()
