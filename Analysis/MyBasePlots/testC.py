"""
test_suite_C_mixed_modalities_v2.py
===================================
Fix leggibilità dei rettangolini categoriali:
- evita sfondi troppo scuri (li schiarisce)
- calcola automaticamente il colore del testo (bianco/nero) per contrasto

Esempio:
  python3 test_suite_C_mixed_modalities_v2.py --mode overlay
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import plot_cfg as cfg
from myTemplates import two_panels, finalize_multi
from utils_style import auto_style
from utils_plot import standardize_ticks
from myColors import apply_cycle, lighten, darken, make_monochrome_ramp, colors_for_categories

def _luminance(hex_color: str) -> float:
    r, g, b = mpl.colors.to_rgb(hex_color)
    return 0.2126*r + 0.7152*g + 0.0722*b

def _text_for_bg(hex_color: str) -> str:
    # soglia comoda per sRGB: >= 0.53 -> testo nero, altrimenti bianco
    return "black" if _luminance(hex_color) >= 0.53 else "white"

def _safe_bg(hex_color: str, min_lum: float = 0.30) -> str:
    # Se il colore è troppo scuro, lo schiarisco parecchio
    if _luminance(hex_color) < min_lum:
        return lighten(hex_color, 0.60)  # sposta verso bianco
    return hex_color

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base","overlay","latex","auto"], default="overlay")
    args = ap.parse_args()

    with auto_style(mode=args.mode):
        fig, (a1, a2) = two_panels(title="Mixed modalities (v2)", legend="outside",
                                    labels=True, inner_labels="outer", gap_w="cfg")

        # --- pannello sinistro: linee + fill ---
        x = np.linspace(0, 10, 800)
        apply_cycle(a1, "okabe_ito")
        y1 = np.sin(x); y2 = np.sin(x)+0.3*np.cos(2*x)
        a1.plot(x, y1, label="signal")
        a1.plot(x, y2, label="signal+noise")
        a1.fill_between(x, y1, y2, color=lighten("#0072B2", 0.55), alpha=0.9, label="Δ")
        standardize_ticks(a1)

        # --- pannello destro: heatmap monocromatica ---
        X = np.linspace(-2,2,240); Y = np.linspace(-2,2,240)
        X, Y = np.meshgrid(X,Y)
        Z = (X**2 + Y**2)
        cmap_m = make_monochrome_ramp("#D55E00", n=256)
        im = a2.imshow(Z, cmap=cmap_m, origin='lower')
        standardize_ticks(a2)

        # --- inset categoriale con contrasto automatico ---
        keys = ["liq", "gas", "solid", "ferro"]
        mapping = colors_for_categories(keys, palette="okabe_ito")  # palette libera; fix contrasto a valle
        x0 = 0.05; w = 0.90/len(keys); y0 = 0.02; h = 0.05
        for i, k in enumerate(keys):
            col = _safe_bg(mapping[k], min_lum=0.30)
            txt = _text_for_bg(col)
            a2.add_patch(plt.Rectangle((x0+i*w, y0), w*0.9, h, color=col,
                                       transform=a2.transAxes, ec="none", zorder=4))
            a2.text(x0+i*w+0.45*w, y0+h*0.5, k, ha="center", va="center",
                    transform=a2.transAxes, fontsize=8, color=txt, zorder=5)

        finalize_multi(fig, (a1, a2), auto_gap_action="warn")
        fig.savefig(f"C_mixed_{args.mode}_v2.png", dpi=300)
        fig.savefig(f"C_mixed_{args.mode}_v2.pdf", dpi=300)

if __name__ == "__main__":
    main()
