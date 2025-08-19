"""
test_suite_A_single_multi_autogap_v1.py
=======================================
- Single + Two-panels con autogap opzionale.
- Usa: myTemplates, plot_cfg, utils_style.auto_style, utils_plot.standardize_ticks, myColors.apply_cycle
- Esporta PNG+PDF; stampa dimensioni AXES/DATA e gap consigliato/usato.

Esempi:
  python3 test_suite_A_single_multi_autogap_v1.py --mode overlay --inner both --gap auto
  python3 test_suite_A_single_multi_autogap_v1.py --mode base    --inner outer --gap cfg
"""
import argparse, sys
import numpy as np
import matplotlib.pyplot as plt

import plot_cfg as cfg
from myTemplates import single_panel, finalize_single, two_panels, finalize_multi
from utils_style import auto_style
from utils_plot import standardize_ticks, axes_bbox_inches, data_bbox_inches, recommend_gap_between
from myColors import apply_cycle

def parse_gap(arg):
    if arg.lower() in ("cfg","auto"):
        return arg.lower()
    try:
        return float(arg)
    except ValueError:
        raise SystemExit("--gap must be 'cfg'|'auto' or a float (inches)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["base","overlay","latex","auto"], default="overlay")
    ap.add_argument("--inner", choices=["outer","both"], default="outer", help="policy ticklabels interni")
    ap.add_argument("--gap", default="cfg", help="cfg|auto|float-inches per gap_w")
    args = ap.parse_args()
    gap = parse_gap(args.gap)

    # ---------- SINGLE ----------
    with auto_style(mode=args.mode):
        fig, ax = single_panel(title=f"Single ({args.mode})", legend="outside")
        x = np.linspace(0, 10, 800)
        apply_cycle(ax, "okabe_ito")
        for k in range(4):
            ax.plot(x, np.sin(x+0.4*k), label=f"curve {k+1}")
        standardize_ticks(ax)
        finalize_single(fig, ax, legend="outside")
        Aw, Ah = axes_bbox_inches(fig, ax)
        Dw, Dh = data_bbox_inches(fig, ax)
        print(f"[single/{args.mode}] AXES: {Aw:.3f}×{Ah:.3f} in | DATA: {Dw:.3f}×{Dh:.3f} in | target {cfg.PANEL.W_MAIN}×{cfg.PANEL.H_MAIN}")
        fig.savefig(f"A_single_{args.mode}.png", dpi=300)
        fig.savefig(f"A_single_{args.mode}.pdf", dpi=300)
        plt.close(fig)

    # ---------- TWO PANELS ----------
    with auto_style(mode=args.mode):
        fig, (a1, a2) = two_panels(title=f"Two panels ({args.mode})", legend="outside",
                                    labels=True, inner_labels=args.inner, gap_w=gap)
        x = np.linspace(0, 10, 800)
        a1.plot(x, np.sin(x), label="sin")
        a2.plot(x, np.cos(0.8*x), label="cos")
        standardize_ticks(a1); standardize_ticks(a2)
        finalize_multi(fig, (a1, a2), auto_gap_action="expand")
        Aw, Ah = axes_bbox_inches(fig, a1)
        gap_need = recommend_gap_between(a1, a2, min_gap_in=cfg.GAPS.GAP_W)
        print(f"[multi/{args.mode}] AXES(each): {Aw:.3f}×{Ah:.3f} in | GAP default: {cfg.GAPS.GAP_W:.3f} in | GAP consigliato: {gap_need:.3f} in")
        fig.savefig(f"A_multi_{args.mode}_{args.inner}_{str(gap).replace('.','p')}.png", dpi=300)
        fig.savefig(f"A_multi_{args.mode}_{args.inner}_{str(gap).replace('.','p')}.pdf", dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    main()
