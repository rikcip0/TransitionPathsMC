
from pathlib import Path
import matplotlib.pyplot as plt
from MyBasePlots.FigCore import myEncodings as enc

def test_palettes_and_cycle_and_save(outdir: Path):
    p = enc.get_palette("okabe_ito")
    assert isinstance(p, list) and len(p) >= 7
    q = enc.get_palette("okabe_ito_no_yellow")
    assert len(q) <= len(p)
    # create a tiny figure that uses the cycle
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4,3))
    enc.apply_cycle(ax, "okabe_ito_no_yellow")
    for i in range(4):
        c,m,ls = enc.get_encoding(i, use_color=True, palette="okabe_ito_no_yellow")
        ax.plot([0,1],[i,i], color=c, marker=m, linestyle=ls, label=f"s{i}")
    ax.legend()
    fig.savefig(outdir / "enc_cycle_demo.png", bbox_inches=None)
    plt.close(fig)
