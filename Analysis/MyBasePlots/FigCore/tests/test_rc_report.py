
from pathlib import Path
import matplotlib.pyplot as plt
from MyBasePlots.FigCore import report_config as rep
from MyBasePlots.FigCore import myTemplates as tmpl
from MyBasePlots.FigCore import utils_style as ustyle

def test_report_checks_true_and_save(outdir: Path):
    with ustyle.auto_style(mode="overlay"):
        fig, ax, meta = tmpl.new_single_panel()
        ax.plot([0,1],[0,1])
        r = rep.report(fig, check_panel_size=True, tol_in=0.05)
        assert "checks" in r and all(r["checks"].values())  # all ok
        # dump json for inspection next to images
        rep.write_json(r, str(outdir / "report_single.json"))
        tmpl.finalize_and_export(meta, str(outdir / "report_single_fig"))
        plt.close(fig)
