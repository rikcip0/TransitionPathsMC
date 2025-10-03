
# FigCore tests (gold-figure QA)

All tests save images in `tests/_artifacts/` to allow quick visual inspection.

## Run
From `Analysis/`:
```bash
export PYTHONPATH=.
python3 -m pytest -q MyBasePlots/FigCore/tests
```
Artifacts appear under:
```
MyBasePlots/FigCore/tests/_artifacts/
```

## What is covered
- **Geometry**: panel/data-box sizes constant (`test_panel_geometry.py`).
- **Legend/Colorbar**: right strips reserved; filtered legend; labeled colorbar; no "tight" (`test_legends_colors.py`, `test_colorbar_auto.py`).
- **Templates**: single/two-panels/grid with overlays, labels, export (`test_styles_templates.py`).
- **Encodings**: palettes, combined encodings, and matplotlib `prop_cycle` application (`test_encodings.py`).
- **Report**: rc + figure QA checks true, JSON saved (`test_rc_report.py`).

## Visual checklist
Open the PNG/PDF in `_artifacts/` and verify:
- ticks/labels, titles, colorbar label **not clipped**;
- legend **not overlapping** with colorbar/data panels (dedicated strip);
- palette consistent with `okabe_ito_no_yellow` for line plots;
- panel labels visible and aligned.
