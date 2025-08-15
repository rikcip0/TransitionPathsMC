# MyBasePlots/plotWithDifferentColorbars.py

def plotWithDifferentColorbars(
    name, x, xName, y, yName, title,
    markerEdgeVariable, edge_shortDescription, edgeColorPerVar,
    markerShapeVariable, markerShapeVariableNames,
    arrayForColorCoordinate, colorMapSpecifier=None,
    edgeColorVariableName='Initialization',
    colorCoordinateVariableName='', colorMapSpecifierName='',
    dynamicalTicksForColorbars=False,
    additionalMarkerTypes=None,                     # compat
    additionalMarkerTypes_Unused=None,              # compat
    yerr=None, fitTypes=None, xscale='', yscale='',
    fittingOverDifferentEdges=True,                 # compat
    nGraphs=None,                                   # compat: riportato in meta
    functionsToPlotContinuously=None,
    theoreticalX=None, theoreticalY=None,
    linesAtXValueAndName=None, linesAtYValueAndName=None,
    connect_points=False,
    fitGroupBy=None,             # tuple in {'shape','edge','spec','color'}; default: tutte
    countUniqueOf=None,          # p.es. ('r','g')
    seriesByName=None,           # dict: nome -> array allineato a x
    legendWidth=None,            # override manuale (inch)
    legendNcol=None,             # override # colonne legenda
    fallbackPointColor=None,     # facecolor dei marker quando non c’è cbar
    hideEdgeWhenSingle=True,     # se True e c’è un solo edge: niente outline
    markerSize=30,               # s di scatter (pt^2); aumentato se no-edge
    showSpecInInfoWhenNoCbar=True,  # mostra "<spec>=<val>" in infoline se cbar assente
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize, hsv_to_rgb
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import ScalarFormatter, MaxNLocator, NullLocator, FormatStrFormatter
    from matplotlib import colors as mcolors
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    from scipy.optimize import curve_fit
    from MyBasePlots.utils import paper_style
    from matplotlib.collections import LineCollection
    
    DELIM = " | "
    fitTypes = tuple(fitTypes or [])
    fitGroupBy = tuple(fitGroupBy) if fitGroupBy is not None else ('shape','edge','spec','color')
    countUniqueOf = tuple(countUniqueOf) if countUniqueOf else tuple()
    seriesByName = dict(seriesByName or {})
    warnings = []

    # ------------------ COSTANTI GEOMETRICHE ------------------
    W_MAIN = 4.2     # inch
    H_MAIN = 3.2     # inch
    H_CB   = 0.14    # inch
    H_SP_BIG   = 0.55
    H_SP_SMALL = 0.40
    TOP_PAD_BASE = 0.52

    # ------------------ Helper varie ------------------
    def _as_str_key(arr):
        arr = np.asarray(arr, dtype=object)
        if arr.ndim == 1:
            return arr.astype(str)
        return np.array(['|'.join(map(str, row)) for row in arr], dtype=object)

    def _edge_label(ev):
        if isinstance(edge_shortDescription, dict):
            if ev in edge_shortDescription:
                return edge_shortDescription[ev]
            evs = str(ev)
            if evs in edge_shortDescription:
                return edge_shortDescription[evs]
        return str(ev)

    def _edge_color(ev):
        if isinstance(edgeColorPerVar, dict):
            if ev in edgeColorPerVar:
                return edgeColorPerVar[ev]
            evs = str(ev)
            if evs in edgeColorPerVar:
                return edgeColorPerVar[evs]
        return "#555555"

    def _cb_decimals(vmin, vmax):
        rng = abs(vmax - vmin)
        if not np.isfinite(rng) or rng == 0:
            return 2
        if rng >= 10:    return 0
        if rng >= 1:     return 1
        if rng >= 0.1:   return 2
        if rng >= 0.01:  return 3
        return 4

    def _try_float(s):
        try: return float(s)
        except Exception: return None

    def _fmt_pm(v, e):
        if not np.isfinite(v): return "nan"
        if not (np.isfinite(e) and e != 0): return f"{v:.3g}±{e:.1g}" if np.isfinite(e) else f"{v:.3g}"
        scale = max(abs(v), abs(e))
        if scale == 0:
            return f"{v:.3g}±{e:.1g}"
        E = int(np.floor(np.log10(scale)))
        if abs(E) >= 2:
            s = 10.0**E
            return f"({v/s:.3g}±{e/s:.1g})×10^{E}"
        return f"{v:.3g}±{e:.1g}"
    
    def _lighten_rgba(rgba, alpha=0.8, w=0.35):
        r,g,b,a = rgba
        r = r*(1-w) + 1.0*w
        g = g*(1-w) + 1.0*w
        b = b*(1-w) + 1.0*w
        return (r, g, b, alpha)

    def _sep():     return Line2D([], [], linestyle='None'), " "
    def _section(t):return Line2D([], [], linestyle='None'), f"{t}"

    # -------------- FITTERS --------------
    def _fit_linear(xv,yv):
        popt, pcov = curve_fit(lambda t, c, m: c + m*t, xv, yv)
        c, m = popt
        errs = np.sqrt(np.diag(pcov)) if (pcov.size) else [np.nan, np.nan]
        return {"model":"linear", "eq":r"$y=c+m\,x$", "params":{"c":c,"m":m}, "stderr":{"c":errs[0],"m":errs[1]}}

    def _fit_quadratic(xv,yv):
        popt, pcov = curve_fit(lambda t, c, a: c + a*t*t, xv, yv)
        c, a = popt
        errs = np.sqrt(np.diag(pcov)) if (pcov.size) else [np.nan, np.nan]
        return {"model":"quadratic", "eq":r"$y=c+a\,x^{2}$", "params":{"c":c,"a":a}, "stderr":{"c":errs[0],"a":errs[1]}}

    def _fit_expo(xv,yv):
        p0 = [yv[-1] if np.isfinite(yv[-1]) else np.nanmax(yv), 1.0, 0.0]
        popt, pcov = curve_fit(lambda t, c, m, s: c*(1.0 - np.exp(-(t-s)*m)), xv, yv, p0=p0, maxfev=200000)
        c, m, s = popt
        errs = np.sqrt(np.diag(pcov)) if (pcov.size) else [np.nan, np.nan, np.nan]
        return {"model":"expo", "eq":r"$y=c\,(1-e^{-m(x-s)})$", "params":{"c":c,"m":m,"s":s}, "stderr":{"c":errs[0],"m":errs[1],"s":errs[2]}}

    FITTERS = {"linear":_fit_linear, "quadratic":_fit_quadratic, "expo":_fit_expo}

    # ------------------ PREPROCESS ------------------
    with paper_style():
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape:
            raise ValueError("x e y devono avere la stessa forma.")

        yerr = None if yerr is None else np.asarray(yerr, dtype=float)
        markerEdgeVariable  = np.asarray(markerEdgeVariable, dtype=object)
        markerShapeVariable = np.asarray(markerShapeVariable, dtype=object)
        arrayForColorCoordinate = np.asarray(arrayForColorCoordinate, dtype=float)
        if colorMapSpecifier is None:
            colorMapSpecifier = np.full(x.size, "default", dtype=object)
        else:
            colorMapSpecifier = np.asarray(colorMapSpecifier, dtype=object)

        valid_xy = np.isfinite(x) & np.isfinite(y)
        if yerr is not None: valid_xy &= np.isfinite(yerr)
        if not np.any(valid_xy):
            return None, None, {"empty": True}

        x, y = x[valid_xy], y[valid_xy]
        if yerr is not None: yerr = yerr[valid_xy]
        markerEdgeVariable  = markerEdgeVariable[valid_xy]
        markerShapeVariable = markerShapeVariable[valid_xy]
        arrayForColorCoordinate = arrayForColorCoordinate[valid_xy]
        colorMapSpecifier   = colorMapSpecifier[valid_xy]

        for k in list(seriesByName.keys()):
            arr = np.asarray(seriesByName[k])
            if arr.shape != valid_xy.shape:
                warnings.append(f"seriesByName['{k}'] shape mismatch, ignored.")
                seriesByName.pop(k, None)
            else:
                seriesByName[k] = arr[valid_xy]

        edge_key  = _as_str_key(markerEdgeVariable)
        shape_key = _as_str_key(markerShapeVariable)
        spec_key  = _as_str_key(colorMapSpecifier)

        order = np.lexsort((spec_key, edge_key, shape_key, x))
        
        if functionsToPlotContinuously is not None:
            funs, masks = functionsToPlotContinuously
            if masks is None or len(masks) != len(funs):
                masks = [None]*len(funs)
            masks_norm = []
            for m in masks:
                if m is None:
                    masks_norm.append(np.ones_like(x, dtype=bool))
                else:
                    m = np.asarray(m).astype(bool)
                    if m.size != x.size:
                        masks_norm.append(np.ones_like(x, dtype=bool))
                    else:
                        masks_norm.append(m)
            functionsToPlotContinuously = (funs, masks_norm)
        x, y = x[order], y[order]
        if yerr is not None: yerr = yerr[order]
        markerEdgeVariable  = markerEdgeVariable[order]
        markerShapeVariable = markerShapeVariable[order]
        arrayForColorCoordinate = arrayForColorCoordinate[order]
        colorMapSpecifier   = colorMapSpecifier[order]
        edge_key, shape_key, spec_key = edge_key[order], shape_key[order], spec_key[order]
        for k in seriesByName:
            seriesByName[k] = seriesByName[k][order]
            
        if functionsToPlotContinuously is not None:
            funs, masks = functionsToPlotContinuously
            if masks is None or len(masks) != len(funs):
                masks = [None]*len(funs)
            masks_norm = []
            for m in masks:
                if m is None:
                    m = np.ones_like(x, dtype=bool)  # ora x è già riordinato
                else:
                    m = np.asarray(m).astype(bool)
                    # se avevi preparato il mask sulla lunghezza pre-ordine, ri-permuta
                    if m.size == order.size:
                        m = m[order]
                    elif m.size != x.size:
                        m = np.ones_like(x, dtype=bool)
                masks_norm.append(m)
            functionsToPlotContinuously = (funs, masks_norm)
            
        uniq_edge_vals = np.unique(markerEdgeVariable)
        uniq_edge_vals = uniq_edge_vals[np.argsort(_as_str_key(uniq_edge_vals))]
        uniq_shapes = np.unique(shape_key)
        uniq_spec   = np.unique(spec_key)

        # ------------------ COLORMAP & CBAR POLICY ------------------
        cmaps = {}
        if uniq_spec.size == 1:
            cmaps[uniq_spec[0]] = plt.cm.cool
        else:
            hues = np.linspace(0.05, 0.95, uniq_spec.size, endpoint=True)
            for sp, h in zip(uniq_spec, hues):
                t = np.linspace(0, 1, 256)
                s = np.full_like(t, 0.90)
                v = 0.35 + 0.60*t
                hsv_arr = np.stack([np.full_like(t, h), s, v], axis=1)
                cmaps[sp] = ListedColormap(hsv_to_rgb(hsv_arr))

        counts = {}
        for sp in uniq_spec:
            m = (spec_key == sp) & np.isfinite(arrayForColorCoordinate)
            counts[sp] = np.unique(arrayForColorCoordinate[m]).size

        suppress_all_cbar = (uniq_spec.size == 1 and next(iter(counts.values()), 0) == 1)

        cbar_ranges = {}
        for sp in uniq_spec:
            m = (spec_key == sp) & np.isfinite(arrayForColorCoordinate)
            if np.any(m) and counts[sp] >= 2:
                vmin, vmax = float(np.min(arrayForColorCoordinate[m])), float(np.max(arrayForColorCoordinate[m]))
            else:
                if np.any(m):
                    v = float(arrayForColorCoordinate[m][0])
                else:
                    v = 0.0
                eps = 1e-9 if v == 0 else 1e-6*abs(v)
                vmin, vmax = v - eps, v + eps
            cbar_ranges[sp] = (vmin, vmax)

        n_cb_shown = 0 if suppress_all_cbar else uniq_spec.size

        # ------------------ PREPASS: FIT meta per dimensionare legenda ------------------
        def _compute_fits_meta():
            if not fitTypes:
                return []
            meta = []

            def groups_iter():
                for sh in uniq_shapes:
                    sh_mask = (shape_key == str(sh))
                    for ev in uniq_edge_vals:
                        e_mask = (markerEdgeVariable == ev)
                        for sp in uniq_spec:
                            s_mask = (spec_key == sp)
                            base = sh_mask & e_mask & s_mask
                            if not np.any(base): 
                                continue
                            vals = arrayForColorCoordinate[base]
                            xx = x[base]; yy = y[base]
                            if 'color' in fitGroupBy and not suppress_all_cbar:
                                u = np.unique(vals[np.isfinite(vals)])
                                if u.size == 0:
                                    yield (sh, ev, sp, None, xx, yy)
                                else:
                                    for uv in u:
                                        m = np.isfinite(vals) & (vals == uv)
                                        yield (sh, ev, sp, float(uv), xx[m], yy[m])
                            else:
                                yield (sh, ev, sp, None, xx, yy)

            for sh, ev, sp, cval, xx, yy in groups_iter():
                if xx.size < 3 or not np.all(np.isfinite(xx)) or not np.all(np.isfinite(yy)):
                    continue
                for fname in fitTypes:
                    fitter = FITTERS.get(fname)
                    if fitter is None: 
                        continue
                    try:
                        res = fitter(xx, yy)
                        entry = {
                            "group":{"shape":str(sh),"edge":_edge_label(ev),"spec":str(sp),"color":cval},
                            "fit":fname, "eq":res["eq"],
                            "params":{k:float(v) for k,v in res["params"].items()},
                            "stderr":{k:float(v) for k,v in res["stderr"].items()},
                            "ignoring":[d for d in ('shape','edge','spec','color') if d not in fitGroupBy]
                        }
                        meta.append(entry)
                    except Exception:
                        pass
            return meta

        fits_meta_pre = _compute_fits_meta()

        # ------------- COSTRUZIONE LISTA VOCI LEGENDA (DRY RUN) -------------
        
        legend_lines = []

        # Edge
        edge_vals = list(uniq_edge_vals)
        edge_vals.sort(key=lambda ev: (_edge_label(ev)))
        single_edge_value = (edge_vals[0] if len(edge_vals) == 1 else None)
        if len(edge_vals) > 1:
            legend_lines.append(('header', f"{edgeColorVariableName}:"))
            for ev in edge_vals:
                legend_lines.append(('item', _edge_label(ev)))
            legend_lines.append(('sep', ' '))

        # Shapes
        
        # Shapes (SOLO testo per la stima colonne/larghezza)
        shape_vals = list(uniq_shapes)
        def _shape_key(sh):
            v = _try_float(sh);  return (0, v) if v is not None else (1, str(sh))
        shape_vals.sort(key=_shape_key)
        shape_title = DELIM.join(map(str, markerShapeVariableNames)) \
                        if isinstance(markerShapeVariableNames, (list, tuple)) else str(markerShapeVariableNames)
        if shape_vals:
            legend_lines.append(('header', f"{shape_title}:"))
            for sh in shape_vals:
                lab = r'$\infty$' if str(sh).lower() in ('inf','infty') else str(sh)
                legend_lines.append(('item', lab))
            legend_lines.append(('sep', ' '))
            marker_pool = ['s','^','o','v','D','p','h','X','<','>','*','P']

            shape_to_marker = {}
            non_inf_shapes = []
            for sh in shape_vals:
                if str(sh).lower() in ('inf', 'infty'):
                    shape_to_marker[sh] = '.'
                else:
                    non_inf_shapes.append(sh)

            # assegna marker alle shape non-inf in ordine deterministico
            for i, sh in enumerate(non_inf_shapes):
                shape_to_marker[sh] = marker_pool[i % len(marker_pool)]

        else:
            shape_to_marker = {}

        # Fits
        if fits_meta_pre:
            legend_lines.append(('header', "Fits:"))
            ORDER = {"linear":("m","c"), "quadratic":("a","c"), "expo":("m","c","s")}
            for fm in fits_meta_pre:
                eq_line = fm["eq"]
                ign = fm.get("ignoring", [])
                if ign: eq_line += f" (ignoring {DELIM.join(ign)})"
                legend_lines.append(('item', eq_line))
                ps, se = fm["params"], fm["stderr"]
                for k in ORDER.get(fm["fit"], ps.keys()):
                    v = ps.get(k, np.nan); e = se.get(k, np.nan)
                    legend_lines.append(('item', f"{k}={_fmt_pm(v, e)}"))
            legend_lines.append(('sep', ' '))

        # Summary
        if countUniqueOf:
            parts = []
            for nm in countUniqueOf:
                arr = seriesByName.get(nm)
                if arr is not None:
                    u = np.unique(arr[np.isfinite(arr)])
                    parts.append(f"#{nm}={u.size}")
            if parts:
                legend_lines.append(('header', "Summary:"))
                legend_lines.append(('item', ", ".join(parts)))
                legend_lines.append(('sep', ' '))

        # Pulizia
        clean = []
        for i, (k, t) in enumerate(legend_lines):
            if k == 'header':
                nxt = next((kk for kk, _ in legend_lines[i+1:] if kk != 'sep'), None)
                if nxt != 'item':
                    continue
            clean.append((k, t))
        while clean and clean[-1][0] != 'item':
            clean.pop()
        legend_lines = clean

        # --------- STIMA N COLONNE E LEG_W ---------
        base_fs = plt.rcParams.get("font.size", 9.0)
        leg_fs  = max(6.0, base_fs - 1.0)
        char_w_in = 0.6 * (leg_fs/72.0)
        handle_w_in = 0.32
        pad_in = 0.20
        col_spacing_in = 0.28

        def estimate_width(labels, ncol):
            if ncol <= 1:
                maxlen = max((len(t) for _, t in labels), default=0)
                return handle_w_in + pad_in + char_w_in*maxlen
            n = len(labels)
            rows = int(np.ceil(n / ncol))
            widths = []
            for c in range(ncol):
                seg = labels[c*rows:(c+1)*rows]
                maxlen = max((len(t) for _, t in seg), default=0)
                widths.append(handle_w_in + pad_in + char_w_in*maxlen)
            return sum(widths) + (ncol-1)*col_spacing_in

        labels_for_width = [(k, t) for (k, t) in legend_lines if k in ('header','item')]
        if legendNcol is not None:
            ncol_est = max(1, int(legendNcol))
        else:
            ncol_est = 1
            w1 = estimate_width(labels_for_width, 1)
            if w1 > 2.2:
                w2 = estimate_width(labels_for_width, 2)
                ncol_est = 2
                if w2 > 3.0:
                    w3 = estimate_width(labels_for_width, 3)
                    ncol_est = 3

        if legendWidth is not None:
            LEG_W = float(legendWidth)
        else:
            LEG_W = estimate_width(labels_for_width, ncol_est) + 0.25

        # ------------------ LAYOUT FIGURA ------------------
        n_title_lines = 1 + (title.count("\n") if title else 0)
        TOP_PAD = TOP_PAD_BASE + 0.30*(n_title_lines-1)

        if n_cb_shown == 0:
            n_cb_rows = 0
            cb_layout = {"type":"none"}
        elif n_cb_shown <= 3:
            n_cb_rows = n_cb_shown
            cb_layout = {"type":"fullwidth"}
        else:
            CB_COLS = 2 if n_cb_shown <= 6 else 3
            n_cb_rows = int(np.ceil(n_cb_shown / CB_COLS))
            cb_layout = {"type":"grid", "cols": CB_COLS}

        H_SP = H_SP_SMALL if n_cb_rows == 1 else H_SP_BIG
        H_total = TOP_PAD + H_MAIN + n_cb_rows*(H_SP + H_CB)

        fig = plt.figure(name, figsize=(W_MAIN + LEG_W, H_total))
        gs = fig.add_gridspec(
            nrows=1 + (2*n_cb_rows if n_cb_rows else 0),
            ncols=2,
            height_ratios=[H_MAIN] + sum(([H_SP, H_CB] for _ in range(n_cb_rows)), []),
            width_ratios=[W_MAIN, LEG_W],
        )
        ax = fig.add_subplot(gs[0, 0])
        ax_leg = fig.add_subplot(gs[:, 1]); ax_leg.axis("off")

        ax.grid(True, which="both")
        ax.set_xlabel(xName); ax.set_ylabel(yName)
        if title:
            fig.suptitle(title, y=1.0 - (TOP_PAD/H_total)*0.25, fontsize=9)
        bottom_pad = 0.20 if n_cb_rows <= 1 else 0.24
        fig.subplots_adjust(top=1.0 - TOP_PAD/H_total, bottom=bottom_pad, left=0.16, right=0.985)

        # X formatter
        xfmt = ScalarFormatter(useMathText=True)
        xfmt.set_powerlimits((-2, 2))
        xfmt.set_useOffset(False)
        ax.xaxis.set_major_formatter(xfmt)
        ax.get_xaxis().get_offset_text().set_visible(False)

        ux = np.unique(x[np.isfinite(x)])
        if ux.size == 1 or (np.ptp(x) <= max(1e-12, 1e-6*max(1.0, abs(np.nanmean(x))))):
            v = float(ux[0])
            delta = max(1e-3, 0.02*max(1.0, abs(v)))
            ax.set_xlim(v - delta, v + delta)
            ax.set_xticks([v])
            ax.xaxis.set_minor_locator(NullLocator())
            ax.tick_params(axis='x', pad=2.5)
        else:
            # se i valori sono "quasi" interi e con range decente → usa tick interi
            vals = x[np.isfinite(x)]
            near_int = np.allclose(vals, np.rint(vals), atol=1e-6)
            if near_int and (np.nanmax(vals) - np.nanmin(vals) >= 3):
                from matplotlib.ticker import MaxNLocator, FormatStrFormatter
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6, prune='both'))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
                ax.get_xaxis().get_offset_text().set_visible(False)
                ax.tick_params(axis='x', pad=2.5)
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.tick_params(axis='x', pad=2.5)

        # scale e griglia log più leggera
        if yscale:
            try: ax.set_yscale(yscale)
            except Exception: pass
        if xscale:
            try: ax.set_xscale(xscale)
            except Exception: pass
        if ax.get_yscale() == 'log':
            ax.grid(which='major', alpha=0.35, linewidth=0.6)
            ax.grid(which='minor', alpha=0.18, linewidth=0.4, axis='y')
            ax.grid(which='minor', visible=False, axis='x')

        # ------------------ HANDLES LEGENDA (costruzione reale) ------------------
        handles, labels = [], []

        # Edge
        if len(edge_vals) > 1:
            h, l = _section(f"{edgeColorVariableName}:"); handles.append(h); labels.append(l)
            for ev in edge_vals:
                lab = _edge_label(ev)
                col = _edge_color(ev)
                handles.append(Line2D([0],[0], linestyle='None', marker='o',
                                      markerfacecolor='none', markeredgecolor=col, markeredgewidth=1.2))
                labels.append(lab)
            h, l = _sep(); handles.append(h); labels.append(l)

        # Shapes
        if shape_vals:
            h, l = _section(f"{shape_title}:"); handles.append(h); labels.append(l)
            marker_pool = ['s','^','o','v','D','p','h','X','<','>','*','P']

            shape_to_marker = {}
            non_inf_shapes = []
            for sh in shape_vals:
                if str(sh).lower() in ('inf','infty'):
                    shape_to_marker[sh] = '.'      # T=inf → puntino
                else:
                    non_inf_shapes.append(sh)

            # assegna marker alle shape “normali”
            for i, sh in enumerate(non_inf_shapes):
                shape_to_marker[sh] = marker_pool[i % len(marker_pool)]

            # voci legenda effettive
            for sh in shape_vals:
                lab = r'$\infty$' if str(sh).lower() in ('inf','infty') else str(sh)
                handles.append(Line2D([0],[0], linestyle='None',
                                    marker=shape_to_marker[sh],
                                    color='0.35', markeredgecolor='0.35'))
                labels.append(lab)

            h, l = _sep(); handles.append(h); labels.append(l)

        # Guide lines
        guide_handles, guide_labels = [], []
        if linesAtXValueAndName:
            for val, lab, col in linesAtXValueAndName:
                guide_handles.append(Line2D([0,1],[0,0], linestyle='--', color=col))
                guide_labels.append(str(lab))
        if linesAtYValueAndName:
            for val, lab, col in linesAtYValueAndName:
                guide_handles.append(Line2D([0,1],[0,0], linestyle='--', color=col))
                guide_labels.append(str(lab))
        if guide_handles:
            h, l = _section("Guide lines:"); handles.append(h); labels.append(l)
            handles += guide_handles; labels += guide_labels
            h, l = _sep(); handles.append(h); labels.append(l)

        # ------------------ PLOT + FIT ------------------
        DEFAULT_POINT_COLOR = (fallbackPointColor if fallbackPointColor is not None
                               else plt.rcParams.get('axes.prop_cycle').by_key().get('color', ['C0'])[0])
        base_s = float(markerSize)
        norms = {sp: Normalize(*cbar_ranges[sp]) for sp in uniq_spec}
        plotted_mask = np.zeros_like(x, dtype=bool)
        fits_meta = []

        # === FIT GLOBALI che rispettano davvero fitGroupBy ===
        if fitTypes:
            # costruiamo le dimensioni da considerare
            dims = []
            if 'shape' in fitGroupBy:
                dims.append(('shape', shape_key))
            if 'edge' in fitGroupBy:
                dims.append(('edge', edge_key))
            if 'spec' in fitGroupBy:
                dims.append(('spec', spec_key))
            if ('color' in fitGroupBy) and (not suppress_all_cbar):
                dims.append(('color', arrayForColorCoordinate))

            groups_masks = []
            groups_keys  = []

            if not dims:
                groups_masks = [np.ones_like(x, dtype=bool)]
                groups_keys  = [tuple()]
            else:
                uniq_lists = []
                for name, arr in dims:
                    uniq_lists.append(np.unique(arr))

                def _recurse(i, cur_mask, cur_vals):
                    if i == len(dims):
                        if np.count_nonzero(cur_mask) >= 3:
                            groups_masks.append(cur_mask.copy())
                            groups_keys.append(tuple(cur_vals))
                        return
                    name, arr = dims[i]
                    for v in uniq_lists[i]:
                        _recurse(i+1, cur_mask & (arr == v), cur_vals + [(name, v)])

                _recurse(0, np.ones_like(x, dtype=bool), [])

            for gmask, gkey in zip(groups_masks, groups_keys):
                xx = x[gmask]; yy = y[gmask]
                if xx.size < 3 or not np.all(np.isfinite(xx)) or not np.all(np.isfinite(yy)):
                    continue
                xt = np.linspace(float(np.nanmin(xx)), float(np.nanmax(xx)), 256)
                for fname in fitTypes:
                    fitter = FITTERS.get(fname)
                    if fitter is None: 
                        continue
                    try:
                        res = fitter(xx, yy)
                        if fname == "linear":
                            c, m = res["params"]["c"], res["params"]["m"]
                            ax.plot(xt, c + m*xt, linestyle='--', linewidth=1.0, color='0.25', zorder=1)
                        elif fname == "quadratic":
                            c, a = res["params"]["c"], res["params"]["a"]
                            ax.plot(xt, c + a*xt*xt, linestyle='--', linewidth=1.0, color='0.25', zorder=1)
                        elif fname == "expo":
                            c, m, s = res["params"]["c"], res["params"]["m"], res["params"]["s"]
                            ax.plot(xt, c*(1.0 - np.exp(-(xt-s)*m)), linestyle='--', linewidth=1.0, color='0.25', zorder=1)

                        # meta per legenda "Fits:"
                        fits_meta.append({
                            "group": {k: (str(v) if not isinstance(v, (float,int)) else v) for k,v in gkey},
                            "fit": fname, "eq": res["eq"],
                            "params": {k: float(v) for k, v in res["params"].items()},
                            "stderr": {k: float(v) for k, v in res["stderr"].items()},
                            "ignoring": [d for d in ('shape','edge','spec','color') if d not in fitGroupBy]
                        })
                    except Exception:
                        continue

        # === MAIN SCATTER (punti + barre errore + connect_points) ===
        for sh in shape_vals:
            mkr = shape_to_marker.get(sh, 'o')       # '.' per T=inf, 's' primo normale, ecc.
            shape_mask = (shape_key == str(sh))

            for ev in edge_vals:
                edge_mask = (markerEdgeVariable == ev)
                # se c'è un solo edge e hideEdgeWhenSingle=True, niente contorno
                edge_col = 'none' if (single_edge_value is not None and hideEdgeWhenSingle) else _edge_color(ev)

                # marker più piccolo per T=inf (puntini meno “importanti”)
                is_inf = str(sh).lower() in ('inf', 'infty')
                this_s = base_s * (0.55 if is_inf else 1.0)

                for sp in uniq_spec:
                    spec_mask = (spec_key == sp)
                    mask_base = shape_mask & edge_mask & spec_mask
                    if not np.any(mask_base):
                        continue

                    vals = arrayForColorCoordinate[mask_base]
                    xx_all, yy_all = x[mask_base], y[mask_base]
                    yyerr_all = None if yerr is None else yerr[mask_base]

                    # colori: colormap vera se cbar presente, altrimenti colore di fallback
                    if suppress_all_cbar:
                        facecols = DEFAULT_POINT_COLOR
                        rgba_cols = None
                    else:
                        cm = cmaps[sp]; norm = norms[sp]
                        cols = np.empty((vals.size, 4), dtype=float)
                        mfin = np.isfinite(vals)
                        if np.any(mfin):
                            cols[mfin] = cm(norm(vals[mfin]))
                        if np.any(~mfin):
                            cols[~mfin] = mcolors.to_rgba(DEFAULT_POINT_COLOR)
                        facecols = None
                        rgba_cols = cols

                    # ordina per x (utile anche per connect_points)
                    idx = np.argsort(xx_all)
                    xx_all, yy_all = xx_all[idx], yy_all[idx]
                    if yyerr_all is not None: yyerr_all = yyerr_all[idx]
                    if rgba_cols is not None: rgba_cols = rgba_cols[idx]

                    # poligonale di collegamento, se richiesta
                    if connect_points:
                        ax.plot(xx_all, yy_all, color='0.6', linewidth=0.7, marker=' ', zorder=2.0)

                    # barre d'errore con colore schiarito coerente al punto
                    if yyerr_all is not None:
                        segs, colors = [], []
                        for i in range(xx_all.size):
                            if not np.isfinite(yyerr_all[i]): continue
                            y0 = yy_all[i] - yyerr_all[i]
                            y1 = yy_all[i] + yyerr_all[i]
                            segs.append([(xx_all[i], y0), (xx_all[i], y1)])
                            base_rgba = (mcolors.to_rgba(facecols) if suppress_all_cbar
                                         else tuple(rgba_cols[i]))
                            colors.append(_lighten_rgba(base_rgba, alpha=0.8, w=0.35))
                        if segs:
                            lc = LineCollection(segs, colors=colors, linewidths=1.0, zorder=2.1)
                            ax.add_collection(lc)

                    # scatter vero e proprio
                    if suppress_all_cbar:
                        ax.scatter(xx_all, yy_all, color=facecols, marker=mkr,
                                   edgecolors=edge_col, linewidths=1.0, s=this_s, zorder=2.2)
                    else:
                        ax.scatter(xx_all, yy_all, c=rgba_cols, marker=mkr,
                                   edgecolors=edge_col, linewidths=1.0, s=this_s, zorder=2.2)

                    # usato per eventuale "Summary:"
                    plotted_mask[np.where(mask_base)[0][idx]] = True


       # ------------------ UNUSED (compat: additionalMarkerTypes_Unused) ------------------
        if additionalMarkerTypes_Unused is not None:
            for additional in additionalMarkerTypes_Unused:
                # atteso: [X, Y, [spec_array, colorval_array], (opzionale) shape_array]
                if len(additional) < 3:
                    continue

                addX = np.asarray(additional[0], dtype=float)
                addY = np.asarray(additional[1], dtype=float)

                spec_and_color = np.asarray(additional[2], dtype=object)
                if spec_and_color.ndim == 2:
                    spec_and_color = spec_and_color.T
                if spec_and_color.size == 0:
                    continue

                # shape opzionale passata dal chiamante (nel tuo uso: tutto "inf")
                shape_arr = None
                if len(additional) >= 4:
                    shape_arr = np.asarray(additional[3], dtype=object)
                    if shape_arr.size == addX.size:
                        pass  # per ora la usiamo solo per ricavare il marker comune
                    elif shape_arr.size > 0:
                        shape_arr = np.repeat(shape_arr.flat[0], addX.size)
                    else:
                        shape_arr = None

                # marker unused = marker della shape "inf" (se esiste) altrimenti fallback
                # NB: shape_to_marker è definito sopra nella sezione "Shapes"
                mkr_unused = '.'

                common_kwargs = dict(
                    marker=mkr_unused, s=markerSize*0.75, alpha=0.18,
                    edgecolors='none', linewidths=0.0, zorder=0.5
                )

                # ordina per x per tracce ordinate (coerente col resto)
                add_sort = np.argsort(addX)
                addX = addX[add_sort]; addY = addY[add_sort]
                spec_and_color = spec_and_color[add_sort]

                spec_arr  = np.asarray(spec_and_color[:,0]).astype(str)
                color_arr = np.asarray(spec_and_color[:,1], dtype=float)


                if suppress_all_cbar:
                    face = fallbackPointColor or plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0'])[0]
                    ax.scatter(addX, addY, color=face, **common_kwargs)
                else:
                    rgba = np.empty((addX.size, 4))
                    for i in range(addX.size):
                        sp = spec_arr[i] if spec_arr[i] in cmaps else uniq_spec[0]
                        nm = Normalize(*cbar_ranges[sp])
                        rgba[i] = cmaps[sp](nm(color_arr[i]))
                    ax.scatter(addX, addY, c=rgba, **common_kwargs)

        # ------------------ CURVE CONTINUE / TEORIA ------------------
        if theoreticalX is not None and theoreticalY is not None:
            tx = np.asarray(theoreticalX); ty = np.asarray(theoreticalY)
            ax.plot(tx, ty, linestyle='--', marker=' ', label='Theory', color='0.3', zorder=1)

        if functionsToPlotContinuously is not None:
            funs, masks = functionsToPlotContinuously
            for f_fun, gmask in zip(funs, masks):
                sel = np.asarray(gmask, dtype=bool)
                if sel.shape != x.shape or not np.any(sel):
                    continue
                xx = np.linspace(float(np.nanmin(x[sel])), float(np.nanmax(x[sel])), 512)
                yy = np.array([f_fun(v) for v in xx])
                ax.plot(xx, yy, color='0.2', linewidth=1.1, marker=' ', zorder=1)

        if linesAtXValueAndName is not None:
            for val, lab, col in linesAtXValueAndName:
                ax.axvline(float(val), color=col, linestyle='dashed', linewidth=1.0, zorder=0)
        if linesAtYValueAndName is not None:
            for val, lab, col in linesAtYValueAndName:
                ax.axhline(float(val), color=col, linestyle='dashed', linewidth=1.0, zorder=0)

        # ------------------ COLORBAR ------------------
        fmt = ScalarFormatter(useMathText=True); fmt.set_powerlimits((-2,2)); fmt.set_useOffset(False)
        cb_var_label = colorCoordinateVariableName.strip() if isinstance(colorCoordinateVariableName, str) else ""
        if not cb_var_label:
            cb_var_label = "color"
            warnings.append("colorCoordinateVariableName empty; using fallback label 'color'.")

        if n_cb_shown:
            if cb_layout["type"] == "fullwidth":
                row_i = 1
                for sp in uniq_spec:
                    ax_cb = fig.add_subplot(gs[row_i + 1, 0]); row_i += 2
                    vmin, vmax = cbar_ranges[sp]
                    cm = cmaps[sp]; norm = Normalize(vmin=vmin, vmax=vmax)
                    sm = ScalarMappable(cmap=cm, norm=norm); sm.set_array([])
                    label = cb_var_label
                    cbar = fig.colorbar(sm, cax=ax_cb, orientation='horizontal')
                    cbar.ax.xaxis.set_ticks_position('top')
                    cbar.ax.tick_params(axis='x', length=2.5, width=0.8, pad=0.6)
                    cbar.outline.set_linewidth(0.6)
                    if label:
                        cbar.set_label(label, labelpad=2.0)
                        cbar.ax.xaxis.set_label_position('bottom')
                    cbar.formatter = fmt
                    if dynamicalTicksForColorbars:
                        vals = arrayForColorCoordinate[spec_key == sp]
                        u = np.unique(vals[np.isfinite(vals)])
                        cbar.set_ticks(u if (u.size and u.size <= 8) else np.linspace(vmin, vmax, 5))
                    else:
                        cbar.set_ticks(np.linspace(vmin, vmax, 5))
                    dec = _cb_decimals(vmin, vmax)
                    cbar.formatter = FormatStrFormatter(f"%.{dec}f")
                    cbar.update_ticks()
                    cbar.ax.xaxis.get_offset_text().set_visible(False)
            else:
                cols = cb_layout["cols"]
                gs_cb = GridSpecFromSubplotSpec(
                    2*n_cb_rows, cols, subplot_spec=gs[1:, 0],
                    height_ratios=sum(([H_SP, H_CB] for _ in range(n_cb_rows)), [])
                )
                i = 0
                for sp in uniq_spec:
                    r, c = divmod(i, cols)
                    ax_cb = fig.add_subplot(gs_cb[2*r + 1, c]); i += 1
                    vmin, vmax = cbar_ranges[sp]
                    cm = cmaps[sp]; norm = Normalize(vmin=vmin, vmax=vmax)
                    sm = ScalarMappable(cmap=cm, norm=norm); sm.set_array([])
                    parts = [cb_var_label]
                    if colorMapSpecifierName: parts.append(f"{colorMapSpecifierName}={sp}")
                    label = ", ".join(parts)
                    cbar = fig.colorbar(sm, cax=ax_cb, orientation='horizontal')
                    cbar.ax.xaxis.set_ticks_position('top')
                    cbar.ax.tick_params(axis='x', length=2.5, width=0.8, pad=0.6)
                    cbar.outline.set_linewidth(0.6)
                    if label:
                        cbar.set_label(label, labelpad=2.0)
                        cbar.ax.xaxis.set_label_position('bottom')
                    cbar.formatter = fmt
                    if dynamicalTicksForColorbars:
                        vals = arrayForColorCoordinate[spec_key == sp]
                        u = np.unique(vals[np.isfinite(vals)])
                        cbar.set_ticks(u if (u.size and u.size <= 8) else np.linspace(vmin, vmax, 5))
                    else:
                        cbar.set_ticks(np.linspace(vmin, vmax, 5))
                    dec = _cb_decimals(vmin, vmax)
                    cbar.formatter = FormatStrFormatter(f"%.{dec}f")
                    cbar.update_ticks()
                    cbar.ax.xaxis.get_offset_text().set_visible(False)

        # ------------------ INFO-LINE ------------------
        infos = []
        if suppress_all_cbar:
            fin = np.isfinite(arrayForColorCoordinate)
            if np.any(fin):
                v0 = float(arrayForColorCoordinate[fin][0])
                infos.append(f"At {cb_var_label}={v0:g}")
            if showSpecInInfoWhenNoCbar and colorMapSpecifierName and (uniq_spec.size == 1):
                infos.append(f"{colorMapSpecifierName}={uniq_spec[0]}")
        if len(uniq_edge_vals) == 1:
            infos.append(f"{edgeColorVariableName}: {_edge_label(uniq_edge_vals[0])}")
        if (n_cb_shown == 1) and colorMapSpecifierName and (uniq_spec.size == 1):
            infos.append(f"{colorMapSpecifierName}: {uniq_spec[0]}")
        if infos:
            fs = max(6.0, float(base_fs) - 3.0)
            info_y = 1.0 - (TOP_PAD / H_total) * 0.85
            fig.text(0.02, info_y, f"{' | '.join(infos)}",
                     ha='left', va='center', fontsize=fs, style='italic', color='0.25', alpha=0.9)

        # ------------------ SEZIONE "Fits:" REALE ------------------
        if fitTypes and len(fits_meta):
            h, l = _section("Fits:"); handles.append(h); labels.append(l)
            ORDER = {"linear":("m","c"), "quadratic":("a","c"), "expo":("m","c","s")}
            for fm in fits_meta:
                eq_line = fm["eq"]
                ign = fm.get("ignoring", [])
                if ign: eq_line += f" (ignoring {DELIM.join(ign)})"
                handles.append(Line2D([0,1],[0,0], linestyle='--', color='0.25')); labels.append(eq_line)
                ps, se = fm["params"], fm["stderr"]
                for k in ORDER.get(fm["fit"], ps.keys()):
                    v = ps.get(k, np.nan); e = se.get(k, np.nan)
                    handles.append(Line2D([], [], linestyle='None')); labels.append(f"{k}={_fmt_pm(v, e)}")
            h, l = _sep(); handles.append(h); labels.append(l)

        # ------------------ SEZIONE "Summary:" ------------------
        if countUniqueOf:
            parts = []
            for nm in countUniqueOf:
                arr = seriesByName.get(nm)
                if arr is None:
                    warnings.append(f"countUniqueOf: '{nm}' not provided in seriesByName.")
                    continue
                u = np.unique(arr[plotted_mask])
                parts.append(f"#{nm}={u.size}")
            if parts:
                h, l = _section("Summary:"); handles.append(h); labels.append(l)
                handles.append(Line2D([], [], linestyle='None')); labels.append(", ".join(parts))
                h, l = _sep(); handles.append(h); labels.append(l)

        while labels and labels[-1].strip() == "":
            labels.pop(); handles.pop()

        if handles:
            leg = ax_leg.legend(
                handles, labels, loc='upper left',
                frameon=True, fancybox=True, framealpha=1.0,
                borderpad=0.35, labelspacing=0.20,
                handlelength=1.2, handletextpad=0.5,
                ncol=max(1, ncol_est), columnspacing=0.8,
                prop={'size': leg_fs}
            )
            leg.get_frame().set_edgecolor('0.7'); leg.get_frame().set_linewidth(0.8)

        # ------------------ META ------------------
        if x.size < 3:
            return None, None, {"empty": True}

        meta = {
            "n_colorbars": int(n_cb_shown),
            "n_colorbar_rows": int(n_cb_rows),
            "colorbar_layout": cb_layout,
            "suppress_all_cbar": bool(suppress_all_cbar),
            "cmap_names": {str(sp): (getattr(cmaps[sp], "name", "ListedColormap")) for sp in uniq_spec},
            "cbar_ranges": {str(sp): cbar_ranges[sp] for sp in uniq_spec},
            "single_edge_value": None if len(uniq_edge_vals) != 1 else str(_edge_label(uniq_edge_vals[0])),
            "constants": {"W_MAIN": W_MAIN, "H_MAIN": H_MAIN, "H_CB": H_CB,
                          "H_SP": H_SP, "TOP_PAD": TOP_PAD_BASE, "LEG_W": LEG_W},
            "legend_outside": True,
            "legend_ncol": int(ncol_est),
            "xlim": tuple(ax.get_xlim()), "ylim": tuple(ax.get_ylim()),
            "xscale": xscale or "linear", "yscale": yscale or "linear",
            "x_stats": {"min": float(np.nanmin(x)), "max": float(np.nanmax(x)), "ptp": float(np.ptp(x)), "unique": int(ux.size)},
            "fits": fits_meta,
            "fitGroupBy": fitGroupBy,
            "warnings": warnings,
            "nGraphs": nGraphs,
        }
        return fig, ax, meta
