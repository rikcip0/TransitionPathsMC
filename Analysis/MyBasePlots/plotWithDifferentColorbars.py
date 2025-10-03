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
    from MyBasePlots.FigCore.utils_style import auto_style

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
    H_SP_SMALL = 0.55
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
    with auto_style():
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
        
        markerEdgeVariable  = markerEdgeVariable.astype(str)
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
                #if ign: eq_line += f" (ignoring {DELIM.join(ign)})"
                #legend_lines.append(('item', eq_line))
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
                        ax1.scatter(x[condition], y[condition], color=color, marker=marker, edgecolor=edgeColorPerVar[ext], linewidths=2, s=s)
                        ax1.errorbar(x[condition], y[condition], yerr=yerr[condition], color=color, fmt= ' ', marker='')
                        plottedYs.extend(y[condition])
            
            if fittingOverDifferentEdges is False:
                fitCondition =  [np.array_equal(t,variable) for variable in markerShapeVariable]
                xToPlot=np.linspace(np.nanmin(x[fitCondition]), np.nanmax(x[fitCondition]), 100)
                if len(np.unique(x[fitCondition]))>2:
                    if 'linear' in fitTypes:
                        popt, pcov = curve_fit(lambda x, c, m:  c+(x*m), x[fitCondition], y[fitCondition])
                        c = popt[0]
                        m= popt[1]
                        mErr = np.sqrt(pcov[1,1])
                        plt.plot(xToPlot, m*xToPlot+c, linestyle='--', marker='', color=color)
                        plt.plot([], [], label=f'c+mT', linestyle='--', marker=marker, color=color)
                        plt.plot([], [], label=f'c={c:.3g} ' + r'm'+f'={m:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color=color)
                        plt.plot([], [], label=f' ', linestyle=' ', marker=' ')
                        fitResult= m, c, mErr
                        
                    if 'quadratic' in fitTypes:
                        popt, pcov = curve_fit(lambda x, c, a:  c+(x*x*a), x[fitCondition], y[fitCondition])
                        c = popt[0]
                        a= popt[1]
                        mErr =np.sqrt(pcov[1,1])
                        plt.plot([], [], label=r'c+aT^{2}', linestyle='--', marker=marker, color=color)
                        plt.plot(xToPlot, a*xToPlot**2.+c, linestyle='--', marker='', color=color)
                        plt.plot([], [], label=f'c={c:.3g} ' + r'm'+f'={a:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color=color)
                        fitResult= m, c, mErr
                        
        if fittingOverDifferentEdges is True:
            fitCondition =  [np.array_equal(t,variable) for variable in markerShapeVariable]
            xToPlot=np.linspace(np.nanmin(x[fitCondition]), np.nanmax(x[fitCondition]), 100)
            if len(np.unique(x[fitCondition]))>2:
                if 'linear' in fitTypes:
                    popt, pcov = curve_fit(lambda x, c, m:  c+(x*m), x[fitCondition], y[fitCondition])
                    c = popt[0]
                    m= popt[1]
                    mErr = np.sqrt(pcov[1,1])
                    plt.plot(xToPlot, m*xToPlot+c, linestyle='--', marker='', color='grey')
                    plt.plot([], [], label=f'c+mT', linestyle='--', marker=marker, color='grey')
                    plt.plot([], [], label=f'c={c:.3g} ' + r'm'+f'={m:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color='grey')
                    plt.plot([], [], label=f' ', linestyle=' ', marker=' ')
                    fitResult= m, c, mErr
                if 'quadratic' in fitTypes:
                    popt, pcov = curve_fit(lambda x, c, a:  c+(x*x*a), x[fitCondition], y[fitCondition])
                    c = popt[0]
                    a= popt[1]
                    mErr = np.sqrt(pcov[1,1])
                    plt.plot([], [], label=r'c+aT^{2}', linestyle='--', marker=marker, color='grey')
                    plt.plot(xToPlot, a*xToPlot**2.+c, linestyle='--', marker='', color='grey')
                    plt.plot([], [], label=f'c={c:.3g} ' + r'm'+f'={a:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color='grey')
                    fitResult= m, c, mErr

                    
    if fittingOverDifferentShapes is True:
        xToPlot=np.linspace(np.nanmin(x), np.nanmax(x), 100)
        if len(np.unique(x))>1:
            if 'expo' in fitTypes:
                    popt, pcov = curve_fit(lambda x, c, m,s:  c*(1.-np.exp(-(x-s)*m)), x, y,p0=[y[-1],(y[1]-y[0])/(x[1]-x[0])/y[-1],0], maxfev=100000)
                    c = popt[0]
                    m= popt[1] 
                    s= popt[2] 
                    mErr = np.sqrt(pcov[1,1])
                    #s =popt[2]
                    plt.plot(xToPlot, c*(1.-np.exp(-(xToPlot-s)*m)), linestyle='--', marker='', color='grey')
                    plt.plot([], [], label=f'exp0', linestyle='--', marker=marker, color='grey')
                    plt.plot([], [], label=f's={s:.3g} ' +f'c={c:.3g} ' + r'm'+f'={m:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color='grey')
                    plt.plot([], [], label=f' ', linestyle=' ', marker=' ')
                    fitResult= m, c, mErr
            if 'linear' in fitTypes:
                    popt, pcov = curve_fit(lambda x, c, m:  c+(x*m), x, y)
                    c = popt[0]
                    m= popt[1]
                    mErr = np.sqrt(pcov[1,1])
                    plt.plot(xToPlot, m*xToPlot+c, linestyle='--', marker='', color='grey')
                    plt.plot([], [], label=f'c+mT', linestyle='--', marker=marker, color='grey')
                    plt.plot([], [], label=f'c={c:.3g} ' + r'm'+f'={m:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color='grey')
                    plt.plot([], [], label=f' ', linestyle=' ', marker=' ')
                    fitResult= m, c, mErr
            if 'expo2' in fitTypes:
                def model(x, ty, delta,s):
                    xt = ty / delta**2
                    return xt/(xt - ty) + (xt*ty)/(ty - xt)*(x-s) + (xt/(ty - xt))*np.exp(-(x-s)*ty)
                def model2(x, a, b,c,delta):
                    d = c / delta**2
                    return a+b(xToPlot)+c*np.exp(-(xToPlot)*d)
                p0=[(y[-1]-y[-2])/(x[-1]-x[-2]),(y[-1]-y[-2])/(x[-1]-x[-2]),(y[-1]-y[-2])/(x[-1]-x[-2]),5]
                popt, pcov = curve_fit(lambda x, a, b,c,delta:  model2(x,a,b,c,delta), x, y,p0=p0, maxfev=5000000, bounds=([0.,1.01,0],[np.inf,np.inf,10]))
                a = popt[0]
                ty = popt[0]
                b= popt[1] 
                c=popt[2]
                delta=popt[3]
                d= c/(delta*delta) 
                mErr = np.sqrt(pcov[1,1])
                plt.plot(xToPlot, model2(xToPlot,a,b,c,delta), linestyle='--', marker='x', color='grey')
                plt.plot([], [], label=f'exp02', linestyle='--', marker=marker, color='grey')
                plt.plot([], [], label=f's={s:.3g} ' +f'ty={c:.3g} ' + r'xt'+f'={xt:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color='grey')
                plt.plot([], [], label=f' ', linestyle=' ', marker=' ')
                fitResult= m, c, mErr
            if 'quadratic' in fitTypes:
                    popt, pcov = curve_fit(lambda x, c,a:  c+(a*x**2.), x, y)
                    c = popt[0]
                    a= popt[1]
                    mErr = np.sqrt(pcov[1,1])
                    plt.plot([], [], label=r'$c+aT^{2}$', linestyle='--', marker=marker, color='grey')
                    plt.plot(xToPlot, a*xToPlot**2.+c, linestyle='--', marker='', color='grey')
                    plt.plot([], [], label=f'c={c:.3g} ' + r'm'+f'={a:.3g}'+'±'+f'{mErr:.3g}', linestyle='--', marker=marker, color='grey')
                    fitResult= a, c, mErr
            if 'mix' in fitTypes:
                def mix(x, k1p,k2p):
                    return (k1p/(k2p-k1p))*(np.exp(-k2p*x)-1.+k2p*x)
                k1Test=(y[-1]-y[-2])/(x[-1]-x[-2])
                popt, pcov = curve_fit(mix, x, y,p0=[k1Test,k1Test*100.],method='trf' , max_nfev=10000)
                c=0.
                k1= popt[0]
                k2= popt[1]
                k1Err = np.sqrt(pcov[1,1])
                plt.plot(xToPlot, mix(xToPlot,k1,k2), linestyle='--', marker='', color='grey')
                plt.plot([], [], label=f'c={c:.3g} '+f'k1={k1:.3g} ' + r'k2'+f'={k2:.3g}', linestyle='--', marker=marker, color='grey')
                fitResult= m, c, mErr
 
    if additionalMarkerTypes_Unused is not None:
        for additionalMarkerType in additionalMarkerTypes_Unused:
            additional_X = np.asarray(additionalMarkerType[0])
            additional_Y = np.asarray(additionalMarkerType[1])
            additional_correspBetaOfExAndQif = np.transpose(np.asarray(additionalMarkerType[2]))
            if len(additional_correspBetaOfExAndQif)==0:
                continue
            additional_correspBetaOfExAndQif[1]=additional_correspBetaOfExAndQif[1].astype(np.float64)
            additionalXSort = np.argsort(additional_X)

            additional_X = additional_X[additionalXSort]
            additional_Y = additional_Y[additionalXSort]
            additional_correspBetaOfExAndQif =additional_correspBetaOfExAndQif[additionalXSort]

            marker = "."
            for BetaOfExAndQif in additional_correspBetaOfExAndQif:
                    if BetaOfExAndQif[0] is None:
                        continue
                    BetaOfExAndQif[0] = str(BetaOfExAndQif[0])
                    condition = np.all(additional_correspBetaOfExAndQif == BetaOfExAndQif, axis=1)
                    if len(additional_X[condition]) == 0:
                        continue
                    if keyIsNan:
                        BetaOfExAndQif[0]='nan'

                    color = cmaps[str(BetaOfExAndQif[0])](norm(BetaOfExAndQif[1]))
                    ax1.scatter(additional_X[condition], additional_Y[condition], color=color, marker=marker, s=40, alpha=0.01)

    plottedYs = np.asarray(plottedYs)
    if yscale!='' and len(plottedYs[plottedYs>0])>0:
        #print(name, len(plottedYs[plottedYs>0]))
        plt.yscale(yscale)
    if xscale!='' and len(x[x>0])>0:
         plt.xscale(xscale)

    if theoreticalX is not None:
        plt.plot([],[], label=f"          ", marker=None, color="None")
        if "betTentative" in name:
            f= theoreticalX<=1
            theoreticalX = theoreticalX[f]
            theoreticalY = theoreticalY[f] 
        plt.plot(theoreticalX, theoreticalY, linestyle='--', marker=' ', label='Theory')
    
    if linesAtXValueAndName is not None:
        plt.plot([],[], label=f"          ", marker=None, color="None")
        for l in linesAtXValueAndName :
            l_Value, l_Name, l_color = l
            plt.axvline(l_Value, color=l_color, linestyle='dashed', marker=' ', linewidth=1)
            plt.plot([],[], label=f"{l_Name}", linestyle='dashed', marker=' ', color=l_color)
    
    if linesAtYValueAndName is not None:
        plt.plot([],[], label=f"          ", marker=None, color="None")
        for l in linesAtYValueAndName :
            l_Value, l_Name, l_color = l
            plt.axhline(l_Value, color=l_color, linestyle='dashed', marker=' ', linewidth=1)
            plt.plot([],[], label=f"{l_Name}", linestyle='dashed', marker=' ', color=l_color)
    
    handles, labels = ax1.get_legend_handles_labels()
    """
    fig.legend(handles, labels,
        bbox_to_anchor=(1.05, 1.0),
        loc='upper left',
        borderaxespad=0.,
        bbox_transform=ax1.transAxes)
    """

    theory_handles_labels = [(h, l) for h, l in zip(handles, labels) if l == 'Theory']
    if theory_handles_labels:
        handles, labels = zip(*theory_handles_labels)
        ax1.legend(handles, labels, loc='upper left')
    
    sm = ['']*nColorbars
    uniqueColorMapsSpecifiers=[]
    for i, thisColorMapSpecifier in enumerate(uniqueColorMapsSpecifiers):
        # Crea l'asse per la colorbar (non serve l'asse extra ax_cb, se non lo usi per altro)
        ax_colorbar = fig.add_subplot(gs[2 + 2 * i, 0])
        
        # Seleziona la mappa per questo gruppo
        subset = arrayForColorCoordinate[colorMapSpecifier == thisColorMapSpecifier]
        norm = Normalize(vmin=np.min(subset), vmax=np.max(subset))
        sm[i] = ScalarMappable(cmap=cmaps[thisColorMapSpecifier], norm=norm)
        sm[i].set_array(subset)
        
        # Crea la colorbar orizzontale sull'asse dedicato
        cbar = plt.colorbar(sm[i], orientation='horizontal', cax=ax_colorbar, pad=0.0)

        if dynamicalTicksForColorbars:
            currentTicks = np.sort(np.unique(subset))
            desired_ticks = np.array([float(f'{tick}') for j, tick in enumerate(currentTicks) if j == 0 or np.abs(tick - currentTicks[j-1]) > np.mean(np.diff(currentTicks))/6.])
            cbar.set_ticks(desired_ticks)
            cbar.ax.xaxis.set_major_formatter(ScalarFormatter())
        # Imposta le etichette dei tick (solo per l'asse che gestiremo)
        #cbar.ax.set_xticklabels([f"{tick}" for tick in desired_ticks])
        
        # Forza la posizione dei tick: solo in alto
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(axis='x', which='both', bottom=False, top=True,
                            labelbottom=False, labeltop=True)
        
        # Disattiva ogni tick e ticklabel sugli assi verticali (sinistra/destra)
        cbar.ax.tick_params(axis='y', which='both', left=False, right=False,
                            labelleft=False, labelright=False)
        
        # Se per caso appaiono ancora ticklabel "extra" (ad esempio sul fondo),
        # forzali invisibili:
        for label in cbar.ax.get_xticklabels():
            # Se il centro del label (y) è al di sotto di 0.5 (la metà dell'asse normalizzato),
            # lo nascondiamo (questo è un hack; di solito labelbottom=False è sufficiente)
            if label.get_position()[1] < 0.5:
                label.set_visible(False)
        
        # Imposta il label della colorbar
        if i+1 != nColorbars:
            cbar.set_label(colorMapSpecifierName + '=' + f"{thisColorMapSpecifier}", labelpad=6)
        else:
            if nColorbars == 1 and (str(thisColorMapSpecifier) == 'nan' or thisColorMapSpecifier is None):
                cbar.set_label(colorCoordinateVariableName, labelpad=6)
            else:
                cbar.set_label(colorCoordinateVariableName + ', ' + colorMapSpecifierName + '=' + f"{thisColorMapSpecifier}", labelpad=6)
        
        # Posiziona manualmente il label in basso (modifica il valore di y se serve)
        cbar.ax.xaxis.set_label_coords(0.5, -1.)

    if nGraphs is not None:
        x_min, x_max = ax1.get_xlim()
        y_min, y_max = ax1.get_ylim()
        text_x = x_max + 0.00 * (x_max - x_min)
        if yscale =='log' and (y_min >= 0 and y_max >= 0):
            text_y = y_max * 10 ** (0.04 * (np.log10(y_max) - np.log10(y_min)))
        else:
            text_y = y_max + 0.04 * (y_max - y_min)  
        ax1.text(text_x, text_y,  f"Different graphs: {nGraphs}", fontsize=11, color='black', ha='right', va='top')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1, hspace=0.3)
    return ax1, fitResult