# app_phase1_qa.py (refactored)
import streamlit as st
import pandas as pd
import numpy as np
import io
from io import BytesIO

st.set_page_config(page_title="Medical QA Tool", layout="wide")

st.title("Medical Machine QA Tool – SPEC vs REC comparison")

# --- File Upload ---
spec_file = st.sidebar.file_uploader("Upload SPEC file (CSV / XLSX)")
record_file = st.sidebar.file_uploader("Upload Record file (CSV / XLSX)")

st.sidebar.info("This tool compares baseline SPEC (target) vs irradiated RECORD logs. No QA module selection is required.")

# delimiter controls per-file
delim_options = ["Auto", "Comma (,)", "Semicolon (;)", "Tab (\t)", "Pipe (|)"]
spec_delim_choice = st.sidebar.selectbox("SPEC delimiter", delim_options, index=0)
rec_delim_choice = st.sidebar.selectbox("Record delimiter", delim_options, index=0)
skip_bad_lines = st.sidebar.checkbox("Skip malformed lines (salvage file)", value=False)

def _map_choice_to_delim(choice: str):
    if choice == "Comma (,)":
        return ','
    if choice == "Semicolon (;)":
        return ';'
    if choice == "Tab (\t)":
        return '\t'
    if choice == "Pipe (|)":
        return '|'
    return None

spec_forced_delim = _map_choice_to_delim(spec_delim_choice)
rec_forced_delim = _map_choice_to_delim(rec_delim_choice)

# tolerances / thresholds
st.sidebar.header('Tolerance settings')
dose_tol_pct = st.sidebar.number_input('Dose tolerance (%)', min_value=0.0, value=5.0)
pos_tol_mm = st.sidebar.number_input('Position tolerance (mm)', min_value=0.0, value=1.0)
beam_current_max = st.sidebar.number_input('Beam current max (units)', min_value=0.0, value=6000.0)

from app_utils import normalize_uploaded_df, infer_column_mapping


def _safe_read_csv(f, forced_delim=None, skip_bad_lines=False):
    """Robust CSV/TSV/Excel reader that:
      - supports files with leading commented metadata lines (starting with '#')
      - recognizes commented header lines like "#COL1,COL2,..." and uses them as column names
      - tries pandas fast path, python engine sniffing, and a csv.reader fallback that pads/truncates rows
      - returns (df, preview_dict) where preview_dict contains preview_lines, field_counts, used_delim, header_count, mismatches, skipped_lines
    """
    if f is None:
        return None, {'preview_lines': [], 'field_counts': [], 'used_delim': None, 'header_count': None, 'mismatches': [], 'skipped_lines': []}

    # read raw bytes
    try:
        try:
            f.seek(0)
        except Exception:
            pass
        raw = f.read()
    except Exception as e:
        st.error(f"Failed to read uploaded file bytes: {e}")
        return None, {'preview_lines': [], 'field_counts': [], 'used_delim': None, 'header_count': None, 'mismatches': [], 'skipped_lines': []}

    name = getattr(f, 'name', '') or ''
    lower = name.lower()
    if lower.endswith(('.xls', '.xlsx')):
        try:
            from io import BytesIO
            return pd.read_excel(BytesIO(raw)), {'preview_lines': [], 'field_counts': [], 'used_delim': None, 'header_count': None, 'mismatches': [], 'skipped_lines': []}
        except Exception as e:
            st.error(f"Failed to read Excel file: {e}")
            return None, {'preview_lines': [], 'field_counts': [], 'used_delim': None, 'header_count': None, 'mismatches': [], 'skipped_lines': []}

    # decode text for sniffing
    try:
        text = raw.decode('utf-8')
    except Exception:
        try:
            text = raw.decode('latin1')
        except Exception:
            text = None

    from io import StringIO
    import csv
    from collections import Counter

    preview_lines = []
    field_counts = []
    used_delim = None
    first_data_idx = 0

    if text is None:
        # Give pandas a final try with file-like fallback
        try:
            df = pd.read_csv(BytesIO(raw))
            return df, {'preview_lines': [], 'field_counts': [], 'used_delim': None, 'header_count': df.shape[1], 'mismatches': [], 'skipped_lines': []}
        except Exception as e:
            st.error(f"Failed to decode uploaded file: {e}")
            return None, {'preview_lines': [], 'field_counts': [], 'used_delim': None, 'header_count': None, 'mismatches': [], 'skipped_lines': []}

    # split into raw lines and compute preview
    raw_lines = text.splitlines()
    preview_lines = raw_lines[:200]

    # find first non-comment data line index
    first_data_idx = next((i for i, ln in enumerate(raw_lines) if ln.strip() and not ln.lstrip().startswith('#')), len(raw_lines))

    # detect commented header (e.g. "#COL1,COL2,..." ) — prefer the last commented header before data or the first
    commented_headers = [ln for ln in raw_lines if ln.lstrip().startswith('#') and (',' in ln or ';' in ln or '\t' in ln or '|' in ln)]
    header_tokens = None
    if commented_headers:
        # prefer the commented header that appears before the first data line, else take the first
        candidate = None
        for ln in commented_headers:
            idx = raw_lines.index(ln)
            if idx < first_data_idx:
                candidate = ln
                break
        if candidate is None:
            candidate = commented_headers[0]
        hdr_txt = candidate.lstrip('#').strip()
        # choose delimiter for header line
        seps = [',', ';', '\t', '|']
        if forced_delim:
            hdr_sep = forced_delim
        else:
            hdr_sep = max(seps, key=lambda s: hdr_txt.count(s))
        header_tokens = [t.strip() for t in hdr_txt.split(hdr_sep) if t is not None]

    # Build candidate data lines (non-comment)
    data_lines = [ln for ln in raw_lines if ln.strip() and not ln.lstrip().startswith('#')]

    # If there are no explicit data lines but commented header exists, try to parse the commented header as CSV header-only -> return empty df with header
    if not data_lines and header_tokens:
        df_empty = pd.DataFrame(columns=header_tokens)
        preview = {'preview_lines': preview_lines, 'field_counts': [], 'used_delim': hdr_sep if 'hdr_sep' in locals() else None, 'header_count': len(header_tokens), 'mismatches': [], 'skipped_lines': []}
        return df_empty, preview

    # Determine delimiter: forced_delim -> sniff on sample -> default ','
    if forced_delim:
        used_delim = forced_delim
    else:
        sample = '\n'.join(data_lines[:50]) if data_lines else '\n'.join(preview_lines[:50])
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';', '\t', '|'])
            used_delim = dialect.delimiter
        except Exception:
            # fallback: choose the sep that appears the most in sample
            seps = [',', ';', '\t', '|']
            counts = {s: sample.count(s) for s in seps}
            used_delim = max(seps, key=lambda s: counts[s])

    # Compute field counts from data_lines using used_delim
    field_counts = [len(ln.split(used_delim)) for ln in (data_lines[:200] if data_lines else preview_lines[:200])]

    # Try pandas fast read with comment skipping and header handling when possible
    try:
        if header_tokens:
            # feed pandas the data lines only and use header_tokens as columns
            sio = StringIO('\n'.join(data_lines))
            df_try = pd.read_csv(sio, sep=used_delim, header=None, names=header_tokens, dtype=str)
        else:
            sio = StringIO('\n'.join(data_lines))
            df_try = pd.read_csv(sio, sep=used_delim, comment='#', engine='python', header=0)
        df_try = df_try.replace({pd.NA: None})
        preview = {'preview_lines': preview_lines, 'field_counts': field_counts, 'used_delim': used_delim, 'header_count': field_counts[0] if field_counts else (df_try.shape[1] if df_try is not None else None), 'mismatches': [i+1 for i, c in enumerate(field_counts) if field_counts and c != field_counts[0]], 'skipped_lines': []}
        return df_try, preview
    except Exception as e:
        last_exc = e

    # Fallback: use csv.reader on the data_lines and build a dataframe, padding/truncating to modal width
    try:
        if not data_lines:
            st.error("No data lines found after skipping comments.")
            return None, {'preview_lines': preview_lines, 'field_counts': field_counts, 'used_delim': used_delim, 'header_count': None, 'mismatches': [], 'skipped_lines': []}

        rows = [row for row in csv.reader(data_lines, delimiter=used_delim)]
        if not rows:
            raise RuntimeError("csv.reader produced no rows")

        counts = [len(r) for r in rows]
        modal_count = Counter(counts).most_common(1)[0][0]

        # choose header: prefer header_tokens if it matches modal_count; otherwise use first row as header
        if header_tokens and len(header_tokens) >= modal_count:
            header = header_tokens[:modal_count]
            data_rows = rows
        else:
            header = rows[0][:modal_count]
            data_rows = rows[1:]

        processed = []
        for r in data_rows:
            if len(r) < modal_count:
                r = r + [None] * (modal_count - len(r))
            elif len(r) > modal_count:
                # join extras into the last column
                r = r[:modal_count - 1] + [used_delim.join(r[modal_count - 1:])]
            processed.append(r)

        df_final = pd.DataFrame(processed, columns=header)
        mismatches = [i+1 for i, c in enumerate(counts[:200]) if c != modal_count]
        preview = {'preview_lines': preview_lines, 'field_counts': counts[:200], 'used_delim': used_delim, 'header_count': modal_count, 'mismatches': mismatches, 'skipped_lines': mismatches if skip_bad_lines else []}
        return df_final, preview
    except Exception as e:
        st.error(f"Failed to parse CSV/TSV file: {e} (last: {last_exc if 'last_exc' in locals() else None})")
        preview = {'preview_lines': preview_lines, 'field_counts': field_counts, 'used_delim': used_delim, 'header_count': field_counts[0] if field_counts else None, 'mismatches': [i+1 for i, c in enumerate(field_counts) if field_counts and c != field_counts[0]], 'skipped_lines': []}
        return None, preview


def _find_column(cols, candidates):
    norm = {c.lower().replace(' ', '').replace('(', '').replace(')', '').replace('-', ''): c for c in cols}
    for cand in candidates:
        key = cand.lower().replace(' ', '').replace('(', '').replace(')', '').replace('-', '')
        if key in norm:
            return norm[key]
    # fuzzy: look for candidate token in any normalized column
    for cand in candidates:
        token = ''.join(ch for ch in cand.lower() if ch.isalnum())
        for nrm, actual in norm.items():
            if token in nrm or nrm in token:
                return actual
    return None


def compute_ic23_width(values, x=None):
    """Compute a robust FWHM (IC23-style) from a 1D signal.

    Features:
      - Optional smoothing (Savgol filter if scipy available, fallback to pandas rolling median)
      - Optional Gaussian fit (using scipy.optimize.curve_fit) for a fitted FWHM estimate
      - Linear interpolation used to find half-maximum crossing points on the (optionally smoothed) signal

    Returns a dict with keys:
      - fwhm: measured FWHM from crossings (or nan)
      - peak: peak value
      - peak_index: index of peak
      - left_pos, right_pos: crossing positions
      - fwhm_fit: optional fitted FWHM from Gaussian fit (if available)
      - used_smoothing: bool
    """
    arr = np.asarray(values, dtype=float)
    if x is None:
        x = np.arange(len(arr))
    else:
        x = np.asarray(x, dtype=float)

    # handle empty or all-nan
    if arr.size == 0 or np.all(np.isnan(arr)):
        return {'fwhm': np.nan, 'peak': np.nan, 'peak_index': None, 'left_pos': np.nan, 'right_pos': np.nan, 'fwhm_fit': np.nan, 'used_smoothing': False}

    # smoothing: try savgol (scipy) then pandas rolling median fallback
    used_smoothing = False
    arr_s = arr.copy()
    # detect plateau: if a sizable fraction of samples equal the max, avoid smoothing which blurs edges
    peak_count = np.sum(np.isclose(arr, np.nanmax(arr))) if arr.size > 0 else 0
    plateau_fraction = float(peak_count) / float(len(arr)) if len(arr) > 0 else 0.0
    try:
        # attempt to use scipy's savgol if available
        from scipy.signal import savgol_filter
        # choose a window length: odd and <= len(arr)
        wl = min(len(arr) if len(arr) % 2 == 1 else len(arr)-1, 51)
        if wl < 3:
            wl = 3
        if wl % 2 == 0:
            wl = wl-1 if wl > 3 else 3
        if plateau_fraction < 0.05:
            arr_s = savgol_filter(np.nan_to_num(arr, nan=0.0), window_length=wl, polyorder=3, mode='interp')
            used_smoothing = True
        else:
            arr_s = arr.copy()
            used_smoothing = False
    except Exception:
        try:
            import pandas as _pd
            win = min(11, max(3, len(arr)//10))
            if plateau_fraction < 0.05:
                arr_s = _pd.Series(arr).rolling(window=win, center=True, min_periods=1).median().to_numpy()
                used_smoothing = True
            else:
                arr_s = arr.copy()
                used_smoothing = False
        except Exception:
            arr_s = arr.copy()
            used_smoothing = False

    # find peak on smoothed signal
    try:
        peak_idx = int(np.nanargmax(arr_s))
    except ValueError:
        return {'fwhm': np.nan, 'peak': np.nan, 'peak_index': None, 'left_pos': np.nan, 'right_pos': np.nan, 'fwhm_fit': np.nan, 'used_smoothing': used_smoothing}
    peak_val = float(arr_s[peak_idx])
    half = peak_val / 2.0

    # helper for crossing interpolation
    def interp_cross(i1, i2, y_half, y1, y2, x1, x2):
        if y2 == y1:
            return float(x1)
        t = (y_half - y1) / (y2 - y1)
        return float(x1 + t * (x2 - x1))

    # left crossing
    left_pos = None
    for i in range(peak_idx, 0, -1):
        if np.isnan(arr_s[i]) or np.isnan(arr_s[i-1]):
            continue
        y1, y2 = arr_s[i-1], arr_s[i]
        if (y1 <= half <= y2) or (y1 >= half >= y2):
            left_pos = interp_cross(i-1, i, half, y1, y2, x[i-1], x[i])
            break

    # right crossing
    right_pos = None
    for i in range(peak_idx, len(arr_s)-1):
        if np.isnan(arr_s[i]) or np.isnan(arr_s[i+1]):
            continue
        y1, y2 = arr_s[i], arr_s[i+1]
        if (y1 >= half >= y2) or (y1 <= half <= y2):
            right_pos = interp_cross(i, i+1, half, y1, y2, x[i], x[i+1])
            break

    fwhm = np.nan
    if left_pos is not None and right_pos is not None:
        fwhm = float(right_pos - left_pos)

    # optional Gaussian fit for a fitted FWHM estimate
    fwhm_fit = np.nan
    try:
        from math import sqrt, log
        from scipy.optimize import curve_fit

        def gauss(xv, a, mu, sigma, offset):
            return a * np.exp(-0.5 * ((xv - mu) / sigma)**2) + offset

        # prepare data for fit: restrict to region around peak
        half_width = max(3, int(len(arr_s) * 0.2))
        i0 = max(0, peak_idx - half_width)
        i1 = min(len(arr_s), peak_idx + half_width)
        xf = x[i0:i1]
        yf = arr_s[i0:i1]
        if len(xf) >= 5:
            offset0 = float(np.nanmin(yf))
            a0 = float(peak_val - offset0)
            mu0 = float(x[peak_idx])
            sigma0 = max(1.0, (x[i1-1] - x[i0]) / 6.0)
            p0 = [a0, mu0, sigma0, offset0]
            popt, pcov = curve_fit(gauss, xf, yf, p0=p0, maxfev=20000)
            sigma_fit = abs(popt[2])
            fwhm_fit = 2.0 * sqrt(2.0 * log(2.0)) * sigma_fit
    except Exception:
        # ignore if scipy not installed or fit fails
        fwhm_fit = np.nan

    return {'fwhm': fwhm, 'peak': peak_val, 'peak_index': peak_idx, 'left_pos': left_pos, 'right_pos': right_pos, 'fwhm_fit': fwhm_fit, 'used_smoothing': used_smoothing}


if spec_file and record_file:
    df_spec, spec_preview = _safe_read_csv(spec_file, forced_delim=spec_forced_delim, skip_bad_lines=skip_bad_lines)
    df_record, rec_preview = _safe_read_csv(record_file, forced_delim=rec_forced_delim, skip_bad_lines=skip_bad_lines)

    if df_spec is not None and df_record is not None:
        st.success("Files loaded successfully!")

        # --- Parsing preview and diagnostics ---
        st.sidebar.subheader("Parsing preview & diagnostics")
        with st.expander("SPEC preview (first lines)"):
            plines = spec_preview.get('preview_lines', []) if isinstance(spec_preview, dict) else []
            st.text('\n'.join(plines[:20]))
            counts = spec_preview.get('field_counts', []) if isinstance(spec_preview, dict) else []
            if counts:
                st.write("Field counts (first 20 lines):", counts[:20])
                # highlight lines that differ from header count
                if len(counts) > 0:
                    hdr = counts[0]
                    mismatches = [i+1 for i,c in enumerate(counts) if c != hdr]
                    if mismatches:
                        st.warning(f"SPEC file: lines with unexpected field counts: {mismatches[:10]} (showing up to 10)")

        with st.expander("Record preview (first lines)"):
            plines = rec_preview.get('preview_lines', []) if isinstance(rec_preview, dict) else []
            st.text('\n'.join(plines[:20]))
            counts = rec_preview.get('field_counts', []) if isinstance(rec_preview, dict) else []
            if counts:
                st.write("Field counts (first 20 lines):", counts[:20])
                hdr = counts[0]
                mismatches = [i+1 for i,c in enumerate(counts) if c != hdr]
                if mismatches:
                    st.warning(f"Record file: lines with unexpected field counts: {mismatches[:10]} (showing up to 10)")
                    # show skipped lines if parser skipped them
                    skipped = rec_preview.get('skipped_lines', []) if isinstance(rec_preview, dict) else []
                    if skipped:
                        st.info(f"Skipped lines during parsing: {skipped}")
                        # offer download of skipped lines (small preview)
                        snippet_lines = [plines[i-1] for i in skipped if 0 < i <= len(plines)]
                        if snippet_lines:
                            skipped_text = '\n'.join(snippet_lines)
                            st.download_button('Download skipped/malformed preview (text)', data=skipped_text, file_name='skipped_lines_preview.txt')

        # Provide manual delimiter override suggestion
        if rec_preview and isinstance(rec_preview, dict) and rec_forced_delim is None:
            st.sidebar.info("If you see mismatched field counts, try selecting a different delimiter for the Record file and press Re-parse.")
        if st.sidebar.button('Re-parse files'):
            df_spec, spec_preview = _safe_read_csv(spec_file, forced_delim=spec_forced_delim, skip_bad_lines=skip_bad_lines)
            df_record, rec_preview = _safe_read_csv(record_file, forced_delim=rec_forced_delim, skip_bad_lines=skip_bad_lines)

        # --- Outlier plotting / inspection UI ---
        st.sidebar.subheader("Plot & Outlier Inspection")
        plot_cols = list(df_record.select_dtypes(include=['number']).columns)
        selected_cols = st.sidebar.multiselect('Numeric columns to plot (Record)', plot_cols, default=plot_cols[:2])
        outlier_method = st.sidebar.selectbox('Outlier method', ['IQR', 'Z-score'], index=0)
        if selected_cols:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            fig, axes = plt.subplots(max(1, len(selected_cols)), 1, figsize=(8, 3*len(selected_cols)))
            if len(selected_cols) == 1:
                axes = [axes]
            outlier_indices = set()
            for ax, col in zip(axes, selected_cols):
                series = pd.to_numeric(df_record[col], errors='coerce')
                ax.plot(series.fillna(np.nan).values, marker='o', linestyle='-')
                ax.set_title(col)
                # compute outliers
                if outlier_method == 'IQR':
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    mask = (series < lower) | (series > upper)
                else:
                    mean = series.mean()
                    std = series.std()
                    mask = (series - mean).abs() > 3 * std
                ax.scatter(np.where(mask)[0], series[mask].values, color='red', zorder=3)
                outlier_indices.update(np.where(mask.fillna(False))[0].tolist())
            st.pyplot(fig)
            if outlier_indices:
                st.info(f"Detected outlier rows (0-based indices): {sorted(list(outlier_indices))[:50]}")
                if st.button('Show outlier rows'):
                    st.dataframe(df_record.iloc[sorted(list(outlier_indices))])

        # ensure ic23_res defined
        ic23_res = None

        # --- IC23 width computation UI ---
        st.sidebar.subheader("IC23 / FWHM analysis")
        ic23_cols = list(df_record.select_dtypes(include=['number']).columns)
        ic23_col = st.sidebar.selectbox('Select signal column for IC23 width', options=['-- none --'] + ic23_cols, index=0)
        # provide an explicit X-axis selector
        x_candidates = ['-- index --'] + list(df_record.columns)
        x_choice = st.sidebar.selectbox('Optional X-axis column (for non-uniform spacing)', options=x_candidates, index=0)
        if ic23_col and ic23_col != '-- none --':
            sig = pd.to_numeric(df_record[ic23_col], errors='coerce').fillna(np.nan)
            x_axis = None
            if x_choice and x_choice != '-- index --':
                try:
                    x_axis = pd.to_numeric(df_record[x_choice], errors='coerce').fillna(np.arange(len(sig))).values
                except Exception:
                    x_axis = None

            ic23_res = compute_ic23_width(sig.values, x=x_axis)
            st.write('IC23 result:', ic23_res)
            # plot signal with markers
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(sig.values, label=ic23_col)
                if ic23_res.get('left_pos') is not None:
                    ax.axvline(ic23_res['left_pos'], color='red', linestyle='--', label='left half-max')
                if ic23_res.get('right_pos') is not None:
                    ax.axvline(ic23_res['right_pos'], color='red', linestyle='--', label='right half-max')
                ax.set_title(f"{ic23_col} — FWHM: {ic23_res.get('fwhm')}")
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Failed to plot IC23 result: {e}")

            # offer download of IC23 result
            import json
            st.download_button('Download IC23 result (json)', data=json.dumps(ic23_res), file_name='ic23_result.json', mime='application/json')


        # Normalize uploaded DataFrames (avoid duplicate headers and ensure Sample column)
        # normalize_uploaded_df returns (df, info) — assign defensively
        try:
            df_record_norm, _rec_info = normalize_uploaded_df(df_record)
            if isinstance(df_record_norm, pd.DataFrame):
                df_record = df_record_norm
        except Exception:
            # fallback: ensure Sample exists
            if 'Sample' not in df_record.columns:
                df_record.insert(0, 'Sample', range(1, len(df_record) + 1))

        try:
            df_spec_norm, _spec_info = normalize_uploaded_df(df_spec)
            if isinstance(df_spec_norm, pd.DataFrame):
                df_spec = df_spec_norm
        except Exception:
            if 'Sample' not in df_spec.columns:
                df_spec.insert(0, 'Sample', range(1, len(df_spec) + 1))

        # Infer likely column names in record
        rec_cols = list(df_record.columns)
        dose_col = _find_column(rec_cols, ['X_DOSE(C)', 'X_DOSE', 'DOSE', 'DOSE_C'])
        xpos_col = _find_column(rec_cols, ['X_POSITION(mm)', 'X_POSITION', 'XPOS', 'X_POSITIONmm'])
        ypos_col = _find_column(rec_cols, ['Y_POSITION(mm)', 'Y_POSITION', 'YPOS', 'Y_POSITIONmm'])
        beam_col = _find_column(rec_cols, ['BEAMCURRENT(V)', 'BEAMCURRENT', 'BEAM_CURRENT', 'BEAMCURRENTV'])
        layer_col = _find_column(rec_cols, ['LAYER_ID', 'LAYER', 'LAYERID']) or 'Sample'
        spot_col = _find_column(rec_cols, ['SPOT_ID', 'SPOT', 'SPOTID'])

        # Spec target column candidates
        spec_target_col = _find_column(list(df_spec.columns), ['TARGET_CHARGE', 'TARGET', 'CHARGE', 'TARGET_CHARGE(C)'])

        # Coerce numerics safely: always create float64 Series (np.nan for missing)
        n_rows = len(df_record)
        if dose_col:
            df_record['__dose'] = pd.to_numeric(df_record[dose_col], errors='coerce').astype('float64')
        else:
            df_record['__dose'] = pd.Series([np.nan] * n_rows, dtype='float64')
        if xpos_col:
            df_record['__xpos'] = pd.to_numeric(df_record[xpos_col], errors='coerce').astype('float64')
        else:
            df_record['__xpos'] = pd.Series([np.nan] * n_rows, dtype='float64')
        if ypos_col:
            df_record['__ypos'] = pd.to_numeric(df_record[ypos_col], errors='coerce').astype('float64')
        else:
            df_record['__ypos'] = pd.Series([np.nan] * n_rows, dtype='float64')
        if beam_col:
            df_record['__beam'] = pd.to_numeric(df_record[beam_col], errors='coerce').astype('float64')
        else:
            df_record['__beam'] = pd.Series([np.nan] * n_rows, dtype='float64')

        # Build target series from spec (repeat to match record length)
        if spec_target_col:
            spec_vals = pd.to_numeric(df_spec[spec_target_col], errors='coerce').fillna(1).values
        else:
            spec_vals = np.ones(max(1, len(df_record)))

        # vectorized QA checks
        n = len(df_record)
        # align spec values by modulo index as original
        spec_aligned = np.array([spec_vals[i % len(spec_vals)] for i in range(n)])

        dose_dev = (df_record['__dose'].astype('float64') - spec_aligned) / spec_aligned * 100
        dose_status = dose_dev.abs() <= float(dose_tol_pct)
        dose_status = dose_status.where(~df_record['__dose'].isna(), other=pd.NA)

        pos_dev = np.sqrt((df_record['__xpos'].astype('float64') - df_spec.get('X_POSITION', df_record.get('__xpos', pd.Series(np.zeros(n)))).iloc[:n].fillna(0).values)**2 +
                          (df_record['__ypos'].astype('float64') - df_spec.get('Y_POSITION', df_record.get('__ypos', pd.Series(np.zeros(n)))).iloc[:n].fillna(0).values)**2)
        pos_status = pos_dev <= float(pos_tol_mm)
        pos_status = pos_status.where(~(df_record['__xpos'].isna() | df_record['__ypos'].isna()), other=pd.NA)

        beam_status = df_record['__beam'].astype('float64') <= float(beam_current_max)
        beam_status = beam_status.where(~df_record['__beam'].isna(), other=pd.NA)

        # Build result DataFrame
        res = pd.DataFrame({
            'Layer': df_record.get(layer_col, df_record.index + 1),
            'Spot': df_record.get(spot_col, pd.Series([None]*n)),
            'Dose_dev_pct': dose_dev,
            'Dose': dose_status.map({True: 'PASS', False: 'FAIL'}),
            'Position_mm': pos_dev,
            'Position': pos_status.map({True: 'PASS', False: 'FAIL'}),
            'Beam_Current': df_record['__beam'],
            'Beam Current': beam_status.map({True: 'PASS', False: 'FAIL'})
        })

        # Root cause
        def _root_cause(row):
            fails = []
            if row['Dose'] == 'FAIL':
                fails.append('Dose deviation')
            if row['Position'] == 'FAIL':
                fails.append('Position deviation')
            if row['Beam Current'] == 'FAIL':
                fails.append('Beam current high')
            return ' | '.join(fails) if fails else 'OK'

        res['Root Cause'] = res.apply(_root_cause, axis=1)

        st.subheader("QA Results")
        st.dataframe(res)

        # Excel download: include raw sheets for traceability
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            res.to_excel(writer, index=False, sheet_name='QA_Report')
            try:
                df_record.to_excel(writer, index=False, sheet_name='Raw_Record')
            except Exception:
                pass
            try:
                df_spec.to_excel(writer, index=False, sheet_name='Spec')
            except Exception:
                pass
            # add IC23 result sheet if available
            try:
                if ic23_res:
                    ic23_df = pd.DataFrame([ic23_res])
                    ic23_df.to_excel(writer, index=False, sheet_name='IC23_Result')
            except Exception:
                pass
        buf.seek(0)
        st.download_button('Download Excel Report', data=buf.getvalue(), file_name='QA_Report.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        # PDF download (matplotlib PdfPages)
        if st.button('Download PDF Report'):
            try:
                import matplotlib.pyplot as plt
                from matplotlib.backends.backend_pdf import PdfPages

                pdf_buf = BytesIO()
                with PdfPages(pdf_buf) as pdf:
                    # PASS counts
                    try:
                        summary = pd.DataFrame({
                            'Dose': res['Dose'].value_counts(dropna=False),
                            'Position': res['Position'].value_counts(dropna=False),
                            'Beam Current': res['Beam Current'].value_counts(dropna=False)
                        }).fillna(0)
                        fig, ax = plt.subplots(figsize=(8, 4))
                        # plot PASS counts per parameter
                        pass_counts = [summary.loc['PASS', 'Dose'] if 'PASS' in summary.index else 0,
                                       summary.loc['PASS', 'Position'] if 'PASS' in summary.index else 0,
                                       summary.loc['PASS', 'Beam Current'] if 'PASS' in summary.index else 0]
                        ax.bar(['Dose', 'Position', 'Beam Current'], pass_counts, color='tab:green')
                        ax.set_ylabel('PASS count')
                        ax.set_title('PASS counts')
                        pdf.savefig(fig)
                        plt.close(fig)
                    except Exception:
                        pass

                    # Text summary page
                    try:
                        fig2, ax2 = plt.subplots(figsize=(8, 6))
                        ax2.axis('off')
                        txt = f"QA Report\nSamples: {len(res)}\n\n"
                        txt += res[['Dose', 'Position', 'Beam Current', 'Root Cause']].head(100).to_string()
                        ax2.text(0.01, 0.99, txt, va='top', ha='left', fontsize=8, family='monospace')
                        pdf.savefig(fig2)
                        plt.close(fig2)
                    except Exception:
                        pass

                pdf_buf.seek(0)
                st.download_button('Download PDF Report', data=pdf_buf.getvalue(), file_name='QA_Report.pdf', mime='application/pdf')
            except Exception as e:
                st.error(f'Failed to create PDF: {e}')
