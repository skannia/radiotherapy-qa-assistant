import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict


def _normalize_key(s: str) -> str:
    if s is None:
        return ''
    return ''.join(ch.lower() for ch in s if ch.isalnum())


def _find_column(cols, candidates):
    norm = { _normalize_key(c): c for c in cols }
    for cand in candidates:
        key = _normalize_key(cand)
        if key in norm:
            return norm[key]
    # fuzzy
    for cand in candidates:
        token = _normalize_key(cand)
        for nrm, actual in norm.items():
            if token in nrm or nrm in token:
                return actual
    return None


def perform_qa(df_record: pd.DataFrame, df_spec: pd.DataFrame,
               dose_tol_pct: float = 5.0,
               pos_tol_mm: float = 1.0,
               beam_current_max: float = 6000.0,
               id_candidates_record: Optional[list] = None,
               id_candidates_spec: Optional[list] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform QA checks and return (result_df, merged_raw_or_aligned_spec_series).

    result_df: DataFrame with columns ['Layer','Spot','Dose_dev_pct','Dose','Position_mm','Position','Beam_Current','Beam Current','Root Cause']
    merged_raw_or_aligned_spec: DataFrame or Series used for alignment (for traceability)
    """
    if id_candidates_record is None:
        id_candidates_record = ['Sample', 'LAYER_ID', 'LAYER', 'LAYERID', 'ID']
    if id_candidates_spec is None:
        id_candidates_spec = ['Sample', 'LAYER_ID', 'LAYER', 'LAYERID', 'ID']

    # copy to avoid mutating inputs
    rec = df_record.copy()
    spec = df_spec.copy()

    # ensure indices
    n = len(rec)

    # find columns
    rec_cols = list(rec.columns)
    dose_col = _find_column(rec_cols, ['X_DOSE(C)', 'X_DOSE', 'DOSE', 'DOSE_C'])
    xpos_col = _find_column(rec_cols, ['X_POSITION(mm)', 'X_POSITION', 'XPOS'])
    ypos_col = _find_column(rec_cols, ['Y_POSITION(mm)', 'Y_POSITION', 'YPOS'])
    beam_col = _find_column(rec_cols, ['BEAMCURRENT(V)', 'BEAMCURRENT', 'BEAM_CURRENT'])
    layer_col = _find_column(rec_cols, ['LAYER_ID', 'LAYER', 'LAYERID'])
    spot_col = _find_column(rec_cols, ['SPOT_ID', 'SPOT', 'SPOTID'])

    spec_cols = list(spec.columns)
    spec_target_col = _find_column(spec_cols, ['TARGET_CHARGE', 'TARGET', 'CHARGE', 'TARGET_CHARGE(C)'])

    # Coerce numerics
    rec['__dose'] = pd.to_numeric(rec[dose_col], errors='coerce') if dose_col else pd.Series([np.nan]*n)
    rec['__xpos'] = pd.to_numeric(rec[xpos_col], errors='coerce') if xpos_col else pd.Series([np.nan]*n)
    rec['__ypos'] = pd.to_numeric(rec[ypos_col], errors='coerce') if ypos_col else pd.Series([np.nan]*n)
    rec['__beam'] = pd.to_numeric(rec[beam_col], errors='coerce') if beam_col else pd.Series([np.nan]*n)

    # Attempt alignment by ID if possible
    rec_id_col = _find_column(rec_cols, id_candidates_record)
    spec_id_col = _find_column(spec_cols, id_candidates_spec)

    if rec_id_col and spec_id_col and rec_id_col in rec.columns and spec_id_col in spec.columns:
        # merge on id (left join) to align spec values where available
        rec_left = rec.copy()
        spec_subset = spec[[spec_id_col, spec_target_col]] if spec_target_col else spec[[spec_id_col]]
        merged = pd.merge(rec_left, spec_subset, how='left', left_on=rec_id_col, right_on=spec_id_col, suffixes=('', '_spec'))
        # build spec_aligned from merged
        if spec_target_col and spec_target_col in merged.columns:
            spec_aligned = pd.to_numeric(merged[spec_target_col], errors='coerce').fillna(1).values
        else:
            spec_aligned = np.ones(len(merged))
        source_for_alignment = merged
    else:
        # fallback: align by modulo as before
        if spec_target_col and spec_target_col in spec.columns:
            spec_vals = pd.to_numeric(spec[spec_target_col], errors='coerce').fillna(1).values
        else:
            spec_vals = np.ones(max(1, len(rec)))
        spec_aligned = np.array([spec_vals[i % len(spec_vals)] for i in range(n)])
        merged = rec.copy()
        source_for_alignment = spec_aligned

    # compute deviations
    dose_dev = (rec['__dose'].astype('float64') - spec_aligned) / spec_aligned * 100
    dose_status = dose_dev.abs() <= float(dose_tol_pct)
    dose_status = dose_status.where(~rec['__dose'].isna(), other=pd.NA)

    # position: need spec X/Y if merged DataFrame has them, else try spec columns
    if isinstance(source_for_alignment, pd.DataFrame):
        # try to get spec X/Y from merged (columns may be named X_POSITION or similar)
        spec_x = None
        spec_y = None
        for cand in ['X_POSITION', 'X_POSITIONmm', 'X_POSITIONmm', 'X_POSITION']: 
            if cand in source_for_alignment.columns:
                spec_x = pd.to_numeric(source_for_alignment[cand], errors='coerce').fillna(0).values
                break
        for cand in ['Y_POSITION', 'Y_POSITIONmm', 'Y_POSITIONmm', 'Y_POSITION']:
            if cand in source_for_alignment.columns:
                spec_y = pd.to_numeric(source_for_alignment[cand], errors='coerce').fillna(0).values
                break
        if spec_x is None:
            spec_x = np.zeros(len(rec))
        if spec_y is None:
            spec_y = np.zeros(len(rec))
    else:
        # no detailed spec positions available
        spec_x = np.zeros(len(rec))
        spec_y = np.zeros(len(rec))

    pos_dev = np.sqrt((rec['__xpos'].astype('float64').fillna(0) - spec_x)**2 + (rec['__ypos'].astype('float64').fillna(0) - spec_y)**2)
    pos_status = pos_dev <= float(pos_tol_mm)
    pos_status = pos_status.where(~(rec['__xpos'].isna() | rec['__ypos'].isna()), other=pd.NA)

    beam_status = rec['__beam'].astype('float64') <= float(beam_current_max)
    beam_status = beam_status.where(~rec['__beam'].isna(), other=pd.NA)

    res = pd.DataFrame({
        'Layer': rec[rec_id_col] if rec_id_col in rec.columns else rec.index + 1,
        'Spot': rec[spot_col] if spot_col in rec.columns else pd.Series([None]*len(rec)),
        'Dose_dev_pct': dose_dev,
        'Dose': dose_status.map({True: 'PASS', False: 'FAIL'}),
        'Position_mm': pos_dev,
        'Position': pos_status.map({True: 'PASS', False: 'FAIL'}),
        'Beam_Current': rec['__beam'],
        'Beam Current': beam_status.map({True: 'PASS', False: 'FAIL'})
    })

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

    return res, source_for_alignment
