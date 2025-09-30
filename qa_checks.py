import pandas as pd
from typing import List, Dict


def _get_sample_id(df: pd.DataFrame, idx: int):
    """Return 1-based Sample id: prefer df['Sample'] when present, else idx+1."""
    if 'Sample' in df.columns:
        try:
            val = df.at[idx, 'Sample']
            # If it's numeric and zero-based, make 1-based; otherwise return as-is
            if pd.api.types.is_integer_dtype(type(val)) and int(val) == 0:
                return int(val) + 1
            return val
        except Exception:
            pass
    return idx + 1


def _safe_numeric(val):
    try:
        return pd.to_numeric(val, errors='coerce')
    except Exception:
        return float('nan')


def _emit_result(results: List[Dict], sample, parameter, measured, tolerance, target=100.0, comparator='abs'):
    """Append a result dict with optional metadata. comparator: 'abs' or 'gte' or custom."""
    entry = {'Sample': sample, 'parameter': parameter, 'value': None, 'tolerance': tolerance, 'delta': None, 'status': 'MISSING'}
    num = _safe_numeric(measured)
    if pd.isna(num):
        # missing measurement
        results.append(entry)
        return
    entry['value'] = float(num)
    if comparator == 'abs':
        entry['delta'] = float(abs(num - target))
        entry['status'] = 'PASS' if abs(num - target) <= tolerance else 'FAIL'
    elif comparator == 'gte':
        entry['delta'] = float(num)
        entry['status'] = 'PASS' if num >= tolerance else 'FAIL'
    else:
        # custom comparator (callable) not used here
        entry['delta'] = float(num - target)
        entry['status'] = 'PASS' if abs(num - target) <= tolerance else 'FAIL'
    results.append(entry)


def tg142_check(df: pd.DataFrame):
    """TG142 checks: output (±2), flatness (±3), symmetry (±3) around 100."""
    results = []
    for idx, row in df.iterrows():
        sample = _get_sample_id(df, idx)
        _emit_result(results, sample, 'output', row.get('output') if hasattr(row, 'get') else row['output'], tolerance=2, target=100.0)
        _emit_result(results, sample, 'flatness', row.get('flatness') if hasattr(row, 'get') else row['flatness'], tolerance=3, target=100.0)
        _emit_result(results, sample, 'symmetry', row.get('symmetry') if hasattr(row, 'get') else row['symmetry'], tolerance=3, target=100.0)
    return results


def tg224_check(df: pd.DataFrame):
    """TG224 checks: dose_uniformity (±2 around 100), beam_energy (±1 around 6 MeV)."""
    results = []
    for idx, row in df.iterrows():
        sample = _get_sample_id(df, idx)
        _emit_result(results, sample, 'dose_uniformity', row.get('dose_uniformity') if hasattr(row, 'get') else row['dose_uniformity'], tolerance=2, target=100.0)
        _emit_result(results, sample, 'beam_energy', row.get('beam_energy') if hasattr(row, 'get') else row['beam_energy'], tolerance=1, target=6.0)
    return results


def electron_beam_check(df: pd.DataFrame):
    results = []
    for idx, row in df.iterrows():
        sample = _get_sample_id(df, idx)
        _emit_result(results, sample, 'electron_output', row.get('electron_output') if hasattr(row, 'get') else row['electron_output'], tolerance=3, target=100.0)
        _emit_result(results, sample, 'PDD', row.get('PDD') if hasattr(row, 'get') else row['PDD'], tolerance=2, target=100.0)
    return results


def tps_check(df: pd.DataFrame):
    results = []
    for idx, row in df.iterrows():
        sample = _get_sample_id(df, idx)
        _emit_result(results, sample, 'planned_vs_measured', row.get('planned_vs_measured') if hasattr(row, 'get') else row['planned_vs_measured'], tolerance=3, target=100.0)
        # gamma index: comparator is 'gte' with threshold 95
        _emit_result(results, sample, 'gamma_index', row.get('gamma_index') if hasattr(row, 'get') else row['gamma_index'], tolerance=95, comparator='gte')
    return results
def apply_physics_rules(df: pd.DataFrame):
    """
    Run TG142, TG224, electron beam, and TPS QA checks.
    Returns a combined DataFrame of results + log messages.
    """
    results = []
    logs = []

    # TG142 check
    if any(col in df.columns for col in ["output", "flatness", "symmetry"]):
        results.extend(tg142_check(df))
    else:
        logs.append("⚠️ TG142 check skipped: required columns (output, flatness, symmetry) missing.")

    # TG224 check
    if any(col in df.columns for col in ["dose_uniformity", "beam_energy"]):
        results.extend(tg224_check(df))
    else:
        logs.append("⚠️ TG224 check skipped: required columns (dose_uniformity, beam_energy) missing.")

    # Electron beam QA
    if any(col in df.columns for col in ["electron_output", "PDD"]):
        results.extend(electron_beam_check(df))
    else:
        logs.append("⚠️ Electron beam QA skipped: required columns (electron_output, PDD) missing.")

    # TPS QA
    if any(col in df.columns for col in ["planned_vs_measured", "gamma_index"]):
        results.extend(tps_check(df))
    else:
        logs.append("⚠️ TPS QA skipped: required columns (planned_vs_measured, gamma_index) missing.")

    # Convert results into DataFrame
    results_df = pd.DataFrame(results)

    return results_df, logs
