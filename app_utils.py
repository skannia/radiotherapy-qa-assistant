from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from qa_checks import tg142_check, tg224_check, electron_beam_check, tps_check
from functools import lru_cache
import re
import json
from pathlib import Path


def _df_to_bytes(df: pd.DataFrame) -> bytes:
    # Stable CSV representation for caching key. Exclude index to keep consistent.
    return df.to_csv(index=False).encode('utf-8')


@lru_cache(maxsize=128)
def _cached_run_qa_lru(df_bytes: bytes, selected_qas_tuple: Tuple[str, ...], thresholds_tuple: Tuple[tuple, ...] = tuple(), col_map_tuple: Tuple[tuple, ...] = tuple()):
    """Internal LRU-cached runner that accepts a CSV bytes fingerprint and tuple of QA names.

    Returns the pivoted DataFrame serialized as CSV bytes to keep the cache value simple and hashable.
    """
    # Reconstruct DataFrame from bytes
    import io
    df = pd.read_csv(io.BytesIO(df_bytes))
    # reconstruct thresholds and col_map from tuples
    thresholds = dict(thresholds_tuple) if thresholds_tuple else {}
    user_col_map = dict(col_map_tuple) if col_map_tuple else {}
    # Infer column mapping (canonical -> actual)
    inferred_map = infer_column_mapping(list(df.columns))
    # Merge with user overrides: user overrides take precedence
    final_map = {}
    for canon, actual in inferred_map.items():
        if canon in user_col_map and user_col_map[canon]:
            final_map[canon] = user_col_map[canon]
        else:
            final_map[canon] = actual
    # create a working copy and add alias columns for expected names
    df_work = df.copy()
    for canon, actual in final_map.items():
        if actual and canon not in df_work.columns and actual in df.columns:
            try:
                df_work[canon] = df[actual]
            except Exception:
                pass
    results = []
    # pass thresholds into QA functions where supported
    for qa in selected_qas_tuple:
        func = { 'Photon QA': tg142_check, 'Proton/ Ion QA': tg224_check, 'Electron QA': electron_beam_check, 'TPS QA': tps_check }.get(qa)
        if func:
            try:
                results += func(df_work, thresholds=thresholds)
            except TypeError:
                results += func(df_work)
    if not results:
        return b''
    df_results = pd.DataFrame(results)
    if 'index' in df_results.columns and 'Sample' not in df_results.columns:
        df_results = df_results.rename(columns={'index': 'Sample'})
    pivot = df_results.pivot(index='Sample', columns='parameter', values='status')
    pivot.reset_index(inplace=True)
    # Return CSV bytes as the cache value
    return pivot.to_csv(index=False).encode('utf-8')


def _normalize_name(name: str) -> str:
    s = name.lower()
    s = re.sub(r'[^a-z0-9]', '', s)
    return s


def infer_column_mapping(columns: List[str]) -> Dict[str, str]:
    """Return a mapping from canonical column names to actual column names when possible.

    Matching is case-insensitive and ignores non-alphanumeric characters. Some synonyms are handled.
    """
    canonical = ['Sample', 'output', 'flatness', 'symmetry', 'dose_uniformity', 'beam_energy', 'electron_output', 'PDD', 'planned_vs_measured', 'gamma_index']
    norm_to_actual = { _normalize_name(c): c for c in columns }
    mapping = {}
    syn = {
        'plannedvsmeasured': 'planned_vs_measured',
        'beamenergy': 'beam_energy',
        'doseuniformity': 'dose_uniformity',
        'electronoutput': 'electron_output',
        'gammaindex': 'gamma_index',
        'pdd': 'PDD'
    }
    for canon in canonical:
        nc = _normalize_name(canon)
        # direct match
        if nc in norm_to_actual:
            mapping[canon] = norm_to_actual[nc]
            continue
        # synonym match
        if nc in syn and _normalize_name(syn[nc]) in norm_to_actual:
            mapping[canon] = norm_to_actual[_normalize_name(syn[nc])]
            continue
        # fuzzy: look for any column whose normalized form contains the canonical token
        found = None
        for nrm, actual in norm_to_actual.items():
            if nc in nrm or nrm in nc:
                found = actual
                break
        mapping[canon] = found
    return mapping


def cached_run_qa_lru(df: pd.DataFrame, selected_qas: List[str], thresholds: Dict[str, float] = None, column_map: Dict[str, str] = None) -> pd.DataFrame:
    """Public wrapper for tests to call cached QA runner using LRU caching.

    Converts df to bytes and selected_qas to tuple for hashing, calls _cached_run_qa_lru,
    and returns a DataFrame reconstructed from the cached CSV bytes.
    """
    csv_bytes = _df_to_bytes(df)
    sel_tuple = tuple(selected_qas)
    # include thresholds in key by serializing to a tuple of items
    thr_tuple = tuple(sorted((thresholds or {}).items()))
    col_map_tuple = tuple(sorted((column_map or {}).items()))
    csv_out = _cached_run_qa_lru(csv_bytes, sel_tuple, thr_tuple, col_map_tuple)
    if not csv_out:
        return pd.DataFrame()
    import io
    return pd.read_csv(io.BytesIO(csv_out))


def cached_run_qa_with_flag(df: pd.DataFrame, selected_qas: List[str], thresholds: Dict[str, float] = None, column_map: Dict[str, str] = None):
    """Run the LRU-cached QA runner and return (pivot_df, cache_hit: bool).

    Detects cache hit/miss by sampling _cached_run_qa_lru.cache_info() before and after the call.
    """
    # Ensure selected_qas order deterministic
    sel_tuple = tuple(selected_qas)
    csv_bytes = _df_to_bytes(df)
    thr_tuple = tuple(sorted((thresholds or {}).items()))
    col_map_tuple = tuple(sorted((column_map or {}).items()))
    # sample cache info
    try:
        before = _cached_run_qa_lru.cache_info()
    except Exception:
        # If cache not available, fallback to running without flag
        return cached_run_qa_lru(df, selected_qas), False

    csv_out = _cached_run_qa_lru(csv_bytes, sel_tuple, thr_tuple, col_map_tuple)

    try:
        after = _cached_run_qa_lru.cache_info()
        # If hits increased, it was a cache hit; if misses increased, it was computed and cached
        cache_hit = (after.hits > before.hits)
    except Exception:
        cache_hit = False

    if not csv_out:
        return pd.DataFrame(), cache_hit

    import io
    return pd.read_csv(io.BytesIO(csv_out)), cache_hit


def get_columns_to_show(selected_qas: List[str], qa_columns_map: Dict[str, List[str]], df_columns: List[str]) -> Tuple[List[str], List[str]]:
    """Return ordered list of columns to show and list of missing expected columns."""
    columns_to_show = ["Sample"]
    missing = []
    for qa in selected_qas:
        for col in qa_columns_map.get(qa, []):
            if col in df_columns and col not in columns_to_show:
                columns_to_show.append(col)
            elif col not in df_columns:
                missing.append(col)
    return columns_to_show, sorted(set(missing))


def highlight_fail_value(val, col: str, thresholds: Dict[str, float]):
    """Return CSS style for a value and column using provided thresholds.

    thresholds keys: 'gamma_index', 'beam_energy_tol', 'deviation_tol' (percent points)
    """
    if col == 'Sample':
        return ''
    try:
        if pd.isna(val):
            return ''
    except Exception:
        return ''

    try:
        if col == 'gamma_index':
            return 'background-color: red' if float(val) < thresholds.get('gamma_index', 95) else 'background-color: lightgreen'
        elif col == 'beam_energy':
            return 'background-color: red' if abs(float(val) - 6) > thresholds.get('beam_energy_tol', 1) else 'background-color: lightgreen'
        else:
            # default: compare to 100 with percent deviation tolerance
            return 'background-color: red' if abs(float(val) - 100) > thresholds.get('deviation_tol', 3) else 'background-color: lightgreen'
    except Exception:
        return ''


_QA_FUNC_MAP = {
    'Photon QA': tg142_check,
    'Proton/ Ion QA': tg224_check,
    'Electron QA': electron_beam_check,
    'TPS QA': tps_check,
}


def run_qa_and_pivot(df: pd.DataFrame, selected_qas: List[str]) -> pd.DataFrame:
    """Run QA modules (from qa_checks) for selected modules and return pivoted PASS/FAIL DataFrame.

    The returned DataFrame has 'Sample' as first column and one column per parameter with PASS/FAIL.
    """
    results = []
    # Map selected_qas to qa functions where available
    for qa in selected_qas:
        func = _QA_FUNC_MAP.get(qa)
        if func:
            results += func(df)

    if not results:
        return pd.DataFrame()

    df_results = pd.DataFrame(results)
    # normalize index key name
    if 'index' in df_results.columns and 'Sample' not in df_results.columns:
        df_results = df_results.rename(columns={'index': 'Sample'})

    # Ensure we have status column; some entries may be missing 'status' if new structure used
    if 'status' not in df_results.columns and 'value' in df_results.columns:
        # derive status from value/delta/tolerance if possible
        def _derive_status(row):
            if pd.isna(row.get('value')):
                return 'MISSING'
            tol = row.get('tolerance')
            if row.get('parameter') == 'gamma_index' and tol is not None:
                return 'PASS' if row.get('value') >= tol else 'FAIL'
            if tol is None:
                return 'FAIL' if pd.isna(row.get('value')) else 'PASS'
            # default comparator: abs around 100
            return 'PASS' if abs(row.get('value') - 100) <= tol else 'FAIL'

        df_results['status'] = df_results.apply(_derive_status, axis=1)

    # Pivot to PASS/FAIL table
    pivot = df_results.pivot(index='Sample', columns='parameter', values='status')
    pivot.reset_index(inplace=True)
    return pivot


def normalize_uploaded_df(df: pd.DataFrame):
    """Make uploaded DataFrame column names unique and ensure a human-friendly 'Sample' column.

    - If duplicate column names are present, rename duplicates by appending .1, .2, ...
    - If 'Sample' is missing, add it as a 1-based index column.
    - If 'Sample' exists and is zero-based integers, convert to 1-based.

    Returns the normalized DataFrame and a dict with info: {'renamed': bool, 'renames': {old:new}, 'sample_added': bool}
    """
    info = {'renamed': False, 'renames': {}, 'sample_added': False}

    def _make_unique_columns(columns):
        counts = {}
        new_cols = []
        for col in columns:
            if col in counts:
                counts[col] += 1
                new_name = f"{col}.{counts[col]}"
                # record original name for reference
                info['renames'][new_name] = col
                new_cols.append(new_name)
            else:
                counts[col] = 0
                new_cols.append(col)
        return new_cols

    # Ensure duplicate column names are made unique first
    if df.columns.duplicated().any():
        info['renamed'] = True
        df.columns = _make_unique_columns(list(df.columns))

    # Guarantee a 'Sample' column exists. Avoid relying on reset_index() rename which
    # can sometimes miss if the reset produces unexpected names (edge cases with named
    # index or collisions). Insert an explicit 1-based index column if absent.
    if 'Sample' not in list(df.columns):
        # create a 1-based integer Sample column at the front
        try:
            samples = np.arange(1, len(df) + 1, dtype=int)
            df.insert(0, 'Sample', samples)
            info['sample_added'] = True
        except Exception:
            # fallback: use reset_index rename strategy
            try:
                df = df.reset_index().rename(columns={'index': 'Sample'})
                if 'Sample' not in df.columns:
                    df.insert(0, 'Sample', np.arange(1, len(df) + 1, dtype=int))
                else:
                    # ensure 1-based
                    try:
                        df['Sample'] = pd.to_numeric(df['Sample'], errors='coerce') + 1
                    except Exception:
                        pass
                info['sample_added'] = True
            except Exception:
                # If even this fails, ensure at least that the DataFrame is returned unchanged
                pass
    else:
        # If Sample exists, try to coerce to numeric and adjust zero-based indices.
        try:
            s = pd.to_numeric(df['Sample'], errors='coerce')
            # If coercion produced mostly numbers, use them; otherwise leave as-is.
            if s.notna().sum() >= max(1, int(len(s) * 0.5)):
                # If the minimum value looks zero-based, shift to 1-based
                if s.min() == 0:
                    df['Sample'] = s.fillna(0).astype(int) + 1
                else:
                    # keep numeric values (preserve ints where possible)
                    if pd.api.types.is_integer_dtype(s.dropna()):
                        df['Sample'] = s.astype(int)
                    else:
                        df['Sample'] = s
        except Exception:
            # leave the original Sample column untouched on any error
            pass

    return df, info


### Preset persistence helpers
PRESET_DIR = Path('.qa_presets')


def _ensure_preset_dir():
    try:
        PRESET_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def save_preset(name: str, mapping: Dict[str, str]) -> bool:
    """Save a mapping preset by name as JSON under .qa_presets/name.json."""
    _ensure_preset_dir()
    if not name:
        return False
    try:
        p = PRESET_DIR / f"{name}.json"
        with p.open('w', encoding='utf-8') as f:
            json.dump(mapping or {}, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def load_preset(name: str) -> Dict[str, str]:
    """Load a preset mapping by name. Returns dict or empty dict on error."""
    try:
        p = PRESET_DIR / f"{name}.json"
        if not p.exists():
            return {}
        with p.open('r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def list_presets() -> list:
    _ensure_preset_dir()
    try:
        return [p.stem for p in PRESET_DIR.glob('*.json')]
    except Exception:
        return []


def delete_preset(name: str) -> bool:
    try:
        p = PRESET_DIR / f"{name}.json"
        if p.exists():
            p.unlink()
            return True
        return False
    except Exception:
        return False
