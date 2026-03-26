#!/usr/bin/env python
# coding: utf-8
"""
to_vem.py — Convert MAVE score files of any supported format to VEM.tsv (wide matrix).

Supported input formats
-----------------------
1. VEM.tsv          Wide matrix (Position × amino acid, 3-letter headers) — passed through unchanged
2. MaveDB CSV/TSV   hgvs_pro column with HGVS protein notation (p.Met1Ala, p.Met1Ter …)
3. Pre-parsed TSV   Columns: Position, Amino_Acid_2, score  (output of ADSLcsv2tsv.sh)
4. Long TSV         Columns: Position, Amino_Acid, score    (output of VEM2tsv.py)
5. Simple 1-letter  var or aa_substitutions column, 1-letter codes (M1A, M1*, M1=)
6. Wide 1-letter    Wide matrix with 1-letter AA column headers (e.g. GBA1_VEM_R1.csv);
                    extra columns (wt_aa, median_score, mean_score …) are ignored

Format is auto-detected from column names; separator (comma / tab) is auto-detected.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THREE_LETTER_AAS = {
    'Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Glu', 'Gln', 'Gly',
    'His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser',
    'Thr', 'Trp', 'Tyr', 'Val', 'Ter',
}

ONE_TO_THREE = {
    'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys',
    'E': 'Glu', 'Q': 'Gln', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
    'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe', 'P': 'Pro',
    'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val',
    '*': 'Ter',
}

ONE_LETTER_AAS = set(ONE_TO_THREE.keys())  # single-letter AA codes + *

DESIRED_COL_ORDER = [
    'Ter', 'Pro', 'Gly', 'Tyr', 'Trp', 'Phe', 'Val', 'Leu', 'Ile',
    'Ala', 'Thr', 'Ser', 'Gln', 'Asn', 'Met', 'Cys', 'Glu', 'Asp',
    'Arg', 'Lys', 'His',
]


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def sniff_separator(file_path):
    """Detect whether the file is comma- or tab-separated."""
    with open(file_path, 'r') as f:
        first_line = f.readline()
    return '\t' if '\t' in first_line else ','


def _looks_numeric(s):
    """Return True if string looks like a number (int or float, possibly negative)."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def detect_format(df):
    """
    Infer the input format from column names.

    Detection priority (most specific first):
      vem_wide        — Position + only 3-letter AA column names
      vem_wide_1letter— position (case-insensitive) + 1-letter AA column names
      preparsed_2     — Amino_Acid_2 + Position  (ADSLcsv2tsv.sh output)
      preparsed       — Amino_Acid + Position     (VEM2tsv.py output)
      hgvs            — hgvs_pro column           (MaveDB CSV/TSV)
      simple          — var or aa_substitutions   (Parkin / ADSL 1-letter)
      headerless      — no header; col 1 looks like variant (E2C), col 2 numeric
    """
    cols = set(df.columns)
    first_col = df.columns[0]

    # Format 1: VEM wide — first column is Position, all others are 3-letter AA names (+ optional wt_aa)
    if first_col == 'Position' and cols - {'Position', 'wt_aa'} <= THREE_LETTER_AAS:
        return 'vem_wide'

    # Format 6: wide 1-letter — first column is position (any case),
    # and at least half the remaining columns are 1-letter AA codes
    if first_col.lower() == 'position':
        aa_cols = [c for c in df.columns[1:] if c in ONE_LETTER_AAS]
        if len(aa_cols) >= 10:   # enough 1-letter AA columns to be confident
            return 'vem_wide_1letter'

    # Format 3: pre-parsed TSV produced by ADSLcsv2tsv.sh → variantEffectMap.py
    if 'Amino_Acid_2' in cols and 'Position' in cols:
        return 'preparsed_2'

    # Format 4: long TSV produced by VEM2tsv.py
    if 'Amino_Acid' in cols and 'Position' in cols:
        return 'preparsed'

    # Format 2: MaveDB CSV/TSV with HGVS protein notation
    if 'hgvs_pro' in cols:
        return 'hgvs'

    # Format 5: simple 1-letter notation (Parkin supplementary, ADSL)
    if 'var' in cols or 'aa_substitutions' in cols:
        return 'simple'

    # Format 7: headerless file — first "column name" looks like a variant (e.g. E2C, M1A),
    # second looks like a score (numeric). The file has no header row.
    if (re.match(r'^[A-Z]\d+[A-Z*]$', first_col)
            and _looks_numeric(df.columns[1])):
        return 'headerless'

    raise ValueError(
        f"Cannot detect format from columns: {list(df.columns)}\n"
        "Expected one of: Position + AA headers, hgvs_pro, "
        "Amino_Acid_2 + Position, Amino_Acid + Position, var, aa_substitutions, "
        "or wide matrix with 1-letter AA headers."
    )


# ---------------------------------------------------------------------------
# Score column detection
# ---------------------------------------------------------------------------

# Columns that are never score columns
_NON_SCORE_COLS = {
    'Position', 'Amino_Acid', 'Amino_Acid_1', 'Amino_Acid_2',
    'accession', 'hgvs_pro', 'hgvs_nt', 'hgvs_splice',
    'var', 'aa_substitutions', 'WT',
}

def find_score_col(df, hint=None):
    """Return the name of the score column, auto-detecting if not specified."""
    if hint:
        if hint in df.columns:
            return hint
        raise ValueError(
            f"Specified score column '{hint}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    for candidate in ('score', 'activity', 'Score', 'Activity'):
        if candidate in df.columns:
            return candidate

    # Fallback: first numeric column that is not a metadata column
    for col in df.columns:
        if col not in _NON_SCORE_COLS and pd.api.types.is_numeric_dtype(df[col]):
            print(f"  Auto-detected score column: '{col}'")
            return col

    raise ValueError(
        f"Cannot identify score column. Use --score_col to specify. "
        f"Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# Format-specific parsers  →  normalised long DataFrame
#                              columns: Position (int), Amino_Acid (str), score (float)
# ---------------------------------------------------------------------------

def parse_hgvs(df, score_col):
    """
    Parse MaveDB HGVS hgvs_pro notation: p.Met1Ala, p.Met1Ter …
    Skips multi-mutant variants (brackets / semicolons).
    Collects wild-type scores from pure p.= and pure synonymous variants.

    Returns (df_long, wt_info) where wt_info is a dict:
      'p_equals':    list of scores for pure p.= variants
      'synonymous':  list of scores for pure synonymous (ref==alt) variants
      'pos_to_ref':  dict mapping position (int) -> ref amino acid (3-letter)
    """
    rows = []
    n_multi = n_wt_equals = n_pos_syn = n_invalid = 0
    wt_info = {'p_equals': [], 'synonymous': [], 'pos_to_ref': {}}

    for _, row in df.iterrows():
        hgvs  = str(row['hgvs_pro']).strip()
        score = row[score_col]

        # Multi-mutant: p.[Val2Asp;=] etc — skip entirely, NOT wild-type
        if '[' in hgvs or ';' in hgvs:
            n_multi += 1
            continue

        if hgvs.startswith('p.'):
            hgvs = hgvs[2:]

        # Pure p.= (wild-type, no position info)
        if hgvs == '=':
            n_wt_equals += 1
            try:
                wt_info['p_equals'].append(float(score))
            except (ValueError, TypeError):
                pass
            continue

        # Position-specific synonymous: Met1= (from p.Met1=)
        m_syn = re.match(r'^([A-Z][a-z]{2})(\d+)=$', hgvs)
        if m_syn:
            ref_aa, pos = m_syn.groups()
            pos = int(pos)
            n_pos_syn += 1
            wt_info['pos_to_ref'][pos] = ref_aa
            try:
                wt_info['synonymous'].append(float(score))
            except (ValueError, TypeError):
                pass
            continue

        # Standard single AA substitution: Met1Ala, Ala131Ter, Ala131Ala
        m = re.match(r'^([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', hgvs)
        if m:
            ref_aa, pos, mut_aa = m.groups()
            pos = int(pos)
            wt_info['pos_to_ref'][pos] = ref_aa
            # Synonymous (e.g. Ala131Ala): add to VEM AND collect for WT median
            if ref_aa == mut_aa:
                try:
                    wt_info['synonymous'].append(float(score))
                except (ValueError, TypeError):
                    pass
            rows.append({'Position': pos, 'Amino_Acid': mut_aa, 'score': score})
        else:
            n_invalid += 1

    if n_multi:     print(f"  Skipped {n_multi} multi-mutant variants")
    if n_wt_equals: print(f"  Found {n_wt_equals} p.= wild-type variant(s)")
    if n_pos_syn:   print(f"  Found {n_pos_syn} position-specific synonymous variant(s)")
    if n_invalid:   print(f"  Skipped {n_invalid} variants with unrecognised notation")

    return pd.DataFrame(rows), wt_info


def parse_simple(df, score_col):
    """
    Parse simple 1-letter notation: M1A, M1*, M1=
    Column may be named 'var' (Parkin supplementary) or 'aa_substitutions' (ADSL).
    Collects wild-type scores from WT rows, synonymous (M1=), and ref==alt (M1M).

    Returns (df_long, wt_info) — same structure as parse_hgvs.
    """
    var_col = 'var' if 'var' in df.columns else 'aa_substitutions'
    rows = []
    n_wt = n_syn = n_invalid = 0
    wt_info = {'p_equals': [], 'synonymous': [], 'pos_to_ref': {}}

    for _, row in df.iterrows():
        var   = str(row[var_col]).strip()
        score = row[score_col]

        # Empty aa_substitutions — WT barcode (no AA change, or silent codon change)
        if var in ('', 'nan', 'NA', 'NaN'):
            n_wt += 1
            try:
                wt_info['p_equals'].append(float(score))
            except (ValueError, TypeError):
                pass
            continue

        # WT label — global wild-type, no position info
        if var.lower() in ('wt', 'wildtype', 'wild-type', 'wild_type'):
            n_wt += 1
            try:
                wt_info['p_equals'].append(float(score))
            except (ValueError, TypeError):
                pass
            continue

        # Known summary rows — skip silently
        if var.lower() in ('synonymous', 'aa_substitutions'):
            continue

        # Synonymous: M1=
        m_syn = re.match(r'^([A-Z])(\d+)=$', var)
        if m_syn:
            ref_1, pos = m_syn.groups()
            ref_3 = ONE_TO_THREE.get(ref_1)
            if ref_3:
                pos = int(pos)
                wt_info['pos_to_ref'][pos] = ref_3
                try:
                    wt_info['synonymous'].append(float(score))
                except (ValueError, TypeError):
                    pass
            n_syn += 1
            continue

        # Regular variant: M1A, M1*
        m = re.match(r'^([A-Z])(\d+)([A-Z*])$', var)
        if m:
            ref_1, pos, mut_1 = m.groups()
            pos = int(pos)
            ref_3 = ONE_TO_THREE.get(ref_1)
            mut_3 = ONE_TO_THREE.get(mut_1)
            if ref_3:
                wt_info['pos_to_ref'][pos] = ref_3
            if mut_3:
                # Synonymous ref==alt (e.g. M1M): add to VEM AND collect for WT
                if ref_3 and ref_3 == mut_3:
                    try:
                        wt_info['synonymous'].append(float(score))
                    except (ValueError, TypeError):
                        pass
                rows.append({'Position': pos, 'Amino_Acid': mut_3, 'score': score})
            else:
                n_invalid += 1
        else:
            n_invalid += 1

    if n_wt:      print(f"  Found {n_wt} WT row(s)")
    if n_syn:     print(f"  Found {n_syn} synonymous variant(s)")
    if n_invalid: print(f"  Skipped {n_invalid} variants with unrecognised notation")

    return pd.DataFrame(rows), wt_info


# ---------------------------------------------------------------------------
# Wild-type score determination and diagonal filling
# ---------------------------------------------------------------------------

def determine_wt_score(wt_info, wt_score_override=None):
    """
    Determine the wild-type score for filling diagonal cells.

    Priority:
      1. --wt_score override (if provided by user)
      2. Median of p.= scores (if present in data)
      3. Median of synonymous scores (if present in data)
      4. None (leave diagonal as NaN, warn user)
    """
    if wt_score_override is not None:
        print(f"  WT score: {wt_score_override} (user-specified via --wt_score)")
        return wt_score_override

    if wt_info['p_equals']:
        wt_score = float(np.mean(wt_info['p_equals']))
        n = len(wt_info['p_equals'])
        print(f"  WT score: {wt_score:.6f} (mean of {n} p.= / WT barcode(s))")
        return wt_score

    if wt_info['synonymous']:
        wt_score = float(np.mean(wt_info['synonymous']))
        n = len(wt_info['synonymous'])
        print(f"  WT score: {wt_score:.6f} (mean of {n} synonymous variant(s))")
        return wt_score

    print("  WARNING: No wild-type scores found in data. Diagonal will remain NaN.")
    print("  Use --wt_score to specify a wild-type score manually.")
    return None


def fill_wt_diagonal(vem, pos_to_ref, wt_score):
    """
    Fill NaN diagonal cells (where column AA == ref AA at that position)
    with the determined wild-type score.
    """
    if wt_score is None:
        return vem

    n_filled = 0
    for pos in vem.index:
        ref_aa = pos_to_ref.get(int(pos))
        if ref_aa and ref_aa in vem.columns:
            if pd.isna(vem.loc[pos, ref_aa]):
                vem.loc[pos, ref_aa] = wt_score
                n_filled += 1

    n_already = sum(
        1 for pos in vem.index
        if pos_to_ref.get(int(pos)) and pos_to_ref.get(int(pos)) in vem.columns
        and not pd.isna(vem.loc[pos, pos_to_ref[int(pos)]])
    ) - n_filled  # subtract the ones we just filled
    n_no_ref = len(vem.index) - len([p for p in vem.index if pos_to_ref.get(int(p))])

    print(f"  Filled {n_filled} diagonal (wild-type) cells with score = {wt_score:.6f}")
    if n_already > 0:
        print(f"  ({n_already} diagonal cells already had measured values — kept)")
    if n_no_ref > 0:
        print(f"  ({n_no_ref} positions have unknown ref AA — diagonal not filled)")

    return vem


# ---------------------------------------------------------------------------
# Pivot  →  VEM wide matrix
# ---------------------------------------------------------------------------

def pivot_to_vem(df_long):
    """
    Aggregate duplicate (Position, Amino_Acid) pairs by mean score and
    pivot to the standard VEM wide matrix.

    The output always covers:
    - Every integer position from min to max (gaps filled with NaN)
    - Every amino acid in DESIRED_COL_ORDER (missing columns filled with NaN)
    This ensures the full data extent is visible when plotting.
    """
    df_agg = (
        df_long
        .groupby(['Position', 'Amino_Acid'])['score']
        .mean()
        .reset_index()
    )
    vem = df_agg.pivot(index='Position', columns='Amino_Acid', values='score')
    vem.index.name   = 'Position'
    vem.columns.name = None

    # Fill in any missing positions across the full range
    full_range = range(int(vem.index.min()), int(vem.index.max()) + 1)
    vem = vem.reindex(full_range)

    # Always include all amino acids from DESIRED_COL_ORDER, even if all-NaN
    for col in DESIRED_COL_ORDER:
        if col not in vem.columns:
            vem[col] = float('nan')
    return vem.reindex(columns=DESIRED_COL_ORDER)


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------

def to_vem(file_path, score_col_hint=None, wt_score_override=None):
    """Load any supported MAVE score file and return a VEM wide-matrix DataFrame."""
    sep = sniff_separator(file_path)
    df  = pd.read_csv(file_path, sep=sep)
    fmt = detect_format(df)
    print(f"  Detected format : {fmt}")

    # --- Format 1: already wide (3-letter headers), just reorder columns and return ---
    if fmt == 'vem_wide':
        df = df.set_index('Position')
        has_wt_aa = 'wt_aa' in df.columns
        wt_aa_col = df['wt_aa'].copy() if has_wt_aa else None
        ordered = [c for c in DESIRED_COL_ORDER if c in df.columns]
        df = df.reindex(columns=ordered)
        if has_wt_aa:
            df.insert(0, 'wt_aa', wt_aa_col)
        return df

    # --- Format 6: wide matrix with 1-letter AA headers ---
    if fmt == 'vem_wide_1letter':
        df = df.rename(columns={df.columns[0]: 'Position'})
        df = df.set_index('Position')
        # Keep only columns that are valid 1-letter AA codes, drop extras (wt_aa, mean_score …)
        aa_cols = [c for c in df.columns if c in ONE_LETTER_AAS]
        df = df[aa_cols]
        # Rename 1-letter → 3-letter
        df = df.rename(columns=ONE_TO_THREE)
        # Reindex to full position range and standard column order
        full_range = range(int(df.index.min()), int(df.index.max()) + 1)
        df = df.reindex(full_range)
        for col in DESIRED_COL_ORDER:
            if col not in df.columns:
                df[col] = float('nan')
        return df.reindex(columns=DESIRED_COL_ORDER)

    # --- Format 7: headerless — re-read without header, use col 0 as 'var', col 1 as 'score' ---
    if fmt == 'headerless':
        df = pd.read_csv(file_path, sep=sep, header=None)
        df = df.rename(columns={df.columns[0]: 'var', df.columns[1]: 'score'})
        df['score'] = pd.to_numeric(df['score'], errors='coerce')
        fmt = 'simple'   # now parse as simple 1-letter format
        print(f"  Re-reading as   : simple (headerless file, using col 1 as var, col 2 as score)")

    # --- All long formats: find score column (keep NA scores — they become NaN in the matrix) ---
    score_col = find_score_col(df, hint=score_col_hint)
    print(f"  Score column    : '{score_col}'")

    # --- Parse to normalised long form ---
    wt_info = {'p_equals': [], 'synonymous': [], 'pos_to_ref': {}}

    if fmt == 'hgvs':
        df_long, wt_info = parse_hgvs(df, score_col)

    elif fmt == 'preparsed_2':
        # ADSLcsv2tsv.sh output: Position, Amino_Acid_2 already in 3-letter
        df_long = df[['Position', 'Amino_Acid_2', score_col]].copy()
        df_long.columns = ['Position', 'Amino_Acid', 'score']
        df_long['Position'] = df_long['Position'].astype(int)

    elif fmt == 'preparsed':
        # VEM2tsv.py output: Position, Amino_Acid already in 3-letter
        df_long = df[['Position', 'Amino_Acid', score_col]].copy()
        df_long.columns = ['Position', 'Amino_Acid', 'score']
        df_long['Position'] = df_long['Position'].astype(int)

    elif fmt == 'simple':
        df_long, wt_info = parse_simple(df, score_col)

    vem = pivot_to_vem(df_long)

    # --- Fill wild-type diagonal ---
    wt_score = determine_wt_score(wt_info, wt_score_override)
    vem = fill_wt_diagonal(vem, wt_info['pos_to_ref'], wt_score)

    # --- Add wt_aa column (ref AA per position, empty for gap positions) ---
    pos_to_ref = wt_info['pos_to_ref']
    vem.insert(0, 'wt_aa', [pos_to_ref.get(int(p), '') for p in vem.index])

    return vem


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_output_path(input_path):
    """Derive a sensible output path, stripping known extensions before adding .VEM.tsv."""
    base = input_path
    for ext in ('.VEM.tsv', '.VEM.txt', '.tsv', '.csv', '.txt', '.tab'):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    return base + '.VEM.tsv'


def main():
    """CLI entry point for to_vem."""
    parser = argparse.ArgumentParser(
        description='Convert MAVE score files of any supported format to VEM.tsv.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('-i', '--input_file',  required=True,
                        help='Path to input file.')
    parser.add_argument('-o', '--output_file', required=False,
                        help='Path to output VEM.tsv (default: <input>.VEM.tsv).')
    parser.add_argument('--score_col',         required=False,
                        help='Name of the score column (auto-detected if omitted).')
    parser.add_argument('--wt_score',          required=False, type=float,
                        help='Wild-type score to fill diagonal cells (auto-detected if omitted).')
    args = parser.parse_args()

    print(f'\nInput:  {args.input_file}')
    vem = to_vem(args.input_file, score_col_hint=args.score_col, wt_score_override=args.wt_score)

    out_path = args.output_file or build_output_path(args.input_file)
    vem.to_csv(out_path, sep='\t', index_label='Position')

    print(f'Output: {out_path}')
    print(f'Shape:  {vem.shape[0]} positions × {vem.shape[1]} amino acids\n')


if __name__ == '__main__':
    main()
