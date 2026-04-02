#!/usr/bin/env python
# coding: utf-8
"""
mavepolish_core.py — Analysis engine for MavePolish.

Refactored from maveqc_v7.py: no argparse, no global state, no file I/O side effects.
All functions accept explicit parameters and return results as DataFrames/dicts.
"""

import warnings
import pickle
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import gaussian_kde
from sklearn.decomposition import DictionaryLearning, PCA, SparseCoder

# Suppress benign OMP warning about linear dependence in dictionary
warnings.filterwarnings('ignore', message='.*Orthogonal matching pursuit ended prematurely.*')


def kde_wt_peak(values):
    """
    Estimate the wild-type score from a distribution of variant scores
    using KDE peak detection with prominence ranking.

    Returns the rightmost of the two most prominent peaks, which corresponds
    to the WT/neutral mode in a typical MAVE bimodal (or multimodal) distribution.
    Falls back to median if no peaks are found.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 10:
        return float(np.median(vals))

    kde = gaussian_kde(vals)
    x_grid = np.linspace(vals.min(), vals.max(), 1000)
    density = kde(x_grid)

    peaks, _ = find_peaks(density)
    if len(peaks) == 0:
        return float(np.median(vals))
    if len(peaks) == 1:
        return float(x_grid[peaks[0]])

    prominences, _, _ = peak_prominences(density, peaks)
    top2_idx = np.argsort(prominences)[-2:]
    top2_peaks = peaks[top2_idx]
    return float(x_grid[top2_peaks].max())


def handle_missing_values(dataframe):
    """
    Fill missing values using a hybrid strategy:
    - Ter column (if present): fill with column mean (stop codons have a distinct pattern)
    - All other columns: fill with row-wise mean (missense AAs correlate within position)
    """
    if 'Ter' in dataframe.columns and dataframe['Ter'].isna().any():
        dataframe['Ter'] = dataframe['Ter'].fillna(dataframe['Ter'].mean())
    dataframe = dataframe.apply(lambda row: row.fillna(row.mean()), axis=1)
    # If any rows are still all-NaN (e.g. originally completely missing), fill them with 0s
    dataframe.loc[dataframe.isna().all(axis=1)] = 0
    return dataframe


def reorder_amino_acid_columns(X):
    """
    Reorder columns to standard amino acid order and optionally remove 'Ter' if coverage is too low.
    """
    full_ordered_cols = [
        'Ter', 'Pro', 'Gly', 'Tyr', 'Trp', 'Phe', 'Val', 'Leu', 'Ile',
        'Ala', 'Thr', 'Ser', 'Gln', 'Asn', 'Met', 'Cys', 'Glu', 'Asp',
        'Arg', 'Lys', 'His'
    ]

    stop_threshold = 0.2
    include_ter = 'Ter' in X.columns and X['Ter'].notna().mean() >= stop_threshold

    if not include_ter:
        full_ordered_cols = full_ordered_cols[1:]  # drop 'Ter'

    # Keep only columns that exist in X, in the defined order
    reordered_cols = [col for col in full_ordered_cols if col in X.columns]
    X = X.reindex(columns=reordered_cols)

    return X


def determine_rescaling_factor(X, target_iqr=1.0):
    """
    Determine a rescaling factor that brings the data IQR to target_iqr.
    """
    q1 = X.stack().quantile(0.25)
    q3 = X.stack().quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 1.0
    return target_iqr / iqr


def rec_error(Y_hat, Y, valid_rows, row_norms):
    """Compute mean normalised reconstruction error over valid (non-zero-variance) rows."""
    return np.mean(np.sum((Y_hat - Y) ** 2, axis=1)[valid_rows] / row_norms[valid_rows])


def load_pretrained_model(model_path):
    """Load a pretrained dictionary learning model from a .pkl file.

    Returns (components, expected_cols) where components is the dictionary
    matrix and expected_cols is the list of amino acid column names.
    """
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    dl = model_data['model']
    return dl.components_, list(dl.feature_names_in_)


def save_model(dict_learner, model_path):
    """Save a trained DictionaryLearning model to a .pkl file."""
    with open(model_path, 'wb') as f:
        pickle.dump({'model': dict_learner}, f)


def _preprocess_vem(vem_df, expected_cols=None):
    """Shared preprocessing: separate wt_aa, center, KDE-fill WT diagonal,
    reorder columns, fill remaining NaN.

    If *expected_cols* is given (pretrained model), columns are reordered to
    match the model (Ter is kept even if sparse).  Otherwise columns are
    reordered in standard order and Ter is dropped if coverage < 20%.

    Returns (Y, Y_original_raw, Y_nan_mask, wt_aa, Ymean_global, kde_wt).
    """
    Y = vem_df.copy()
    wt_aa = None
    if 'wt_aa' in Y.columns:
        wt_aa = Y['wt_aa'].copy()
        Y = Y.drop(columns=['wt_aa'])

    Y_original_raw = Y.copy()

    Ymean_global = Y.stack().mean()
    Y = Y - Ymean_global

    Y_nan_mask = Y.isna()

    # --- Fill WT diagonal with KDE estimate if missing ---
    kde_wt = None
    if wt_aa is not None:
        wt_nan_count = sum(
            1 for pos in Y.index
            if wt_aa.get(pos, '') and wt_aa.get(pos, '') in Y.columns
            and pd.isna(Y.loc[pos, wt_aa[pos]])
        )
        if wt_nan_count > 0:
            Y_no_diag = Y.copy()
            for pos in Y_no_diag.index:
                ref = wt_aa.get(pos, '')
                if ref and ref in Y_no_diag.columns:
                    Y_no_diag.loc[pos, ref] = np.nan
            real_vals = Y_no_diag.stack().dropna().values
            if len(real_vals) >= 10:
                kde_wt = kde_wt_peak(real_vals)
                for pos in Y.index:
                    ref = wt_aa.get(pos, '')
                    if ref and ref in Y.columns and pd.isna(Y.loc[pos, ref]):
                        Y.loc[pos, ref] = kde_wt

    # --- Column ordering ---
    if expected_cols is not None:
        # Pretrained model: match its columns exactly (add missing as NaN)
        for col in expected_cols:
            if col not in Y.columns:
                Y[col] = float('nan')
        Y = Y.reindex(columns=expected_cols)
        Y_nan_mask = Y_nan_mask.reindex(columns=expected_cols, fill_value=True)
    else:
        # Self-trained: standard order, drop Ter if sparse (before NaN fill)
        Y = reorder_amino_acid_columns(Y)
        Y_nan_mask = Y_nan_mask.reindex_like(Y)

    # --- Fill remaining NaN values ---
    Y = handle_missing_values(Y)

    Y_original_raw = Y_original_raw.reindex(columns=Y.columns)

    return Y, Y_original_raw, Y_nan_mask, wt_aa, Ymean_global, kde_wt


def run_pretrained(vem_df, model_path, n_components=6):
    """Fast reconstruction using a pretrained dictionary model.

    Skips dictionary training — only does OMP sparse coding + error computation.
    """
    data_dict, expected_cols = load_pretrained_model(model_path)

    Y, Y_original_raw, Y_nan_mask, wt_aa, Ymean_global, kde_wt = \
        _preprocess_vem(vem_df, expected_cols=expected_cols)

    # --- Row norms ---
    row_norms = np.sum(Y ** 2, axis=1)
    valid_rows = row_norms > 0

    # --- Reconstruct with pretrained dictionary ---
    coder = SparseCoder(
        dictionary=data_dict,
        transform_algorithm='omp',
        transform_n_nonzero_coefs=n_components
    )
    Y_trans = coder.transform(Y.values)
    Y_hat = pd.DataFrame(Y_trans @ data_dict, columns=Y.columns, index=Y.index)

    # --- Error ---
    err_pretrained = float(rec_error(Y_hat, Y, valid_rows, row_norms))

    # --- Restore WT diagonal from preprocessed values ---
    # These are either the original measured values or KDE estimates
    if wt_aa is not None:
        for pos in Y_hat.index:
            ref = wt_aa.get(pos, '')
            if ref and ref in Y_hat.columns:
                Y_hat.loc[pos, ref] = Y.loc[pos, ref]

    Y_hat_orig = Y_hat + Ymean_global

    # --- Drop Ter from output if original data had sparse Ter coverage ---
    # (pretrained model needs 21 cols for reconstruction, but output should
    # match what the data actually contains)
    if 'Ter' in Y_nan_mask.columns:
        ter_coverage = (~Y_nan_mask['Ter']).mean()
        if ter_coverage < 0.20:
            drop_cols = [c for c in Y_hat_orig.columns if c != 'Ter']
            Y_hat_orig = Y_hat_orig[drop_cols]
            Y_original_raw = Y_original_raw[drop_cols]
            Y_nan_mask = Y_nan_mask[drop_cols]

    return {
        'original': Y_original_raw,
        'pretrained_recon': Y_hat_orig,
        'err_pretrained': err_pretrained,
        'nan_mask': Y_nan_mask,
        'global_mean': float(Ymean_global),
        'wt_aa': wt_aa,
        'columns': list(Y_hat_orig.columns),
        'kde_wt': float(kde_wt + Ymean_global) if kde_wt is not None else None,
    }


def run_mavepolish(vem_df, target_iqr=1.0, n_components=6):
    """
    Full MavePolish pipeline: preprocess VEM, train dictionary + PCA, reconstruct, return results.

    Args:
        vem_df:        DataFrame from to_vem() (includes wt_aa column)
        target_iqr:    rescaling target for dictionary fitting (default 1.0)
        n_components:  number of dictionary/PCA components (default 6)

    Returns:
        dict with keys:
            'original'          DataFrame — original data, original scale, NaN preserved
            'dict_recon'        DataFrame — dictionary reconstruction, original scale
            'pca_recon'         DataFrame — PCA reconstruction, original scale
            'naive_recon'       DataFrame — naive mean reconstruction, original scale
            'err_dict'          float — dictionary reconstruction error
            'err_pca'           float — PCA reconstruction error
            'err_naive'         float — naive mean reconstruction error
            'nan_mask'          DataFrame — True where original had NaN
            'global_mean'       float — centering value
            'n_iterations'      int — DictionaryLearning iterations
            'rescaling_factor'  float — data rescaling factor
            'wt_aa'             Series — reference amino acid per position
            'columns'           list — ordered amino acid column names
    """
    Y, Y_original_raw, Y_nan_mask, wt_aa, Ymean_global, kde_wt = \
        _preprocess_vem(vem_df)

    # --- Row norms for error computation ---
    row_norms = np.sum(Y ** 2, axis=1)
    valid_rows = row_norms > 0

    # --- Train: Dictionary Learning ---
    X_scale = determine_rescaling_factor(Y, target_iqr=target_iqr)
    X_scaled = Y * X_scale

    dict_learner = DictionaryLearning(
        n_components=n_components,
        transform_algorithm='lasso_lars',
        transform_alpha=0.1,
        random_state=42
    )
    dict_learner.fit(X_scaled)
    data_dict = dict_learner.components_
    n_iterations = dict_learner.n_iter_

    # --- Score: Dictionary reconstruction (OMP on unscaled data) ---
    coder = SparseCoder(
        dictionary=data_dict,
        transform_algorithm='omp',
        transform_n_nonzero_coefs=n_components
    )
    Y_trans = coder.transform(Y.values)
    Y_hat = pd.DataFrame(Y_trans @ data_dict, columns=Y.columns, index=Y.index)

    # --- Score: PCA reconstruction ---
    pca_local = PCA(n_components=n_components)
    pca_local.fit(Y)
    Y_pca_hat = pd.DataFrame(
        pca_local.inverse_transform(pca_local.transform(Y)),
        columns=Y.columns, index=Y.index
    )

    # --- Score: Naive mean reconstruction ---
    Y_means = np.mean(Y, axis=1).to_numpy()[:, np.newaxis]
    Y_mean_hat = pd.DataFrame(
        np.tile(Y_means, (1, Y.shape[1])),
        columns=Y.columns, index=Y.index
    )

    # --- Reconstruction errors ---
    err_dict = float(rec_error(Y_hat, Y, valid_rows, row_norms))
    err_pca = float(rec_error(Y_pca_hat, Y, valid_rows, row_norms))
    err_naive = float(rec_error(Y_mean_hat, Y, valid_rows, row_norms))

    # --- Restore WT values on diagonal of dictionary reconstruction ---
    # These are either the original measured values or KDE estimates
    if wt_aa is not None:
        for pos in Y_hat.index:
            ref = wt_aa.get(pos, '')
            if ref and ref in Y_hat.columns:
                Y_hat.loc[pos, ref] = Y.loc[pos, ref]

    # --- Convert all reconstructions back to original scale ---
    Y_hat_orig = Y_hat + Ymean_global
    Y_pca_hat_orig = Y_pca_hat + Ymean_global
    Y_mean_hat_orig = Y_mean_hat + Ymean_global

    return {
        'original': Y_original_raw,
        'preprocessed': X_scaled,
        'dict_recon': Y_hat_orig,
        'pca_recon': Y_pca_hat_orig,
        'naive_recon': Y_mean_hat_orig,
        'err_dict': err_dict,
        'err_pca': err_pca,
        'err_naive': err_naive,
        'nan_mask': Y_nan_mask,
        'global_mean': float(Ymean_global),
        'n_iterations': n_iterations,
        'rescaling_factor': float(X_scale),
        'dict_learner': dict_learner,
        'wt_aa': wt_aa,
        'columns': list(Y.columns),
        'kde_wt': float(kde_wt + Ymean_global) if kde_wt is not None else None,
    }
