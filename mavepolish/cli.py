#!/usr/bin/env python
# coding: utf-8
"""
mavepolish — Command-line interface for MAVEpolish analysis.

Runs dictionary learning + PCA quality control on VEM files,
using the same analysis engine as the MAVEpolish web app.
"""

import os
import copy
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for cluster jobs
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from mavepolish.core import run_pretrained, run_mavepolish
from mavepolish.to_vem import to_vem


# ---------------------------------------------------------------------------
# Plotting (CLI-only — the web app has its own Plotly-based plots)
# ---------------------------------------------------------------------------

def plot_distributions(original, dict_recon, pca_recon, naive_recon,
                       nan_mask, base_name, out_dir):
    """Save a 4-panel distribution histogram (PDF)."""
    real_mask = ~nan_mask.values
    panels = [
        (original, 'Input'),
        (dict_recon, 'Dictionary'),
        (pca_recon, 'PCA'),
        (naive_recon, 'Naive'),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for ax, (data, label) in zip(axes, panels):
        vals = data.values[real_mask]
        vals = vals[~np.isnan(vals)]
        ax.hist(vals, bins=80, density=True, color='steelblue')
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(label, fontsize=14)
    axes[-1].set_xlabel("Value", fontsize=12)
    fig.suptitle(f"Distributions: {base_name}", y=1.01, fontsize=16)
    fig.tight_layout()
    path = os.path.join(out_dir, f'{base_name}_distributions.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_vem_heatmaps(original, dict_recon, pca_recon, naive_recon,
                      nan_mask, err_dict, err_pca, err_naive,
                      base_name, out_dir):
    """Save a 4-panel VEM heatmap (PDF)."""
    # Colour scale: 1st/99th percentiles
    vals = original.stack().dropna().values
    Ymin = np.percentile(vals, 1)
    Ymax = np.percentile(vals, 99)

    # Detect WT mode for centering
    kde = gaussian_kde(vals)
    x_grid = np.linspace(vals.min(), vals.max(), 1000)
    density = kde(x_grid)
    peaks, _ = find_peaks(density, prominence=0.05 * density.max())
    if len(peaks) >= 2:
        center_val = x_grid[peaks[-1]]
    elif len(peaks) == 1:
        center_val = x_grid[peaks[0]]
    else:
        center_val = 0
    print(f'VEM heatmap center: {center_val:.4f}')
    print(f'VEM colour range:   [{Ymin:.4f}, {Ymax:.4f}] (1st-99th percentile)')

    cmap = copy.copy(plt.get_cmap('seismic'))
    cmap.set_bad(color='lightgray')

    cell_size = 0.25
    n_pos = original.shape[0]
    n_aa = original.shape[1]
    fig_width = n_pos * cell_size + 3
    fig_height = n_aa * cell_size * 4 + 3

    pos_labels = [str(p) if i % 2 == 0 else '' for i, p in enumerate(original.index)]
    errors = [None, err_dict, err_pca, err_naive]

    fig, axes = plt.subplots(4, 1, figsize=(fig_width, fig_height))
    for ax, data, title, err in zip(axes,
        [original, dict_recon, pca_recon, naive_recon],
        ['Original data', 'Dictionary model', 'PCA model', 'Naive mean model'],
        errors):
        data_plot = data.copy()
        data_plot[nan_mask] = np.nan
        sns.heatmap(data_plot.T, annot=False, cmap=cmap, center=center_val,
                    vmin=Ymin, vmax=Ymax, ax=ax,
                    square=True,
                    cbar_kws={'shrink': 0.7, 'pad': 0.005},
                    xticklabels=pos_labels, yticklabels=True)
        if err is not None:
            ax.set_title(f'{title}  (rec. error = {err:.4f})', fontsize=72)
        else:
            ax.set_title(title, fontsize=72)
        ax.tick_params(axis='x', labelsize=12, rotation=90)
        ax.tick_params(axis='y', labelsize=12, rotation=0)
    fig.tight_layout()
    path = os.path.join(out_dir, f'{base_name}_VEM.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ---------------------------------------------------------------------------
# Process a single VEM file
# ---------------------------------------------------------------------------

def process_file(file_path, model_path=None, nan_handling='Mean',
                 target_iqr=1.0, out_dir=None, do_plot=True):
    """Run analysis on one VEM file and save outputs."""

    # Read VEM file
    vem_df = to_vem(file_path)

    # Derive output base name
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    if base_name.endswith('.VEM'):
        base_name = base_name[:-4]
    if out_dir is None:
        out_dir = os.path.dirname(file_path) or '.'
    os.makedirs(out_dir, exist_ok=True)

    if model_path:
        # --- Pretrained mode ---
        results = run_pretrained(vem_df, model_path, nan_handling=nan_handling)

        print(f'{model_path}\t{file_path}\tReconstruction_error_dict\t'
              f'{results["err_pretrained"]:.5f}')
        print()

        # Save reconstructed VEM
        dict_path = os.path.join(out_dir, f'{base_name}.dict.tsv')
        results['pretrained_recon'].to_csv(dict_path, sep='\t',
                                            index_label='Position')
        print(f'Saved: {dict_path}')

        # Pretrained mode only has one reconstruction — skip 4-panel plots

    else:
        # --- Self-trained mode ---
        results = run_mavepolish(vem_df, nan_handling=nan_handling,
                                 target_iqr=target_iqr)

        print(f'{file_path}\t{file_path}\tReconstruction_error_dict\t'
              f'{results["err_dict"]:.5f}')
        print(f'{file_path}\t{file_path}\tReconstruction_error_PCA\t'
              f'{results["err_pca"]:.5f}')
        print(f'{file_path}\t{file_path}\tReconstruction_error_naive\t'
              f'{results["err_naive"]:.5f}')
        print()

        # Save reconstructed VEM
        dict_path = os.path.join(out_dir, f'{base_name}.dict.tsv')
        results['dict_recon'].to_csv(dict_path, sep='\t',
                                      index_label='Position')
        print(f'Saved: {dict_path}')

        if do_plot:
            original = results['original']
            nan_mask = results['nan_mask']
            dict_recon = results['dict_recon']
            pca_recon = results['pca_recon']
            naive_recon = results['naive_recon']

            plot_distributions(original, dict_recon, pca_recon, naive_recon,
                               nan_mask, base_name, out_dir)
            plot_vem_heatmaps(original, dict_recon, pca_recon, naive_recon,
                              nan_mask,
                              results['err_dict'], results['err_pca'],
                              results['err_naive'],
                              base_name, out_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    print('')
    print("************************************************")
    print("*                                              *")
    print("*          MAVEpolish Analysis Tool             *")
    print("*                                              *")
    print("************************************************")
    print('')

    parser = argparse.ArgumentParser(description='MAVEpolish analysis.')
    parser.add_argument('--nan_handling', type=str, default='Mean',
                        choices=['Mean', 'Median', 'Zeros'],
                        help='Method for handling missing values')
    parser.add_argument('-t', '--training_file', type=str, required=False,
                        help='Path to the training file in VEM.tsv format.')
    parser.add_argument('-m', '--model_file', type=str, required=False,
                        help='Path to a pretrained model (.pkl). Use instead of -t.')
    parser.add_argument('-e', '--test_file', type=str, required=False,
                        help='Path to the test file in VEM.tsv format.')
    parser.add_argument('-l', '--test_file_list', type=str, required=False,
                        help='Path to a list of test file names (one per line).')
    parser.add_argument('-o', '--output_dir', type=str, required=False,
                        help='Output directory for results (default: same as input).')
    parser.add_argument('--no_plot', dest='plot', action='store_false',
                        help='Disable saving plot files')
    parser.add_argument('--target_iqr', type=float, default=1.0,
                        help='Target IQR for rescaling training data (default: 1.0)')
    parser.set_defaults(plot=True)
    args = parser.parse_args()

    # Validate arguments
    if not args.model_file and not args.training_file:
        parser.error('Either -t (training file) or -m (pretrained model) is required.')
    if args.model_file and args.training_file:
        parser.error('Use either -t (training file) or -m (pretrained model), not both.')

    # Print parameter summary
    print('Parameters:')
    print(f'  nan_handling:  {args.nan_handling}')
    if args.model_file:
        print(f'  model:         {args.model_file} (pretrained)')
    else:
        print(f'  target_iqr:    {args.target_iqr}')
        print(f'  training_file: {args.training_file}')
    print()

    # Collect test files
    test_files = []
    if args.test_file:
        test_files.append(args.test_file)
    if args.test_file_list:
        with open(args.test_file_list, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    test_files.append(line)

    # Process each test file
    for fpath in test_files:
        process_file(
            fpath,
            model_path=args.model_file,
            nan_handling=args.nan_handling,
            target_iqr=args.target_iqr,
            out_dir=args.output_dir,
            do_plot=args.plot,
        )


if __name__ == '__main__':
    main()
