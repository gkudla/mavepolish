#!/usr/bin/env python
# coding: utf-8
"""
MAVEpolish — Quality Control & Reconstruction for MAVE Data

A Dash web application that lets users upload raw MAVE data in any format,
converts it to VEM format, runs dictionary learning + PCA quality control,
and displays interactive heatmaps with downloadable reconstructed data.
"""

import os
import io
import base64
import tempfile
import json

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

import threading

from mavepolish.core import kde_wt_peak, run_pretrained, run_mavepolish

from mavepolish.to_vem import to_vem, sniff_separator, detect_format, find_score_col

# Path to the pretrained model (shipped with the app)
PRETRAINED_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'pretrained_model.pkl'
)

# Load example datasets manifest
EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples')
with open(os.path.join(EXAMPLES_DIR, 'manifest.json')) as _f:
    EXAMPLE_DATASETS = json.load(_f)

# Background results cache (filled by background thread)
_bg_results = {}       # key → results dict
_bg_lock = threading.Lock()


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "MAVEpolish"
server = app.server  # for Render / gunicorn


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
CARD_STYLE = {
    'backgroundColor': '#f8f9fa',
    'borderRadius': '6px',
    'padding': '10px 14px',
    'textAlign': 'center',
    'boxShadow': '0 1px 3px rgba(0,0,0,0.12)',
    'flex': '1',
    'minWidth': '140px',
}

HEADER_STYLE = {
    'textAlign': 'center',
    'padding': '30px 20px 10px',
    'backgroundColor': '#1a1a2e',
    'color': 'white',
    'marginBottom': '20px',
}

SECTION_STYLE = {
    'maxWidth': '1400px',
    'margin': '0 auto',
    'padding': '0 20px 20px',
}

UPLOAD_STYLE = {
    'width': '100%',
    'height': '100px',
    'lineHeight': '100px',
    'borderWidth': '2px',
    'borderStyle': 'dashed',
    'borderRadius': '8px',
    'borderColor': '#adb5bd',
    'textAlign': 'center',
    'cursor': 'pointer',
    'backgroundColor': '#f8f9fa',
    'fontSize': '16px',
    'color': '#495057',
}

BUTTON_STYLE = {
    'backgroundColor': '#0d6efd',
    'color': 'white',
    'border': 'none',
    'borderRadius': '6px',
    'padding': '12px 32px',
    'fontSize': '16px',
    'cursor': 'pointer',
    'marginTop': '10px',
}

BUTTON_DISABLED_STYLE = {
    **BUTTON_STYLE,
    'backgroundColor': '#6c757d',
    'cursor': 'not-allowed',
}

DOWNLOAD_BUTTON_STYLE = {
    'backgroundColor': '#198754',
    'color': 'white',
    'border': 'none',
    'borderRadius': '6px',
    'padding': '10px 24px',
    'fontSize': '14px',
    'cursor': 'pointer',
    'marginRight': '10px',
}


# ---------------------------------------------------------------------------
# Colorscale — replicates seaborn heatmap(center=...) from maveqc_v7
# ---------------------------------------------------------------------------
# seaborn computes: vrange = max(vmax - center, center - vmin)
# then uses Normalize(center - vrange, center + vrange).
# This makes the range symmetric around center (WT mode), so equal
# absolute distance from WT gives equal colour saturation.
# Typical MAVE data: many dark blues (loss-of-function), few dark reds.

SEISMIC = [
    [0.0,  '#00004c'],
    [0.25, '#0000ff'],
    [0.5,  '#ffffff'],
    [0.75, '#ff0000'],
    [1.0,  '#7f0000'],
]


def symmetric_color_range(zmin, zmax, center):
    """Expand zmin/zmax symmetrically around center (matching seaborn logic).

    Returns (eff_zmin, eff_zmax) so that white maps exactly to center
    and equal distance from center gives equal colour saturation.
    """
    max_dist = max(abs(center - zmin), abs(zmax - center))
    if max_dist == 0:
        max_dist = 1.0
    return center - max_dist, center + max_dist


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = html.Div([

    # --- Header ---
    html.Div([
        html.H1("MAVEpolish", style={'margin': '0', 'fontSize': '36px'}),
        html.P("Quality Control & Reconstruction for MAVE Data",
               style={'margin': '5px 0 0', 'fontSize': '16px', 'opacity': '0.8'}),
    ], style=HEADER_STYLE),

    # --- Main content ---
    html.Div([

        # === Upload & Options Panel ===
        html.Div([
            html.H3("Upload Data", style={'marginTop': '0'}),

            dcc.Tabs(id='data-source-tabs', value='upload', children=[
                dcc.Tab(label='Upload Your Data', value='upload', children=[
                    html.Div([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div(['Drag and drop a MAVE file here, or ',
                                               html.A('click to browse')]),
                            style=UPLOAD_STYLE,
                            multiple=False,
                        ),
                    ], style={'padding': '15px 0 5px'}),
                ], style={'padding': '10px'}, selected_style={'padding': '10px'}),

                dcc.Tab(label='Example Datasets', value='example', children=[
                    html.Div([
                        html.Label("Select a dataset:",
                                   style={'fontWeight': 'bold', 'marginBottom': '6px'}),
                        dcc.Dropdown(
                            id='example-dropdown',
                            options=[{'label': d['label'], 'value': d['id']}
                                     for d in EXAMPLE_DATASETS],
                            value=None,
                            placeholder='Choose an example dataset...',
                            style={'width': '100%'},
                        ),
                    ], style={'padding': '15px 0 5px'}),
                ], style={'padding': '10px'}, selected_style={'padding': '10px'}),
            ], style={'marginBottom': '5px'}),

            html.Div(id='upload-info', style={'marginTop': '10px', 'fontSize': '14px'}),

            # Options row
            html.Div([
                html.Div([
                    html.Label("NaN handling:", style={'fontWeight': 'bold', 'marginBottom': '4px'}),
                    dcc.Dropdown(
                        id='nan-handling',
                        options=[
                            {'label': 'Row mean', 'value': 'Mean'},
                            {'label': 'Row median', 'value': 'Median'},
                            {'label': 'Zeros', 'value': 'Zeros'},
                        ],
                        value='Mean',
                        clearable=False,
                        style={'width': '180px'},
                    ),
                ], style={'marginRight': '30px'}),

                html.Div([
                    html.Label("Score column:", style={'fontWeight': 'bold', 'marginBottom': '4px'}),
                    dcc.Dropdown(
                        id='score-col-dropdown',
                        options=[],
                        value=None,
                        placeholder='(auto-detect)',
                        style={'width': '280px'},
                    ),
                ], style={'marginRight': '30px'}),

            ], style={'display': 'flex', 'alignItems': 'flex-end', 'marginTop': '15px', 'flexWrap': 'wrap', 'gap': '10px'}),

            html.Button("Run Analysis", id='run-button', n_clicks=0,
                         style=BUTTON_DISABLED_STYLE, disabled=True),

        ], style={
            'backgroundColor': 'white',
            'borderRadius': '8px',
            'padding': '20px',
            'boxShadow': '0 1px 4px rgba(0,0,0,0.1)',
            'marginBottom': '20px',
        }),

        # === Hidden stores ===
        dcc.Store(id='uploaded-file-store'),   # {'filename': ..., 'content': base64}
        dcc.Store(id='results-store'),          # JSON-serialised analysis results
        dcc.Store(id='bg-run-id'),              # unique ID for background run
        dcc.Interval(id='bg-poll', interval=2000, disabled=True),  # poll for background results

        # === Loading spinner wrapping results ===
        dcc.Loading(
            id='loading-results',
            type='default',
            children=html.Div(id='results-container'),
            style={'minHeight': '60px'},
        ),

        # === Colorscale controls (hidden until results) ===
        html.Div(id='controls-container', children=[
            html.Div([
                html.Label("Heatmap color range:  ", style={'fontWeight': 'bold'}),
                html.Label("Min: "),
                dcc.Input(id='zmin-input', type='number', value=-0.8, step=0.1,
                          style={'width': '80px', 'marginRight': '15px'}),
                html.Label("Max: "),
                dcc.Input(id='zmax-input', type='number', value=0.5, step=0.1,
                          style={'width': '80px', 'marginRight': '25px'}),
                html.Label("WT score: "),
                dcc.Input(id='wt-input', type='number', value=0.0, step=0.01,
                          style={'width': '90px', 'marginRight': '5px'}),
                html.Span(id='wt-source-label', children='',
                          style={'fontSize': '12px', 'color': '#6c757d',
                                 'fontStyle': 'italic'}),
                html.Span(style={'display': 'inline-block', 'width': '30px'}),
                html.Label("Reconstruction:  ", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='recon-method',
                    options=[
                        {'label': 'Pretrained Dictionary', 'value': 'pretrained_recon'},
                        {'label': 'Self-trained Dictionary', 'value': 'dict_recon', 'disabled': True},
                        {'label': 'Naive (row mean)', 'value': 'naive_recon', 'disabled': True},
                    ],
                    value='pretrained_recon',
                    clearable=False,
                    style={'width': '240px', 'display': 'inline-block',
                           'verticalAlign': 'middle'},
                ),
            ], style={'marginBottom': '15px', 'marginTop': '10px',
                       'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap',
                       'gap': '4px'}),
        ], style={'display': 'none'}),

        # === Heatmaps container ===
        html.Div(id='heatmaps-container'),

        # === Download buttons (hidden until results) ===
        html.Div(id='downloads-container', children=[
            html.Button("Download Reconstructed VEM (.dict.tsv)", id='btn-download-dict',
                         style=DOWNLOAD_BUTTON_STYLE),
            html.Button("Download Input VEM (.VEM.tsv)", id='btn-download-vem',
                         style=DOWNLOAD_BUTTON_STYLE),
        ], style={'display': 'none', 'marginTop': '25px', 'marginBottom': '40px'}),

        # === Download components ===
        dcc.Download(id='download-dict'),
        dcc.Download(id='download-vem'),

    ], style=SECTION_STYLE),
])


# ---------------------------------------------------------------------------
# Callback: Handle file upload → show info, populate score columns, enable button
# ---------------------------------------------------------------------------
def _process_file(text, filename):
    """Shared logic: detect format, find score columns, build upload info.

    Returns (info, score_options, default_score, store_data) or raises on error.
    ``store_data`` contains {'filename': ..., 'content': base64-encoded text}.
    """
    suffix = os.path.splitext(filename)[1] if filename else '.csv'
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as tmp:
        tmp.write(text)
        tmp_path = tmp.name

    try:
        sep = sniff_separator(tmp_path)
        df = pd.read_csv(tmp_path, sep=sep)
        fmt = detect_format(df)

        score_options = []
        default_score = None

        if fmt not in ('headerless', 'vem_wide', 'vem_wide_1letter'):
            _NON_SCORE = {
                'Position', 'Amino_Acid', 'Amino_Acid_1', 'Amino_Acid_2',
                'accession', 'hgvs_pro', 'hgvs_nt', 'hgvs_splice',
                'var', 'aa_substitutions', 'WT', 'position', 'wt_aa',
            }
            numeric_cols = [
                c for c in df.columns
                if c not in _NON_SCORE and pd.api.types.is_numeric_dtype(df[c])
            ]

            for candidate in ('score', 'activity', 'Score', 'Activity'):
                if candidate in df.columns:
                    default_score = candidate
                    break
            if default_score is None and numeric_cols:
                default_score = numeric_cols[0]

            score_options = [{'label': c, 'value': c} for c in numeric_cols]

        n_rows = len(df)
        fmt_label = fmt
        if fmt == 'headerless':
            fmt_label = 'headerless (auto-detected)'
            n_rows += 1

        info = html.Div([
            html.Span(f"{filename}", style={'fontWeight': 'bold'}),
            html.Span(f"  —  format: {fmt_label}, {n_rows} rows",
                       style={'color': '#6c757d'}),
        ])

        content_b64 = base64.b64encode(text.encode('utf-8')).decode('ascii')
        store_data = {'filename': filename, 'content': content_b64}

        return info, score_options, default_score, store_data
    finally:
        os.unlink(tmp_path)


@app.callback(
    Output('upload-info', 'children'),
    Output('score-col-dropdown', 'options'),
    Output('score-col-dropdown', 'value'),
    Output('run-button', 'disabled'),
    Output('run-button', 'style'),
    Output('uploaded-file-store', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if contents is None:
        return '', [], None, True, BUTTON_DISABLED_STYLE, None

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    text = decoded.decode('utf-8')

    try:
        info, score_options, default_score, store_data = _process_file(text, filename)
        return info, score_options, default_score, False, BUTTON_STYLE, store_data
    except Exception as e:
        info = html.Div([
            html.Span(f"Error reading {filename}: ", style={'color': 'red', 'fontWeight': 'bold'}),
            html.Span(str(e)),
        ])
        return info, [], None, True, BUTTON_DISABLED_STYLE, None


@app.callback(
    Output('upload-info', 'children', allow_duplicate=True),
    Output('score-col-dropdown', 'options', allow_duplicate=True),
    Output('score-col-dropdown', 'value', allow_duplicate=True),
    Output('run-button', 'disabled', allow_duplicate=True),
    Output('run-button', 'style', allow_duplicate=True),
    Output('uploaded-file-store', 'data', allow_duplicate=True),
    Input('example-dropdown', 'value'),
    prevent_initial_call=True,
)
def load_example(dataset_id):
    if dataset_id is None:
        raise dash.exceptions.PreventUpdate

    # Look up dataset in manifest
    dataset = next((d for d in EXAMPLE_DATASETS if d['id'] == dataset_id), None)
    if dataset is None:
        raise dash.exceptions.PreventUpdate

    filepath = os.path.join(EXAMPLES_DIR, dataset['file'])
    with open(filepath, 'r') as f:
        text = f.read()

    try:
        info, score_options, default_score, store_data = _process_file(text, dataset['file'])
        # Use manifest-specified score column if available
        if dataset.get('score_col') and any(
            o['value'] == dataset['score_col'] for o in score_options
        ):
            default_score = dataset['score_col']
        return info, score_options, default_score, False, BUTTON_STYLE, store_data
    except Exception as e:
        info = html.Div([
            html.Span("Error loading example: ", style={'color': 'red', 'fontWeight': 'bold'}),
            html.Span(str(e)),
        ])
        return info, [], None, True, BUTTON_DISABLED_STYLE, None


# ---------------------------------------------------------------------------
# Callback: Run analysis → produce results
# ---------------------------------------------------------------------------
def _run_selftrained_bg(run_id, vem_df, nan_handling):
    """Background thread: run full self-trained pipeline, store results."""
    try:
        results = run_mavepolish(vem_df, nan_handling=nan_handling)
        serialised = {
            'dict_recon': results['dict_recon'].to_json(),
            'pca_recon': results['pca_recon'].to_json(),
            'naive_recon': results['naive_recon'].to_json(),
            'err_dict': results['err_dict'],
            'err_pca': results['err_pca'],
            'err_naive': results['err_naive'],
            'n_iterations': results['n_iterations'],
            'rescaling_factor': results['rescaling_factor'],
        }
        with _bg_lock:
            _bg_results[run_id] = serialised
    except Exception as e:
        with _bg_lock:
            _bg_results[run_id] = {'error': str(e)}


@app.callback(
    Output('results-store', 'data'),
    Output('results-container', 'children'),
    Output('controls-container', 'style'),
    Output('heatmaps-container', 'children', allow_duplicate=True),
    Output('downloads-container', 'style'),
    Output('zmin-input', 'value'),
    Output('zmax-input', 'value'),
    Output('wt-input', 'value'),
    Output('wt-input', 'disabled'),
    Output('bg-run-id', 'data'),
    Output('bg-poll', 'disabled'),
    Output('recon-method', 'options', allow_duplicate=True),
    Output('recon-method', 'value', allow_duplicate=True),
    Input('run-button', 'n_clicks'),
    State('uploaded-file-store', 'data'),
    State('nan-handling', 'value'),
    State('score-col-dropdown', 'value'),
    prevent_initial_call=True,
)
def run_analysis(n_clicks, file_store, nan_handling, score_col):
    hide = {'display': 'none'}
    initial_recon_options = [
        {'label': 'Pretrained Dictionary', 'value': 'pretrained_recon'},
        {'label': 'Self-trained Dictionary', 'value': 'dict_recon', 'disabled': True},
        {'label': 'Naive (row mean)', 'value': 'naive_recon', 'disabled': True},
    ]
    if n_clicks == 0 or file_store is None:
        return (None, '', hide, '', hide, -0.8, 0.5, 0.0, False, None, True,
                initial_recon_options, 'pretrained_recon')

    filename = file_store['filename']
    decoded = base64.b64decode(file_store['content'])
    text = decoded.decode('utf-8')

    # Write temp file for to_vem
    suffix = os.path.splitext(filename)[1] if filename else '.csv'
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as tmp:
        tmp.write(text)
        tmp_path = tmp.name

    try:
        # Step 1: Convert to VEM
        vem_df = to_vem(tmp_path, score_col_hint=score_col)

        # Step 2: Fast pretrained reconstruction
        results = run_pretrained(vem_df, PRETRAINED_MODEL_PATH, nan_handling=nan_handling)

        # Step 3: Launch self-trained in background
        import uuid
        run_id = str(uuid.uuid4())
        bg_thread = threading.Thread(
            target=_run_selftrained_bg,
            args=(run_id, vem_df.copy(), nan_handling),
            daemon=True,
        )
        bg_thread.start()

        # Serialise DataFrames to JSON for dcc.Store
        store = {
            'filename': filename,
            'original': results['original'].to_json(),
            'pretrained_recon': results['pretrained_recon'].to_json(),
            'nan_mask': results['nan_mask'].to_json(),
            'err_pretrained': results['err_pretrained'],
            # Self-trained placeholders (filled when background finishes)
            'dict_recon': None,
            'pca_recon': None,
            'naive_recon': None,
            'err_dict': None,
            'err_pca': None,
            'err_naive': None,
            'global_mean': results['global_mean'],
            'wt_aa': results['wt_aa'].to_json() if results['wt_aa'] is not None else None,
            'columns': results['columns'],
            'nan_handling': nan_handling,
        }

        # Auto-detect color range and center
        orig = results['original']
        nan_mask_df = results['nan_mask']
        vals = orig.values[~nan_mask_df.values & ~np.isnan(orig.values.astype(float))]
        if len(vals) > 0:
            auto_zmin = float(np.percentile(vals, 1))
            auto_zmax = float(np.percentile(vals, 99))
            center_val = kde_wt_peak(vals)
        else:
            auto_zmin, auto_zmax, center_val = -1.0, 1.0, 0.0
        store['auto_zmin'] = round(auto_zmin, 2)
        store['auto_zmax'] = round(auto_zmax, 2)
        store['center_val'] = round(center_val, 4)

        # Compute WT score from synonymous (diagonal) cells if available
        wt_aa_series = results['wt_aa']
        wt_score_data = None
        if wt_aa_series is not None:
            wt_scores = [
                float(orig.loc[pos, wt_aa_series[pos]])
                for pos in orig.index
                if wt_aa_series.get(pos, '') in orig.columns
                and not nan_mask_df.loc[pos, wt_aa_series.get(pos, '')]
            ]
            if wt_scores:
                wt_score_data = round(float(np.median(wt_scores)), 4)
        store['wt_score_data'] = wt_score_data

        # Build the full results UI
        wt_val = wt_score_data if wt_score_data is not None else center_val
        wt_disabled = wt_score_data is not None

        ui = build_results_ui(store)
        show = {'display': 'block'}
        show_dl = {'display': 'block', 'marginTop': '25px', 'marginBottom': '40px'}
        heatmaps = build_heatmaps_list(store, store['auto_zmin'], store['auto_zmax'],
                                        wt_val, 'pretrained_recon')
        return (store, ui, show, heatmaps, show_dl,
                store['auto_zmin'], store['auto_zmax'],
                round(wt_val, 4), wt_disabled,
                run_id, False,  # enable polling
                initial_recon_options, 'pretrained_recon')

    except Exception as e:
        hide = {'display': 'none'}
        error_msg = html.Div([
            html.H4("Analysis failed", style={'color': 'red'}),
            html.Pre(str(e), style={'whiteSpace': 'pre-wrap', 'color': '#dc3545'}),
        ])
        return (None, error_msg, hide, '', hide, -0.8, 0.5, 0.0, False, None, True,
                initial_recon_options, 'pretrained_recon')
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Callback: Poll for background self-trained results
# ---------------------------------------------------------------------------
@app.callback(
    Output('results-store', 'data', allow_duplicate=True),
    Output('bg-poll', 'disabled', allow_duplicate=True),
    Output('recon-method', 'options'),
    Input('bg-poll', 'n_intervals'),
    State('bg-run-id', 'data'),
    State('results-store', 'data'),
    prevent_initial_call=True,
)
def poll_background(n_intervals, run_id, store):
    """Poll for background self-trained results.

    IMPORTANT: This callback only outputs to hidden stores and dropdown options
    — NO visible components — so dcc.Loading spinner is never triggered.
    The error cards are updated by a separate clientside callback watching
    results-store.
    """
    if run_id is None or store is None:
        raise dash.exceptions.PreventUpdate

    with _bg_lock:
        bg = _bg_results.get(run_id)

    if bg is None:
        # Still running
        raise dash.exceptions.PreventUpdate

    # Background finished — merge results into store
    if 'error' in bg:
        all_options = [
            {'label': 'Pretrained Dictionary', 'value': 'pretrained_recon'},
            {'label': 'Self-trained Dictionary (failed)', 'value': 'dict_recon', 'disabled': True},
            {'label': 'Naive (failed)', 'value': 'naive_recon', 'disabled': True},
        ]
        with _bg_lock:
            _bg_results.pop(run_id, None)
        return store, True, all_options

    store['dict_recon'] = bg['dict_recon']
    store['pca_recon'] = bg['pca_recon']
    store['naive_recon'] = bg['naive_recon']
    store['err_dict'] = bg['err_dict']
    store['err_pca'] = bg['err_pca']
    store['err_naive'] = bg['err_naive']
    store['n_iterations'] = bg['n_iterations']
    store['rescaling_factor'] = bg['rescaling_factor']

    all_options = [
        {'label': 'Pretrained Dictionary', 'value': 'pretrained_recon'},
        {'label': 'Self-trained Dictionary', 'value': 'dict_recon'},
        {'label': 'Naive (row mean)', 'value': 'naive_recon'},
    ]

    with _bg_lock:
        _bg_results.pop(run_id, None)

    return store, True, all_options


# Clientside callback: update error cards from store WITHOUT triggering dcc.Loading
app.clientside_callback(
    """
    function(store) {
        if (!store) {
            return [
                window.dash_clientside.no_update, window.dash_clientside.no_update,
                window.dash_clientside.no_update, window.dash_clientside.no_update
            ];
        }
        var VALUE_STYLE_PENDING = {'fontSize': '20px', 'fontWeight': 'bold', 'color': '#adb5bd'};
        var errDict, errNaive, styleDict, styleNaive;
        if (store.err_dict !== null && store.err_dict !== undefined) {
            errDict = store.err_dict.toFixed(4);
            styleDict = {'fontSize': '20px', 'fontWeight': 'bold', 'color': '#6f42c1'};
        } else {
            errDict = '...';
            styleDict = VALUE_STYLE_PENDING;
        }
        if (store.err_naive !== null && store.err_naive !== undefined) {
            errNaive = store.err_naive.toFixed(4);
            styleNaive = {'fontSize': '20px', 'fontWeight': 'bold', 'color': '#dc3545'};
        } else {
            errNaive = '...';
            styleNaive = VALUE_STYLE_PENDING;
        }
        return [errDict, styleDict, errNaive, styleNaive];
    }
    """,
    Output('err-dict-val', 'children'),
    Output('err-dict-val', 'style'),
    Output('err-naive-val', 'children'),
    Output('err-naive-val', 'style'),
    Input('results-store', 'data'),
)


def build_results_ui(store):
    """Build the full results UI from stored analysis results."""
    filename = store['filename']
    err_dict = store['err_dict']
    err_naive = store['err_naive']

    # Reconstruct DataFrames
    original = pd.read_json(io.StringIO(store['original']))
    nan_mask = pd.read_json(io.StringIO(store['nan_mask']))

    wt_aa = None
    if store.get('wt_aa') is not None:
        wt_aa = pd.read_json(io.StringIO(store['wt_aa']), typ='series')

    n_pos = original.shape[0]
    n_aa = original.shape[1]

    # --- Data coverage metrics ---
    total_cells = nan_mask.size
    measured_cells = int((~nan_mask).sum().sum())
    completeness = measured_cells / total_cells * 100 if total_cells > 0 else 0

    # Synonymous (WT diagonal): positions where wt_aa matches a column and cell is measured
    if wt_aa is not None:
        n_syn = sum(
            1 for pos in original.index
            if wt_aa.get(pos, '') in original.columns
            and not nan_mask.loc[pos, wt_aa.get(pos, '')]
        )
        n_syn_possible = sum(
            1 for pos in original.index
            if wt_aa.get(pos, '') in original.columns
        )
    else:
        n_syn = 0
        n_syn_possible = n_pos

    # Stop codons (Ter column)
    if 'Ter' in original.columns:
        n_ter = int((~nan_mask['Ter']).sum())
    else:
        n_ter = 0

    # Missense: measured cells that are not synonymous diagonal and not Ter
    n_missense = measured_cells - n_syn - n_ter

    # Strip extension for display
    display_name = filename
    for ext in ('.csv', '.tsv', '.txt', '.tab'):
        if display_name.lower().endswith(ext):
            display_name = display_name[:-len(ext)]
            break

    # --- Helper to build a compact card ---
    LABEL_STYLE = {'fontSize': '11px', 'color': '#6c757d', 'marginTop': '2px'}
    VALUE_STYLE = {'fontSize': '20px', 'fontWeight': 'bold'}

    def card(value, label, color):
        return html.Div([
            html.Div(value, style={**VALUE_STYLE, 'color': color}),
            html.Div(label, style=LABEL_STYLE),
        ], style=CARD_STYLE)

    # Format fraction strings
    syn_str = f"{n_syn} / {n_syn_possible}" if wt_aa is not None else "N/A"
    ter_str = f"{n_ter} / {n_pos}"

    return html.Div([
        html.H3(f"Results: {display_name}", style={'marginBottom': '15px'}),

        # --- Cards (left) + Distribution (right) side by side ---
        html.Div([
            # Left column: summary cards stacked vertically
            html.Div([
                # Section: Data
                html.Div("Data", style={'fontSize': '11px', 'fontWeight': 'bold',
                                         'color': '#6c757d', 'textTransform': 'uppercase',
                                         'letterSpacing': '1px', 'marginBottom': '2px'}),
                card(f"{n_pos} \u00d7 {n_aa}", "matrix size", '#0d6efd'),
                card(f"{completeness:.0f}%", "overall completeness", '#198754'),
                card(f"{n_missense:,}", "missense scores", '#0d6efd'),
                card(syn_str, "synonymous (WT) scores", '#0d6efd'),
                card(ter_str, "stop codon scores", '#0d6efd'),

                # Section: Reconstruction
                html.Div("Reconstruction error", style={
                    'fontSize': '11px', 'fontWeight': 'bold', 'color': '#6c757d',
                    'textTransform': 'uppercase', 'letterSpacing': '1px',
                    'marginTop': '10px', 'marginBottom': '2px'}),
                card(f"{store['err_pretrained']:.4f}", "pretrained dictionary", '#6f42c1'),
                html.Div([
                    html.Div(
                        f"{err_dict:.4f}" if err_dict is not None else "...",
                        id='err-dict-val',
                        style={**VALUE_STYLE,
                               'color': '#6f42c1' if err_dict is not None else '#adb5bd'}),
                    html.Div("self-trained dictionary", style=LABEL_STYLE),
                ], style=CARD_STYLE),
                html.Div([
                    html.Div(
                        f"{err_naive:.4f}" if err_naive is not None else "...",
                        id='err-naive-val',
                        style={**VALUE_STYLE,
                               'color': '#dc3545' if err_naive is not None else '#adb5bd'}),
                    html.Div("naive (row mean)", style=LABEL_STYLE),
                ], style=CARD_STYLE),
            ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '6px',
                       'width': '210px', 'flexShrink': '0'}),

            # Right column: distribution plot
            html.Div([
                html.H4("Score Distribution", style={'marginTop': '0'}),
                dcc.Graph(id='distribution-plot', figure=build_distribution_figure(store),
                          config={'displayModeBar': False},
                          style={'height': '500px'}),
            ], style={'flex': '1', 'minWidth': '300px'}),

        ], style={'display': 'flex', 'gap': '20px', 'marginBottom': '25px',
                   'alignItems': 'stretch'}),
    ])


# ---------------------------------------------------------------------------
# Build distribution figure
# ---------------------------------------------------------------------------
def build_distribution_figure(store):
    original = pd.read_json(io.StringIO(store['original']))
    nan_mask = pd.read_json(io.StringIO(store['nan_mask']))

    # Use best available reconstruction for overlay
    recon_key = 'dict_recon' if store.get('dict_recon') else 'pretrained_recon'
    recon = pd.read_json(io.StringIO(store[recon_key]))
    recon_label = 'Self-trained recon.' if recon_key == 'dict_recon' else 'Pretrained recon.'

    # Mask NaN positions so we only plot real measured values
    real_mask = ~nan_mask
    orig_vals = original.values[real_mask.values]
    recon_vals = recon.values[real_mask.values]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=orig_vals, nbinsx=100, name='Original',
        marker_color='steelblue', opacity=0.7,
    ))
    fig.add_trace(go.Histogram(
        x=recon_vals, nbinsx=100, name=recon_label,
        marker_color='darkorange', opacity=0.5,
    ))

    # Stop codon distribution if Ter column exists
    if 'Ter' in original.columns:
        ter_vals = original['Ter'].dropna().values
        if len(ter_vals) > 0:
            fig.add_trace(go.Histogram(
                x=ter_vals, nbinsx=50, name='Stop codons',
                marker_color='red', opacity=0.5,
            ))

    # WT score vertical line
    wt_score_data = store.get('wt_score_data')
    center_val = store.get('center_val', 0)
    if wt_score_data is not None:
        wt_val = wt_score_data
        wt_label = f'WT = {wt_val:.4f} (data)'
    else:
        wt_val = center_val
        wt_label = f'WT = {wt_val:.4f} (KDE)'

    fig.add_vline(
        x=wt_val, line_dash='dash', line_color='black', line_width=1.5,
        annotation_text=wt_label,
        annotation_position='top right',
        annotation_font_size=11,
    )

    fig.update_layout(
        barmode='overlay',
        xaxis_title='Score',
        yaxis_title='Count',
        template='plotly_white',
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=50, r=20, t=30, b=40),
        height=350,
    )
    return fig


# ---------------------------------------------------------------------------
# Build combined 4-row heatmap figure (shared x-axis, locked y-axis)
# ---------------------------------------------------------------------------
def build_heatmaps_list(store, zmin, zmax, wt_score=None, recon_key='pretrained_recon'):
    """Build a Plotly figure with 2 stacked heatmaps: original + selected reconstruction."""
    err_map = {
        'pretrained_recon': ('Pretrained Dictionary', store.get('err_pretrained')),
        'dict_recon': ('Self-trained Dictionary', store.get('err_dict')),
        'pca_recon': ('PCA', store.get('err_pca')),
        'naive_recon': ('Naive Mean', store.get('err_naive')),
    }

    # Fall back to pretrained if selected reconstruction isn't available yet
    if store.get(recon_key) is None:
        recon_key = 'pretrained_recon'
    recon_name, recon_err = err_map.get(recon_key, err_map['pretrained_recon'])

    nan_mask = pd.read_json(io.StringIO(store['nan_mask']))

    wt_aa = None
    if store.get('wt_aa') is not None:
        wt_aa = pd.read_json(io.StringIO(store['wt_aa']), typ='series')

    configs = [
        ('original', 'Original Data', True),          # apply NaN mask
        (recon_key, f'{recon_name} Reconstruction (error = {recon_err:.4f})' if recon_err is not None else f'{recon_name} Reconstruction', False),
    ]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=[title for _, title, _ in configs],
    )

    # Use wt_score for centering if provided, else fall back to KDE center_val
    center = wt_score if wt_score is not None else store.get('center_val', 0)
    eff_zmin, eff_zmax = symmetric_color_range(zmin, zmax, center)

    # Compute per-row completeness for reconstruction masking:
    # blank out rows (positions) with < 30% original data
    row_completeness = (~nan_mask).sum(axis=1) / nan_mask.shape[1]
    sparse_rows = row_completeness < 0.30

    for row_idx, (key, title, mask_nan) in enumerate(configs, start=1):
        data = pd.read_json(io.StringIO(store[key]))
        display_data = data.copy()

        if mask_nan:
            # Original: show only measured values
            display_data[nan_mask] = np.nan
        else:
            # Reconstruction: show all data, but blank sparse rows (<30% original)
            for pos in display_data.index:
                if sparse_rows.get(pos, False):
                    display_data.loc[pos, :] = np.nan

        z = display_data.T.values
        x_labels = [str(p) for p in display_data.index]
        y_labels = list(display_data.columns)

        # Build hover text
        hover_text = []
        for j, aa in enumerate(y_labels):
            row_hover = []
            for i, pos in enumerate(display_data.index):
                val = display_data.iloc[i, j]
                wt = wt_aa.get(pos, '?') if wt_aa is not None else '?'
                if pd.isna(val):
                    row_hover.append(f"Position: {pos}<br>Mutation: {wt}{pos}{aa}<br>Score: N/A")
                else:
                    row_hover.append(f"Position: {pos}<br>Mutation: {wt}{pos}{aa}<br>Score: {val:.4f}")
            hover_text.append(row_hover)

        # Only show colorbar on the first heatmap
        show_colorbar = (row_idx == 1)

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=x_labels,
                y=y_labels,
                colorscale=SEISMIC,
                zmin=eff_zmin,
                zmax=eff_zmax,
                hoverinfo='text',
                text=hover_text,
                colorbar=dict(title='Score', thickness=15) if show_colorbar else None,
                showscale=show_colorbar,
                xgap=0.5,
                ygap=0.5,
            ),
            row=row_idx, col=1,
        )

        # Overlay black dots at WT (diagonal) positions
        if wt_aa is not None:
            wt_x = []
            wt_y = []
            for pos in display_data.index:
                ref_aa = wt_aa.get(pos, '')
                if ref_aa in y_labels:
                    wt_x.append(str(pos))
                    wt_y.append(ref_aa)
            if wt_x:
                fig.add_trace(
                    go.Scatter(
                        x=wt_x, y=wt_y,
                        mode='markers',
                        marker=dict(symbol='circle', size=3, color='black'),
                        showlegend=False,
                        hoverinfo='skip',
                    ),
                    row=row_idx, col=1,
                )

    # Configure axes
    # NOTE: x_labels are strings (categorical axis), so Plotly uses 0-based
    # category indices for range/minallowed/maxallowed — NOT position values.
    n_rows = 2
    n_positions = len(x_labels)
    x_range = [-0.5, n_positions - 0.5]

    for i in range(1, n_rows + 1):
        yaxis_key = f'yaxis{i}' if i > 1 else 'yaxis'
        xaxis_key = f'xaxis{i}' if i > 1 else 'xaxis'
        fig.update_layout(**{
            yaxis_key: dict(
                tickfont=dict(size=10),
                dtick=1,            # show every amino acid label
                fixedrange=True,    # lock Y axis — zoom only affects X
                categoryorder='array',
                categoryarray=y_labels,
                range=[-0.5, len(y_labels) - 0.5],  # clip to exact rows
            ),
            xaxis_key: dict(
                tickangle=90,
                dtick=2,
                tickfont=dict(size=8),
                showticklabels=True,
                range=x_range,
                constrain='domain',
                minallowed=-0.5,
                maxallowed=n_positions - 0.5,
            ),
        })

    # Only label "Position" on the bottom x-axis
    fig.update_layout(**{
        f'xaxis{n_rows}': dict(title='Position'),
    })

    fig.update_layout(
        template='plotly_white',
        height=300 * n_rows,
        margin=dict(l=60, r=30, t=50, b=60),
        plot_bgcolor='lightgray',
        dragmode='pan',             # default to pan (not zoom)
        # uirevision keyed on filename: resets zoom on new file,
        # preserves zoom when switching recon method / color range
        uirevision=store.get('filename', 'heatmap'),
    )

    # Style subplot titles (annotations) to avoid overlap
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=13)
        ann['y'] = ann['y'] + 0.02  # nudge titles up slightly

    return [dcc.Graph(
        figure=fig,
        config={
            'scrollZoom': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': [
                'zoom2d', 'autoScale2d',
                'select2d', 'lasso2d', 'toggleSpikelines',
                'hoverCompareCartesian', 'hoverClosestCartesian',
            ],
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': store.get('filename', 'mavepolish_heatmap').rsplit('.', 1)[0],
                'height': 300 * n_rows,
                'width': max(1200, len(x_labels) * 4),
                'scale': 2,
            },
        },
    )]


# ---------------------------------------------------------------------------
# Callback: Update heatmaps when zmin/zmax change
# ---------------------------------------------------------------------------
@app.callback(
    Output('heatmaps-container', 'children'),
    Input('zmin-input', 'value'),
    Input('zmax-input', 'value'),
    Input('wt-input', 'value'),
    Input('recon-method', 'value'),
    State('results-store', 'data'),
    prevent_initial_call=True,
)
def update_heatmaps(zmin, zmax, wt_score, recon_key, store):
    if store is None:
        return ''

    zmin = zmin if zmin is not None else -0.8
    zmax = zmax if zmax is not None else 0.5
    wt_score = wt_score if wt_score is not None else store.get('center_val', 0)
    recon_key = recon_key or 'pretrained_recon'

    return build_heatmaps_list(store, zmin, zmax, wt_score, recon_key)


# ---------------------------------------------------------------------------
# Callback: Download dict.tsv
# ---------------------------------------------------------------------------
@app.callback(
    Output('download-dict', 'data'),
    Input('btn-download-dict', 'n_clicks'),
    State('results-store', 'data'),
    prevent_initial_call=True,
)
def download_dict(n_clicks, store):
    if store is None:
        return None

    # Download best available reconstruction
    if store.get('dict_recon'):
        recon = pd.read_json(io.StringIO(store['dict_recon']))
        suffix = 'dict'
    else:
        recon = pd.read_json(io.StringIO(store['pretrained_recon']))
        suffix = 'pretrained'
    tsv = recon.to_csv(sep='\t', index_label='Position')

    # Derive filename
    fname = store['filename']
    for ext in ('.csv', '.tsv', '.txt', '.tab'):
        if fname.lower().endswith(ext):
            fname = fname[:-len(ext)]
            break
    return dcc.send_string(tsv, filename=f"{fname}.{suffix}.tsv")


# ---------------------------------------------------------------------------
# Callback: Download VEM.tsv
# ---------------------------------------------------------------------------
@app.callback(
    Output('download-vem', 'data'),
    Input('btn-download-vem', 'n_clicks'),
    State('results-store', 'data'),
    prevent_initial_call=True,
)
def download_vem(n_clicks, store):
    if store is None:
        return None

    original = pd.read_json(io.StringIO(store['original']))

    # Re-attach wt_aa column
    if store.get('wt_aa') is not None:
        wt_aa = pd.read_json(io.StringIO(store['wt_aa']), typ='series')
        original.insert(0, 'wt_aa', wt_aa)

    tsv = original.to_csv(sep='\t', index_label='Position')

    fname = store['filename']
    for ext in ('.csv', '.tsv', '.txt', '.tab'):
        if fname.lower().endswith(ext):
            fname = fname[:-len(ext)]
            break
    return dcc.send_string(tsv, filename=f"{fname}.VEM.tsv")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.getenv('PORT', 8051))
    app.run(debug=False, port=port, host='0.0.0.0')
