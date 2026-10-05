#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PfPATH Walk Evaluator
Loads walks.csv from a PfPATH run, identifies compensatory ridge residues,
validates against experimental oracle data, and runs ESM2 structural analysis.

Core outputs (in --out directory):
  ridge_heatmap.png            – position × KP frequency heatmap
  ridge_ranking.png            – ranked bar chart with per-KP breakdown
  ridge_freq_fitness.png       – frequency vs fitness scatter
  ridge_per_kp.png             – one bar panel per resistance KP
  ridge_dynamics.png           – fitness trajectories + early/late compensators
  ridge_panel.png              – combined publication figure
  ridge_summary_full.csv       – all positions with metrics
  ridge_candidates.csv         – high-confidence ridge candidates only

Oracle validation (requires --oracle):
  oracle_step1_corr.png        – simulator vs oracle single-mutant correlation
  oracle_feff_landscape.png    – oracle haplotype fitness landscape
  oracle_kp99_corecruitment.png– KP99 walk co-recruitment dynamics
  oracle_lethality_avoidance.png – A7V+S99N lethality avoidance check
  oracle_validation_panel.png  – combined oracle validation figure

ESM2 structural analysis (requires --wt and --esm_model, or --esm_cache):
  esm_contact_deltas.png       – ΔContact maps (mutant − WT)
  esm_embedding_perturbation.png – per-position perturbation profiles
  esm_perturbation_profile.png – mean perturbation across all singles
  esm_ridge_coupling.png       – ridge × resistance contact + perturbation heatmaps
  esm_ridge_contact_partners.png – top contact partners per ridge candidate
  esm_panel.png                – combined ESM2 figure
  esm_results.npz              – cached inference results (reused on re-runs)
"""

import argparse
import os
import re
import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings('ignore')

CANONICAL_OFFSET = 9   # canonical = shifted + 9

# Built-in WT PfDHFR sequence (shifted positions match simulator 1-indexed numbering).
# Overridden at runtime by --wt fasta if provided.
_WT_SEQ_BUILTIN = (
    "DIYAICACCKVESKNEGKKNEVFNNYTFRGLGNKGVLPWKCNSLDMKYFCAVTTYVNESK"
    "YEKLKYKRCKYLNKETVDNVNDMPNSKKLQNVVVMGRTSWESIPKKFKPLSNRINVILSR"
    "TLKKEDFDEDVYIINKVEDLIVLLGKLNYYKCFIIGGSVVYQEFLEKKLIKKIYFTRINS"
    "TYECDVFFPEINENEYQIISVSDVYTSNNTTLDFIIYKK"
)


# All haplotypes to analyse with ESM2 (name → mutation list in shifted numbering).
_ESM_HAPLOTYPES: List[Tuple[str, List[Tuple[int, str, str]]]] = [
    ('WT',                   []),
    # ── single resistance mutants ──────────────────────────────────────────
    ('A7V',                  [(7,   'A', 'V')]),
    ('C41R',                 [(41,  'C', 'R')]),
    ('C41Y',                 [(41,  'C', 'Y')]),
    ('C41N',                 [(41,  'C', 'N')]),
    ('N42I',                 [(42,  'N', 'I')]),
    ('C50R',                 [(50,  'C', 'R')]),
    ('S99N',                 [(99,  'S', 'N')]),
    ('S99T',                 [(99,  'S', 'T')]),
    ('I155L',                [(155, 'I', 'L')]),
    # ── clinically observed multi-mutant haplotypes ────────────────────────
    ('N42I+S99N',            [(42,  'N', 'I'), (99,  'S', 'N')]),
    ('C50R+S99N',            [(50,  'C', 'R'), (99,  'S', 'N')]),
    ('N42I+C50R+S99N',       [(42,  'N', 'I'), (50,  'C', 'R'), (99,  'S', 'N')]),
    ('N42I+C50R+S99N+I155L', [(42,  'N', 'I'), (50,  'C', 'R'),
                               (99,  'S', 'N'), (155, 'I', 'L')]),
]

# Primary singles shown in per-panel embedding figures (oracle-validated + key variants).
# All singles contribute to the mean perturbation profile.
_ESM_PANEL_SINGLES = ['A7V', 'C41R', 'N42I', 'C50R', 'S99N', 'S99T', 'I155L']


# ─── helpers ─────────────────────────────────────────────────────────────────

def to_canonical(pos: int) -> int:
    return pos + CANONICAL_OFFSET


def parse_mutations(mut_str: str):
    """Return list of (pos_shifted, wt_aa, mut_aa) from e.g. 'N42I+C50R'."""
    if not mut_str or str(mut_str).strip().upper() in ('WT', 'NAN', ''):
        return []
    out = []
    for tok in str(mut_str).strip().split('+'):
        m = re.match(r'([A-Z])(\d+)([A-Z])', tok.strip())
        if m:
            out.append((int(m.group(2)), m.group(1), m.group(3)))
    return out


# ─── loading ─────────────────────────────────────────────────────────────────

def load_walks(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['fitness']      = pd.to_numeric(df['fitness'],      errors='coerce')
    df['step']         = pd.to_numeric(df['step'],         errors='coerce')
    df['rep']          = pd.to_numeric(df['rep'],          errors='coerce')
    df['key_position'] = pd.to_numeric(df['key_position'], errors='coerce')
    df = df.dropna(subset=['fitness', 'step', 'rep', 'key_position'])
    return df


# ─── core analysis ───────────────────────────────────────────────────────────

def compute_ridge_stats(df: pd.DataFrame,
                        resistance_kps: set,
                        min_freq: float = 0.05):
    """
    For every (resistance KP, secondary position) pair compute:
      freq            – fraction of walks at that KP with position present at final step
      mean_fitness    – mean final fitness of walks that recruited this position
      mean_first_step – mean step at which position first appeared

    Returns
    -------
    detail_df : long-format DataFrame, one row per (kp, position) pair above min_freq
    freq_pivot: wide DataFrame indexed by position, one column per KP + summary cols
    """
    records = []

    for kp in sorted(resistance_kps):
        sub = df[(df['key_position'] == kp) & (df['rep'] != 0)]
        n_walks = sub['rep'].nunique()
        if n_walks == 0:
            continue

        walk_data = []
        for rep, grp in sub.groupby('rep'):
            grp = grp.sort_values('step')
            final_row    = grp.iloc[-1]
            final_fit    = float(final_row['fitness'])
            final_muts   = parse_mutations(final_row['mutations'])
            final_mut_map = {pos: (wt_aa, mut_aa)
                             for pos, wt_aa, mut_aa in final_muts if pos != kp}

            # first step each secondary position appeared
            pos_first = {}
            for _, row in grp.iterrows():
                for pos, _, _ in parse_mutations(row['mutations']):
                    if pos != kp and pos not in pos_first:
                        pos_first[pos] = int(row['step'])

            walk_data.append({
                'rep':             rep,
                'final_fitness':   final_fit,
                'final_positions': set(final_mut_map.keys()),
                'final_mut_map':   final_mut_map,
                'pos_first_step':  pos_first,
            })

        # aggregate per secondary position
        pos_agg = defaultdict(lambda: {'count': 0, 'fitnesses': [],
                                        'first_steps': [], 'substitutions': []})
        for wd in walk_data:
            for pos in wd['final_positions']:
                pos_agg[pos]['count'] += 1
                pos_agg[pos]['fitnesses'].append(wd['final_fitness'])
                if pos in wd['pos_first_step']:
                    pos_agg[pos]['first_steps'].append(wd['pos_first_step'][pos])
                if pos in wd['final_mut_map']:
                    pos_agg[pos]['substitutions'].append(wd['final_mut_map'][pos])

        for pos, st in pos_agg.items():
            freq = st['count'] / n_walks
            if freq < min_freq:
                continue
            if st['substitutions']:
                top_sub = Counter(st['substitutions']).most_common(1)[0][0]
                mutation_label = f'{top_sub[0]}{pos}{top_sub[1]}'
            else:
                mutation_label = str(pos)
            records.append({
                'kp':              kp,
                'pos_shifted':     pos,
                'pos_canonical':   to_canonical(pos),
                'freq':            freq,
                'count':           st['count'],
                'n_walks':         n_walks,
                'mean_fitness':    float(np.mean(st['fitnesses'])) if st['fitnesses'] else np.nan,
                'mean_first_step': float(np.mean(st['first_steps'])) if st['first_steps'] else np.nan,
                'mutation_label':  mutation_label,
            })

    detail_df = pd.DataFrame(records)
    if detail_df.empty:
        return detail_df, pd.DataFrame()

    # pivot: rows = positions, cols = KPs
    kp_cols = sorted(resistance_kps)
    freq_pivot = detail_df.pivot_table(
        index='pos_shifted', columns='kp', values='freq', fill_value=0.0
    ).reindex(columns=kp_cols, fill_value=0.0)

    present_kps = [k for k in kp_cols if k in freq_pivot.columns]
    threshold = 0.25
    freq_pivot['cross_kp_count'] = (freq_pivot[present_kps] >= threshold).sum(axis=1)
    freq_pivot['mean_freq']      = freq_pivot[present_kps].mean(axis=1)
    freq_pivot['max_freq']       = freq_pivot[present_kps].max(axis=1)
    freq_pivot = freq_pivot.sort_values(['cross_kp_count', 'mean_freq'], ascending=False)

    # Attach most-common substitution label from detail_df
    if 'mutation_label' in detail_df.columns and not detail_df.empty:
        lbl_mode = (detail_df.groupby('pos_shifted')['mutation_label']
                    .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else '')
                    .rename('mutation_label'))
        freq_pivot = freq_pivot.join(lbl_mode)

    return detail_df, freq_pivot


# ─── individual figures ───────────────────────────────────────────────────────

def _kp_cols(freq_pivot, resistance_kps):
    return sorted([k for k in resistance_kps if k in freq_pivot.columns])


def _pos_label(pos: int, freq_pivot: pd.DataFrame) -> str:
    """Return 'S99N (c108)' style label, falling back to '99 (c108)' if no label."""
    can = to_canonical(int(pos))
    if (freq_pivot is not None and 'mutation_label' in freq_pivot.columns
            and int(pos) in freq_pivot.index):
        try:
            mut = freq_pivot.loc[int(pos), 'mutation_label']
            if mut and str(mut) not in ('nan', ''):
                return f'{mut}  (c{can})'
        except (KeyError, TypeError):
            pass
    return f'{int(pos)}  (c{can})'


def fig_heatmap(freq_pivot, resistance_kps, out_dir):
    kp_cols = _kp_cols(freq_pivot, resistance_kps)
    top = freq_pivot.sort_values(['cross_kp_count', 'mean_freq'], ascending=False).head(30)
    mat = top[kp_cols].values
    ylabs = [_pos_label(p, freq_pivot) for p in top.index]

    fig, ax = plt.subplots(figsize=(len(kp_cols) * 1.9 + 2, len(top) * 0.44 + 2))
    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(kp_cols)))
    ax.set_xticklabels([f'KP {k}' for k in kp_cols], fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(ylabs)))
    ax.set_yticklabels(ylabs, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if v > 0.55 else 'black')
    plt.colorbar(im, ax=ax, label='Frequency (fraction of walks)')
    ax.set_title('Secondary Position Frequency per Resistance Walk',
                 fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ridge_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def fig_ranking(freq_pivot, resistance_kps, out_dir):
    kp_cols = _kp_cols(freq_pivot, resistance_kps)
    top = freq_pivot.sort_values(['cross_kp_count', 'mean_freq'], ascending=False).head(25)
    labels = [_pos_label(p, freq_pivot) for p in top.index]

    color_map = {1: '#fee08b', 2: '#fc8d59', 3: '#d73027', 4: '#a50026'}
    bar_cols = [color_map.get(int(v), '#ffffbf') for v in top['cross_kp_count']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, max(7, len(top) * 0.42)))

    # Left — mean frequency
    ax1.barh(range(len(top)), top['mean_freq'], color=bar_cols,
             edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(top)))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Mean frequency across resistance KPs', fontsize=11)
    ax1.set_xlim(0, 1.05)
    ax1.axvline(0.25, color='grey', linestyle='--', linewidth=0.8)
    ax1.set_title('Ranked by Cross-KP Frequency', fontsize=12)
    patches = [mpatches.Patch(color=c, label=f'{n} KPs')
               for n, c in color_map.items() if n <= len(kp_cols)]
    ax1.legend(handles=patches, title='Cross-KP count', fontsize=8)

    # Right — stacked per-KP
    kp_colors = plt.cm.Set2(np.linspace(0, 1, len(kp_cols)))
    bottoms = np.zeros(len(top))
    for ci, kp in enumerate(kp_cols):
        vals = top[kp].values if kp in top.columns else np.zeros(len(top))
        ax2.barh(range(len(top)), vals / len(kp_cols),
                 left=bottoms, color=kp_colors[ci],
                 label=f'KP {kp}', edgecolor='white', linewidth=0.3)
        bottoms += vals / len(kp_cols)
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('Stacked per-KP frequency (scaled)', fontsize=11)
    ax2.set_title('Per-KP Breakdown', fontsize=12)
    ax2.legend(fontsize=9, loc='lower right')

    plt.suptitle('Compensatory Ridge Candidate Ranking', fontsize=14,
                 fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ridge_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()


def fig_freq_fitness(detail_df, resistance_kps, out_dir):
    _ml = lambda x: x.mode().iloc[0] if len(x) > 0 else str(x.name)
    agg = detail_df.groupby('pos_shifted').agg(
        mean_freq      = ('freq',           'mean'),
        mean_fitness   = ('mean_fitness',   'mean'),
        cross_kp_count = ('kp',             'nunique'),
        mutation_label = ('mutation_label', _ml),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        agg['mean_freq'],
        np.log10(np.clip(agg['mean_fitness'], 1e-12, None)),
        s=agg['cross_kp_count'] * 80,
        c=agg['cross_kp_count'], cmap='RdYlGn',
        vmin=1, vmax=len(resistance_kps),
        alpha=0.82, edgecolors='black', linewidths=0.5,
    )
    mask = (agg['mean_freq'] > 0.3) | (agg['cross_kp_count'] >= 3)
    for _, row in agg[mask].iterrows():
        lbl = row.get('mutation_label', str(int(row['pos_shifted'])))
        ax.annotate(lbl,
                    (row['mean_freq'],
                     np.log10(max(row['mean_fitness'], 1e-12))),
                    textcoords='offset points', xytext=(5, 2), fontsize=7)
    plt.colorbar(sc, ax=ax, label='No. of resistance KPs')
    ax.set_xlabel('Mean frequency across KPs', fontsize=12)
    ax.set_ylabel('Mean final fitness (log₁₀)', fontsize=12)
    ax.set_title('Position Frequency vs Fitness Contribution', fontsize=13)
    ax.axvline(0.25, color='grey', linestyle='--', linewidth=0.8, label='25% threshold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ridge_freq_fitness.png'), dpi=300, bbox_inches='tight')
    plt.close()


def fig_per_kp(detail_df, resistance_kps, out_dir, top_n=15):
    kps = sorted(resistance_kps)
    ncols = 2
    nrows = (len(kps) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 5))
    axes = np.array(axes).flatten()

    for ax, kp in zip(axes, kps):
        sub = (detail_df[detail_df['kp'] == kp]
               .sort_values('freq', ascending=False)
               .head(top_n))
        if sub.empty:
            ax.set_visible(False)
            continue
        labels = [f'{r.get("mutation_label", str(int(r["pos_shifted"])))}\n(c{int(r["pos_canonical"])})'
                  for _, r in sub.iterrows()]
        bars = ax.bar(range(len(sub)), sub['freq'],
                      color='steelblue', edgecolor='black', linewidth=0.5)
        for i, bar in enumerate(bars):
            if sub['freq'].iloc[i] >= 0.5:
                bar.set_color('#a50026')
            elif sub['freq'].iloc[i] >= 0.25:
                bar.set_color('#d73027')
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.25, color='grey', linestyle='--', linewidth=0.8,
                   label='25% threshold')
        ax.set_title(f'KP {kp}  (canonical {to_canonical(kp)})',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)

    for ax in axes[len(kps):]:
        ax.set_visible(False)

    plt.suptitle('Secondary Position Frequency per Resistance Mutation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ridge_per_kp.png'), dpi=300, bbox_inches='tight')
    plt.close()


def fig_dynamics(df, detail_df, resistance_kps, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left — mean ± sd fitness trajectory per KP
    traj_cols = plt.cm.Set1(np.linspace(0, 1, len(resistance_kps)))
    for ci, kp in enumerate(sorted(resistance_kps)):
        sub = df[(df['key_position'] == kp) & (df['rep'] != 0)]
        if sub.empty:
            continue
        agg = sub.groupby('step')['fitness'].agg(['mean', 'std'])
        mean_log = np.log10(np.clip(agg['mean'], 1e-12, None))
        std_log  = agg['std'] / (agg['mean'] * np.log(10) + 1e-12)
        ax1.plot(agg.index, mean_log, color=traj_cols[ci],
                 label=f'KP {kp}', linewidth=2)
        ax1.fill_between(agg.index,
                         mean_log - std_log, mean_log + std_log,
                         color=traj_cols[ci], alpha=0.15)
    ax1.set_xlabel('Mutation step', fontsize=12)
    ax1.set_ylabel('Mean fitness (log₁₀) ± 1 SD', fontsize=12)
    ax1.set_title('Walk Fitness Trajectories', fontsize=13)
    ax1.legend(fontsize=10)

    # Right — first-step bubble: early vs late compensators
    _ml2 = lambda x: x.mode().iloc[0] if len(x) > 0 else str(x.name)
    tp = (detail_df.groupby('pos_shifted')
          .agg(mean_first_step=('mean_first_step', 'mean'),
               mean_freq      =('freq',            'mean'),
               cross_kp_count =('kp',              'nunique'),
               mutation_label =('mutation_label',  _ml2))
          .reset_index()
          .nlargest(25, 'mean_freq'))

    sc = ax2.scatter(
        tp['mean_first_step'], tp['mean_freq'],
        s=tp['cross_kp_count'] * 100,
        c=tp['cross_kp_count'], cmap='RdYlGn', vmin=1, vmax=4,
        alpha=0.85, edgecolors='black', linewidths=0.6,
    )
    for _, row in tp.iterrows():
        lbl = row.get('mutation_label', str(int(row['pos_shifted'])))
        ax2.annotate(lbl,
                     (row['mean_first_step'], row['mean_freq']),
                     textcoords='offset points', xytext=(5, 2), fontsize=7)
    plt.colorbar(sc, ax=ax2, label='No. of resistance KPs')
    ax2.set_xlabel('Mean step of first appearance', fontsize=12)
    ax2.set_ylabel('Mean frequency at final step', fontsize=12)
    ax2.set_title('Early vs Late Compensators', fontsize=13)

    plt.suptitle('Walk Dynamics & Compensator Timing',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'ridge_dynamics.png'), dpi=300, bbox_inches='tight')
    plt.close()


def fig_master_panel(freq_pivot, detail_df, df, resistance_kps, out_dir):
    """Single publication-quality multi-panel figure."""
    kp_cols = _kp_cols(freq_pivot, resistance_kps)

    fig = plt.figure(figsize=(20, 15))
    gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.50)
    ax_heat = fig.add_subplot(gs[0, :2])
    ax_rank = fig.add_subplot(gs[0, 2])
    ax_traj = fig.add_subplot(gs[1, 0])
    ax_scat = fig.add_subplot(gs[1, 1])
    ax_step = fig.add_subplot(gs[1, 2])

    # ── A: Heatmap ──────────────────────────────────────────────────────────
    top = freq_pivot.sort_values(['cross_kp_count', 'mean_freq'],
                                 ascending=False).head(25)
    mat  = top[kp_cols].values
    ylabs = [_pos_label(p, freq_pivot) for p in top.index]
    im = ax_heat.imshow(mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax_heat.set_xticks(range(len(kp_cols)))
    ax_heat.set_xticklabels([f'KP {k}' for k in kp_cols],
                             fontsize=11, fontweight='bold')
    ax_heat.set_yticks(range(len(ylabs)))
    ax_heat.set_yticklabels(ylabs, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax_heat.text(j, i, f'{v:.2f}', ha='center', va='center',
                         fontsize=7.5, color='white' if v > 0.55 else 'black')
    plt.colorbar(im, ax=ax_heat, shrink=0.85, label='Frequency')
    ax_heat.set_title('(A)  Secondary Position Frequency Heatmap',
                      fontsize=12, fontweight='bold')

    # ── B: Ranking ──────────────────────────────────────────────────────────
    top15 = freq_pivot.sort_values(['cross_kp_count', 'mean_freq'],
                                   ascending=False).head(15)
    cmap  = {1: '#fee08b', 2: '#fc8d59', 3: '#d73027', 4: '#a50026'}
    bcols = [cmap.get(int(v), '#ffffbf') for v in top15['cross_kp_count']]
    ax_rank.barh(range(len(top15)), top15['mean_freq'],
                 color=bcols, edgecolor='black', linewidth=0.5)
    ax_rank.set_yticks(range(len(top15)))
    ax_rank.set_yticklabels(
        [_pos_label(p, freq_pivot) for p in top15.index], fontsize=8)
    ax_rank.invert_yaxis()
    ax_rank.axvline(0.25, color='grey', linestyle='--', linewidth=0.8)
    ax_rank.set_xlabel('Mean freq.', fontsize=10)
    ax_rank.set_title('(B)  Position Ranking', fontsize=12, fontweight='bold')
    patches = [mpatches.Patch(color=c, label=f'{n} KPs')
               for n, c in cmap.items() if n <= len(kp_cols)]
    ax_rank.legend(handles=patches, title='Cross-KP', fontsize=7)

    # ── C: Trajectories ─────────────────────────────────────────────────────
    tcols = plt.cm.Set1(np.linspace(0, 1, len(resistance_kps)))
    for ci, kp in enumerate(sorted(resistance_kps)):
        sub = df[(df['key_position'] == kp) & (df['rep'] != 0)]
        if sub.empty:
            continue
        agg     = sub.groupby('step')['fitness'].agg(['mean', 'std'])
        mlog    = np.log10(np.clip(agg['mean'], 1e-12, None))
        slog    = agg['std'] / (agg['mean'] * np.log(10) + 1e-12)
        ax_traj.plot(agg.index, mlog, color=tcols[ci],
                     label=f'KP {kp}', linewidth=2)
        ax_traj.fill_between(agg.index, mlog - slog, mlog + slog,
                              color=tcols[ci], alpha=0.15)
    ax_traj.set_xlabel('Mutation step', fontsize=10)
    ax_traj.set_ylabel('Mean fitness (log₁₀)', fontsize=10)
    ax_traj.set_title('(C)  Fitness Trajectories', fontsize=12, fontweight='bold')
    ax_traj.legend(fontsize=9)

    # ── D: Frequency vs Fitness ──────────────────────────────────────────────
    _ml3 = lambda x: x.mode().iloc[0] if len(x) > 0 else str(x.name)
    agg_pos = (detail_df.groupby('pos_shifted')
               .agg(mean_freq      = ('freq',           'mean'),
                    mean_fitness   = ('mean_fitness',   'mean'),
                    cross_kp_count = ('kp',             'nunique'),
                    mutation_label = ('mutation_label', _ml3))
               .reset_index())
    sc = ax_scat.scatter(
        agg_pos['mean_freq'],
        np.log10(np.clip(agg_pos['mean_fitness'], 1e-12, None)),
        s=agg_pos['cross_kp_count'] * 70,
        c=agg_pos['cross_kp_count'], cmap='RdYlGn', vmin=1, vmax=4,
        alpha=0.8, edgecolors='black', linewidths=0.4,
    )
    for _, row in agg_pos[agg_pos['mean_freq'] > 0.3].iterrows():
        lbl = row.get('mutation_label', str(int(row['pos_shifted'])))
        ax_scat.annotate(lbl,
                         (row['mean_freq'],
                          np.log10(max(row['mean_fitness'], 1e-12))),
                         textcoords='offset points', xytext=(4, 2), fontsize=7)
    plt.colorbar(sc, ax=ax_scat, shrink=0.85, label='No. KPs')
    ax_scat.set_xlabel('Mean frequency', fontsize=10)
    ax_scat.set_ylabel('Mean fitness (log₁₀)', fontsize=10)
    ax_scat.axvline(0.25, color='grey', linestyle='--', linewidth=0.8)
    ax_scat.set_title('(D)  Frequency vs Fitness', fontsize=12, fontweight='bold')

    # ── E: Step bubble ──────────────────────────────────────────────────────
    _ml4 = lambda x: x.mode().iloc[0] if len(x) > 0 else str(x.name)
    tp = (detail_df.groupby('pos_shifted')
          .agg(mean_first_step=('mean_first_step', 'mean'),
               mean_freq      =('freq',            'mean'),
               cross_kp_count =('kp',              'nunique'),
               mutation_label =('mutation_label',  _ml4))
          .reset_index()
          .nlargest(25, 'mean_freq'))
    sc2 = ax_step.scatter(
        tp['mean_first_step'], tp['mean_freq'],
        s=tp['cross_kp_count'] * 100,
        c=tp['cross_kp_count'], cmap='RdYlGn', vmin=1, vmax=4,
        alpha=0.85, edgecolors='black', linewidths=0.5,
    )
    for _, row in tp.iterrows():
        lbl = row.get('mutation_label', str(int(row['pos_shifted'])))
        ax_step.annotate(lbl,
                         (row['mean_first_step'], row['mean_freq']),
                         textcoords='offset points', xytext=(4, 2), fontsize=7)
    plt.colorbar(sc2, ax=ax_step, shrink=0.85, label='No. KPs')
    ax_step.set_xlabel('Mean step of first appearance', fontsize=10)
    ax_step.set_ylabel('Mean frequency', fontsize=10)
    ax_step.set_title('(E)  Early vs Late Compensators',
                      fontsize=12, fontweight='bold')

    fig.suptitle('PfDHFR Compensatory Ridge Analysis',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(out_dir, 'ridge_panel.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


# ─── tables ──────────────────────────────────────────────────────────────────

def write_tables(freq_pivot, detail_df, resistance_kps, out_dir, min_cross_kp=2):
    kp_cols = _kp_cols(freq_pivot, resistance_kps)

    full = freq_pivot.reset_index().rename(columns={'pos_shifted': 'position_shifted'})
    full['position_canonical'] = full['position_shifted'] + CANONICAL_OFFSET
    col_order = (['position_shifted', 'position_canonical', 'mutation_label']
                 + kp_cols
                 + ['mean_freq', 'max_freq', 'cross_kp_count'])
    full = full[[c for c in col_order if c in full.columns]]
    full = full.sort_values(['cross_kp_count', 'mean_freq'], ascending=False)
    full.to_csv(os.path.join(out_dir, 'ridge_summary_full.csv'),
                index=False, float_format='%.4f')

    candidates = full[full['cross_kp_count'] >= min_cross_kp].copy()
    candidates.to_csv(os.path.join(out_dir, 'ridge_candidates.csv'),
                      index=False, float_format='%.4f')
    return full, candidates


# ─── console report ──────────────────────────────────────────────────────────

def print_report(candidates, resistance_kps, full):
    kp_cols = _kp_cols(candidates, resistance_kps)
    W = 72

    def sep(c='─'): print(c * W)

    sep('═')
    print(f"  PfDHFR COMPENSATORY RIDGE — CANDIDATE RESIDUES")
    print(f"  (positions appearing in ≥2 of {len(resistance_kps)} resistance KP walks)")
    sep('═')

    hdr = (f"  {'Pos':>4}  {'Can':>4}  {'Mutation':>8}  {'CKP':>4}  {'MeanF':>6}  {'MaxF':>6}  "
           + '  '.join(f'KP{k:>3}' for k in kp_cols))
    print(hdr)
    sep()
    for _, row in candidates.head(30).iterrows():
        kp_str = '  '.join(
            f'{row.get(k, 0.0):>5.2f}' for k in kp_cols
        )
        mut_lbl = row.get('mutation_label', str(int(row['position_shifted'])))
        print(f"  {int(row['position_shifted']):>4}  "
              f"{int(row['position_canonical']):>4}  "
              f"{str(mut_lbl):>8}  "
              f"{int(row['cross_kp_count']):>4}  "
              f"{row['mean_freq']:>6.3f}  "
              f"{row['max_freq']:>6.3f}  "
              + kp_str)
    sep('═')
    print(f"  Total positions analysed : {len(full)}")
    print(f"  Ridge candidates (≥2 KPs): {len(candidates)}")

    # Cluster summary
    if not candidates.empty:
        top5 = candidates.head(5)
        print()
        print("  Top 5 candidates (shifted → canonical):")
        for _, row in top5.iterrows():
            mut_lbl = row.get('mutation_label', str(int(row['position_shifted'])))
            print(f"    {str(mut_lbl):<8}  (shifted {int(row['position_shifted'])} → canonical "
                  f"{int(row['position_canonical'])})  "
                  f"cross-KP {int(row['cross_kp_count'])}/4,  "
                  f"mean freq {row['mean_freq']:.2f}")
    sep('═')


# ─── oracle validation ────────────────────────────────────────────────────────

def load_oracle(path: str) -> pd.DataFrame:
    """Load oracle_wetlab23.csv. Mutation labels use the same shifted numbering
    as the simulator (canonical − 9), so parse_mutations works directly."""
    df = pd.read_csv(path)
    for col in ['F_Ki_Pyr', 'F_Ki_Cyc', 'F_eff', 'Ki_Pyr', 'kcat',
                'Km_H2F', 'Km_NADPH', 'kcatKm_H2F']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['mut_list']   = df['mutations'].apply(parse_mutations)
    df['n_muts']     = df['mut_list'].apply(len)
    df['positions']  = df['mut_list'].apply(
        lambda muts: frozenset(p for p, _, _ in muts))
    df['is_wt']      = df['mutations'].str.upper().str.strip() == 'WT'
    return df


def _step1_fitness(df: pd.DataFrame, resistance_kps: set) -> Dict[int, float]:
    """Mean walk fitness at step 1 for each resistance KP."""
    result: Dict[int, float] = {}
    for kp in sorted(resistance_kps):
        sub = df[(df['key_position'] == kp) &
                 (df['rep'] != 0) &
                 (df['step'] == 1)]
        if not sub.empty:
            result[kp] = float(sub['fitness'].mean())
    return result


def _oracle_single(oracle: pd.DataFrame, kp_to_aa: Dict[int, str]) -> Dict[int, Dict]:
    """Return oracle row data for each single-mutant KP."""
    out: Dict[int, Dict] = {}
    for kp, aa in kp_to_aa.items():
        mask = (oracle['n_muts'] == 1) & oracle['positions'].apply(
            lambda ps: kp in ps)
        row = oracle[mask]
        if not row.empty:
            out[kp] = row.iloc[0].to_dict()
    return out


def fig_oracle_step1_correlation(df: pd.DataFrame, oracle: pd.DataFrame,
                                 resistance_kps: set, out_dir: str):
    """Scatter: simulator step-1 walk fitness vs oracle F_eff for single mutants."""
    kp_to_aa = {42: 'I', 50: 'R', 99: 'N', 155: 'L'}
    step1 = _step1_fitness(df, resistance_kps)
    singles = _oracle_single(oracle, kp_to_aa)

    kps, sim_fit, ora_feff, ora_fki = [], [], [], []
    for kp in sorted(resistance_kps):
        if kp in step1 and kp in singles:
            kps.append(kp)
            sim_fit.append(step1[kp])
            ora_feff.append(singles[kp].get('F_eff', np.nan))
            ora_fki.append(singles[kp].get('F_Ki_Pyr', np.nan))

    if len(kps) < 2:
        return

    sim_fit_arr  = np.array(sim_fit,  dtype=float)
    ora_feff_arr = np.array(ora_feff, dtype=float)
    ora_fki_arr  = np.array(ora_fki,  dtype=float)

    labels = {42: 'N42I', 50: 'C50R', 99: 'S99N', 155: 'I155L'}
    colors = {42: '#d7191c', 50: '#1a9641', 99: '#2c7bb6', 155: '#fd8d3c'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # ── Panel A: step-1 fitness vs F_eff ────────────────────────────────────
    valid_mask = ~np.isnan(ora_feff_arr) & ~np.isnan(sim_fit_arr)
    if valid_mask.sum() >= 2:
        r, p = scipy_stats.pearsonr(
            np.log10(np.clip(ora_feff_arr[valid_mask], 1e-6, None)),
            sim_fit_arr[valid_mask]
        )
    else:
        r, p = np.nan, np.nan

    for i, kp in enumerate(kps):
        if np.isnan(ora_feff_arr[i]):
            continue
        ax1.scatter(
            np.log10(max(ora_feff_arr[i], 1e-6)), sim_fit_arr[i],
            s=200, color=colors.get(kp, 'grey'),
            edgecolors='black', linewidths=1.0, zorder=5
        )
        ax1.annotate(
            labels.get(kp, str(kp)),
            (np.log10(max(ora_feff_arr[i], 1e-6)), sim_fit_arr[i]),
            textcoords='offset points', xytext=(8, 4), fontsize=10, fontweight='bold'
        )

    # WT reference (step-0 baseline and oracle F_eff=1)
    step0 = df[(df['rep'] != 0) & (df['step'] == 0)]
    if not step0.empty:
        wt_fit = float(step0['fitness'].mean())
        ax1.scatter(0.0, wt_fit, s=200, marker='D', color='#888888',
                    edgecolors='black', linewidths=1.0, zorder=5)
        ax1.annotate('WT', (0.0, wt_fit), textcoords='offset points',
                     xytext=(8, 4), fontsize=10, fontweight='bold')

    if not np.isnan(r):
        ax1.set_title(
            f'(A)  Step-1 Walk Fitness vs Oracle Catalytic Efficiency\n'
            f'Pearson r = {r:.3f}  (p = {p:.3f})', fontsize=11)
    else:
        ax1.set_title('(A)  Step-1 Walk Fitness vs Oracle Catalytic Efficiency',
                      fontsize=11)

    ax1.set_xlabel('Oracle F_eff  (log₁₀)', fontsize=11)
    ax1.set_ylabel('Simulator step-1 mean fitness', fontsize=11)
    ax1.axhline(1.0, color='grey', linestyle='--', linewidth=0.8)
    ax1.axvline(0.0, color='grey', linestyle='--', linewidth=0.8)

    # ── Panel B: step-1 fitness vs F_Ki_Pyr ─────────────────────────────────
    valid_ki = ~np.isnan(ora_fki_arr) & ~np.isnan(sim_fit_arr)
    if valid_ki.sum() >= 2:
        r_ki, p_ki = scipy_stats.pearsonr(
            np.log10(np.clip(ora_fki_arr[valid_ki], 1e-6, None)),
            sim_fit_arr[valid_ki]
        )
    else:
        r_ki, p_ki = np.nan, np.nan

    for i, kp in enumerate(kps):
        if np.isnan(ora_fki_arr[i]):
            continue
        ax2.scatter(
            np.log10(max(ora_fki_arr[i], 1e-6)), sim_fit_arr[i],
            s=200, color=colors.get(kp, 'grey'),
            edgecolors='black', linewidths=1.0, zorder=5
        )
        ax2.annotate(
            labels.get(kp, str(kp)),
            (np.log10(max(ora_fki_arr[i], 1e-6)), sim_fit_arr[i]),
            textcoords='offset points', xytext=(8, 4), fontsize=10, fontweight='bold'
        )

    if not step0.empty:
        ax2.scatter(0.0, wt_fit, s=200, marker='D', color='#888888',
                    edgecolors='black', linewidths=1.0, zorder=5)
        ax2.annotate('WT', (0.0, wt_fit), textcoords='offset points',
                     xytext=(8, 4), fontsize=10, fontweight='bold')

    if not np.isnan(r_ki):
        ax2.set_title(
            f'(B)  Step-1 Walk Fitness vs Oracle Drug Resistance (F_Ki_Pyr)\n'
            f'Pearson r = {r_ki:.3f}  (p = {p_ki:.3f})', fontsize=11)
    else:
        ax2.set_title('(B)  Step-1 Walk Fitness vs Oracle Drug Resistance',
                      fontsize=11)

    ax2.set_xlabel('Oracle F_Ki_Pyr  (log₁₀)', fontsize=11)
    ax2.set_ylabel('Simulator step-1 mean fitness', fontsize=11)
    ax2.axhline(1.0, color='grey', linestyle='--', linewidth=0.8)
    ax2.axvline(0.0, color='grey', linestyle='--', linewidth=0.8)

    plt.suptitle('Oracle Validation: Simulator vs Experimental Single Mutants',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'oracle_step1_corr.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def fig_oracle_feff_landscape(oracle: pd.DataFrame, resistance_kps: set,
                              candidates: pd.DataFrame, out_dir: str):
    """Bar chart of oracle F_eff for all haplotypes with clinical pathway
    annotation and ridge-candidate marker."""
    plot_df = oracle.dropna(subset=['F_eff']).copy()
    plot_df = plot_df.sort_values('n_muts')

    labels_map = {
        'WT':                  'WT',
        'A7V':                 'A7V',
        'N42I':                'N42I',
        'C50R':                'C50R',
        'S99N':                'S99N',
        'S99T':                'S99T',
        'I155L':               'I155L',
        'A7V+S99T':            'A7V\n+S99T',
        'N42I+S99N':           'N42I\n+S99N',
        'C50R+S99N':           'C50R\n+S99N',
        'N42I+C50R+S99N':      'N42I+C50R\n+S99N',
        'C50R+S99N+I155L':     'C50R+S99N\n+I155L',
        'N42I+C50R+S99N+I155L':'N42I+C50R\n+S99N+I155L',
    }
    plot_df['label'] = plot_df['mutations'].map(
        lambda m: labels_map.get(m, m))

    n_mut_cmap = {0: '#4dac26', 1: '#b8e186', 2: '#f1b6da',
                  3: '#d01c8b', 4: '#7b3294'}
    bar_colors = [n_mut_cmap.get(int(n), '#aaaaaa') for n in plot_df['n_muts']]

    # Clinical pathway (primary KP99 route) positions
    clinical_pathway = {'WT', 'S99N', 'N42I+S99N', 'C50R+S99N',
                        'N42I+C50R+S99N', 'N42I+C50R+S99N+I155L'}

    fig, ax = plt.subplots(figsize=(13, 6))
    x = range(len(plot_df))
    bars = ax.bar(x, plot_df['F_eff'], color=bar_colors,
                  edgecolor='black', linewidth=0.7)

    # Outline clinical pathway bars
    for i, (_, row) in enumerate(plot_df.iterrows()):
        if row['mutations'] in clinical_pathway:
            bars[i].set_linewidth(2.5)
            bars[i].set_edgecolor('#000000')

    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df['label'], fontsize=8.5, ha='center')
    ax.set_ylabel('Catalytic efficiency (F_eff, relative to WT)', fontsize=11)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1.0,
               label='WT efficiency (F_eff = 1)')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 50)

    # Annotate low-fitness valley
    for i, (_, row) in enumerate(plot_df.iterrows()):
        if row['mutations'] in ('C50R+S99N', 'N42I+C50R+S99N', 'C50R+S99N+I155L'):
            ax.annotate('fitness\nvalley', (i, row['F_eff']),
                        xytext=(i, row['F_eff'] * 0.3),
                        fontsize=7.5, color='#d01c8b', ha='center',
                        arrowprops=dict(arrowstyle='->', color='#d01c8b',
                                        lw=0.8))

    # Legend
    patches = [mpatches.Patch(color=c, label=f'{n} mutations')
               for n, c in sorted(n_mut_cmap.items())]
    patches.append(mpatches.Patch(facecolor='white', edgecolor='black',
                                  linewidth=2.5, label='Clinical pathway'))
    ax.legend(handles=patches, fontsize=8.5, loc='upper left')

    # Ridge-candidate annotation: if ridge candidates exist annotate that
    # these haplotypes are where ridge positions are most needed
    if candidates is not None and not candidates.empty:
        n_cand = len(candidates)
        ax.text(0.98, 0.97,
                f'Ridge candidates predicted: {n_cand}\n'
                f'(compensate multi-drug backgrounds)',
                transform=ax.transAxes, fontsize=8.5, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                          edgecolor='goldenrod', linewidth=1.0))

    ax.set_title('(C)  Oracle Haplotype Fitness Landscape\n'
                 'bold outline = clinical pathway; simulator-predicted ridge '
                 'compensates the low-efficiency valley',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'oracle_feff_landscape.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def fig_clinical_corecruitment(df: pd.DataFrame, oracle: pd.DataFrame,
                               resistance_kps: set, out_dir: str):
    """For KP99 walks, track per-step co-recruitment fraction for known
    resistance positions and check against oracle clinical haplotypes."""
    kp = 99
    if kp not in resistance_kps:
        return

    sub = df[(df['key_position'] == kp) & (df['rep'] != 0)].copy()
    if sub.empty:
        return

    track_pos = sorted(resistance_kps - {kp})  # e.g. [42, 50, 155]
    max_step  = int(sub['step'].max())

    # For each step, fraction of walks that have co-recruited each position
    steps = sorted(sub['step'].unique())
    corecruit: Dict[int, List[float]] = {p: [] for p in track_pos}

    for s in steps:
        step_sub = sub[sub['step'] == s]
        n_reps   = step_sub['rep'].nunique()
        for p in track_pos:
            count = 0
            for _, row in step_sub.iterrows():
                muts = parse_mutations(row['mutations'])
                if any(mp == p for mp, _, _ in muts):
                    count += 1
            corecruit[p].append(count / n_reps if n_reps else 0.0)

    pos_labels = {42: 'N42I (can.51)', 50: 'C50R (can.59)', 155: 'I155L (can.164)'}
    pos_colors = {42: '#d7191c', 50: '#1a9641', 155: '#fd8d3c'}

    # Oracle F_eff values for relevant haplotypes to annotate
    ora_hap: Dict[str, float] = {}
    for _, row in oracle.iterrows():
        if not np.isnan(row.get('F_eff', np.nan)):
            ora_hap[row['mutations']] = row['F_eff']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── A: Co-recruitment trajectories ──────────────────────────────────────
    for p in track_pos:
        ax1.plot(steps, corecruit[p],
                 label=pos_labels.get(p, str(p)),
                 color=pos_colors.get(p, 'grey'),
                 linewidth=2.5, marker='o', markersize=4)

    ax1.set_xlabel('Mutation step', fontsize=11)
    ax1.set_ylabel('Fraction of KP99 walks co-recruiting position', fontsize=11)
    ax1.set_ylim(-0.02, 1.05)
    ax1.axhline(0.25, color='grey', linestyle='--', linewidth=0.8,
                label='25% threshold')
    ax1.legend(fontsize=9)
    ax1.set_title(f'(D)  KP99 (S99N) Walk Co-recruitment Dynamics\n'
                  f'n = {sub["rep"].nunique()} walks', fontsize=11)

    # ── B: Oracle F_eff for KP99-based clinical haplotypes ──────────────────
    kp99_haps = ['S99N', 'N42I+S99N', 'C50R+S99N',
                 'N42I+C50R+S99N', 'N42I+C50R+S99N+I155L']
    feff_vals  = [ora_hap.get(h, np.nan) for h in kp99_haps]
    colors_hap = ['#2c7bb6', '#d7191c', '#1a9641', '#8c510a', '#8073ac']
    valid_haps  = [(h, v, c) for h, v, c in zip(kp99_haps, feff_vals, colors_hap)
                   if not np.isnan(v)]

    if valid_haps:
        hap_labels, hap_vals, hap_cols = zip(*valid_haps)
        short_labels = [h.replace('+', '\n+') for h in hap_labels]
        ax2.bar(range(len(hap_vals)), hap_vals, color=hap_cols,
                edgecolor='black', linewidth=0.8)
        ax2.set_xticks(range(len(hap_vals)))
        ax2.set_xticklabels(short_labels, fontsize=8.5)
        ax2.set_yscale('log')
        ax2.set_ylim(0.01, 10)
        ax2.axhline(1.0, color='black', linestyle='--', linewidth=1.0,
                    label='WT F_eff')
        ax2.set_ylabel('Oracle F_eff (log scale)', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.set_title('(E)  Oracle Catalytic Efficiency — KP99 Clinical Pathway\n'
                      'fitness valley marks where compensatory ridge is needed',
                      fontsize=11)

        # Annotate final-step co-recruitment percentages
        final_step_data = sub[sub['step'] == max_step]
        n_final = final_step_data['rep'].nunique()
        for p in track_pos:
            count = 0
            for _, row in final_step_data.iterrows():
                muts = parse_mutations(row['mutations'])
                if any(mp == p for mp, _, _ in muts):
                    count += 1
            pct = 100 * count / n_final if n_final else 0
            pos_name = {42: 'N42I', 50: 'C50R', 155: 'I155L'}.get(p, str(p))
            ax2.text(0.02, 0.98 - track_pos.index(p) * 0.10,
                     f'Sim. final-step {pos_name}: {pct:.0f}%',
                     transform=ax2.transAxes, fontsize=8, va='top',
                     color=pos_colors.get(p, 'grey'))

    plt.suptitle('Oracle Validation: KP99 Walk Co-recruitment vs Clinical Pathway',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'oracle_kp99_corecruitment.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def fig_lethality_avoidance(df: pd.DataFrame, oracle: pd.DataFrame,
                            resistance_kps: set, out_dir: str):
    """Show that the simulator avoids experimentally lethal combinations.
    A7V+S99N is inactive in the oracle; check no walk co-recruits A7V (pos7)
    and S99N (pos99) together."""
    kp = 99
    if kp not in resistance_kps:
        return

    sub = df[(df['key_position'] == kp) & (df['rep'] != 0)].copy()
    if sub.empty:
        return

    # Count steps with pos7 present at all, and pos7 + pos99 together
    pos_lethal = 7   # A7V (shifted), canonical 16
    total_steps = len(sub)
    steps_with_lethal = 0
    for _, row in sub.iterrows():
        muts = parse_mutations(row['mutations'])
        if any(mp == pos_lethal for mp, _, _ in muts):
            steps_with_lethal += 1

    # Summary bar
    fig, ax = plt.subplots(figsize=(7, 5))
    categories = ['All KP99\nwalk steps', 'Steps with\nA7V (pos7)']
    values     = [total_steps, steps_with_lethal]
    bar_c      = ['#4dac26', '#d01c8b']
    bars = ax.bar(categories, values, color=bar_c, edgecolor='black', width=0.45)
    ax.bar_label(bars, padding=4, fontsize=12)
    ax.set_ylabel('Number of walk steps', fontsize=11)
    ax.set_title(
        '(F)  A7V+S99N Lethality Avoidance\n'
        'Oracle: A7V+S99N is enzymatically inactive\n'
        f'Simulator KP99 walks: {steps_with_lethal}/{total_steps} steps '
        f'contain A7V ({100*steps_with_lethal/total_steps:.1f}%)',
        fontsize=10
    )

    # Oracle annotation
    a7v_s99n_row = oracle[oracle['mutations'] == 'A7V+S99N']
    note = ''
    if not a7v_s99n_row.empty:
        note = a7v_s99n_row.iloc[0].get('notes', 'Inactive')
    ax.text(0.5, 0.60,
            f'Oracle: A7V+S99N — {note}',
            transform=ax.transAxes, ha='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffe0e0',
                      edgecolor='#d01c8b', linewidth=1.0))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'oracle_lethality_avoidance.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    return steps_with_lethal, total_steps


def fig_oracle_validation_panel(df: pd.DataFrame, oracle: pd.DataFrame,
                                resistance_kps: set, freq_pivot: pd.DataFrame,
                                candidates: pd.DataFrame, out_dir: str):
    """Combined publication-quality oracle validation figure."""
    kp_to_aa   = {42: 'I', 50: 'R', 99: 'N', 155: 'L'}
    step1      = _step1_fitness(df, resistance_kps)
    singles    = _oracle_single(oracle, kp_to_aa)
    kps        = sorted(k for k in resistance_kps if k in step1 and k in singles)
    labels_map = {42: 'N42I', 50: 'C50R', 99: 'S99N', 155: 'I155L'}
    colors_map = {42: '#d7191c', 50: '#1a9641', 99: '#2c7bb6', 155: '#fd8d3c'}

    fig = plt.figure(figsize=(20, 14))
    gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.40, hspace=0.52)
    ax_corr    = fig.add_subplot(gs[0, 0])
    ax_ki      = fig.add_subplot(gs[0, 1])
    ax_land    = fig.add_subplot(gs[0, 2])
    ax_corecr  = fig.add_subplot(gs[1, 0])
    ax_hap     = fig.add_subplot(gs[1, 1])
    ax_lethal  = fig.add_subplot(gs[1, 2])

    # ── A: F_eff correlation ─────────────────────────────────────────────────
    sim_fit_v  = [step1[k] for k in kps]
    ora_feff_v = [singles[k].get('F_eff', np.nan) for k in kps]
    valid      = [(s, f) for s, f in zip(sim_fit_v, ora_feff_v)
                  if not np.isnan(f)]

    if len(valid) >= 2:
        x_log = np.log10([max(f, 1e-6) for _, f in valid])
        y_sim = [s for s, _ in valid]
        r, pv = scipy_stats.pearsonr(x_log, y_sim)
        corr_str = f'r = {r:.3f},  p = {pv:.3f}'
    else:
        corr_str = ''

    for i, kp in enumerate(kps):
        feff = ora_feff_v[i]
        if np.isnan(feff):
            continue
        ax_corr.scatter(np.log10(max(feff, 1e-6)), sim_fit_v[i],
                        s=160, color=colors_map.get(kp, 'grey'),
                        edgecolors='black', linewidths=0.8, zorder=5)
        ax_corr.annotate(labels_map.get(kp, str(kp)),
                         (np.log10(max(feff, 1e-6)), sim_fit_v[i]),
                         textcoords='offset points', xytext=(6, 3), fontsize=9)

    step0 = df[(df['rep'] != 0) & (df['step'] == 0)]
    if not step0.empty:
        wt_fit = float(step0['fitness'].mean())
        ax_corr.scatter(0.0, wt_fit, s=160, marker='D', color='#888888',
                        edgecolors='black', linewidths=0.8, zorder=5)
        ax_corr.annotate('WT', (0.0, wt_fit),
                         textcoords='offset points', xytext=(6, 3), fontsize=9)

    ax_corr.axhline(1.0, color='grey', linestyle='--', linewidth=0.8)
    ax_corr.axvline(0.0, color='grey', linestyle='--', linewidth=0.8)
    ax_corr.set_xlabel('Oracle F_eff  (log₁₀)', fontsize=10)
    ax_corr.set_ylabel('Sim. step-1 fitness', fontsize=10)
    ax_corr.set_title(f'(A)  F_eff Correlation\n{corr_str}', fontsize=10)

    # ── B: F_Ki_Pyr correlation ──────────────────────────────────────────────
    ora_ki_v = [singles[k].get('F_Ki_Pyr', np.nan) for k in kps]
    valid_ki = [(s, f) for s, f in zip(sim_fit_v, ora_ki_v) if not np.isnan(f)]
    if len(valid_ki) >= 2:
        x_ki  = np.log10([max(f, 1e-6) for _, f in valid_ki])
        y_ki  = [s for s, _ in valid_ki]
        rk, pk = scipy_stats.pearsonr(x_ki, y_ki)
        ki_str = f'r = {rk:.3f},  p = {pk:.3f}'
    else:
        ki_str = ''

    for i, kp in enumerate(kps):
        fki = ora_ki_v[i]
        if np.isnan(fki):
            continue
        ax_ki.scatter(np.log10(max(fki, 1e-6)), sim_fit_v[i],
                      s=160, color=colors_map.get(kp, 'grey'),
                      edgecolors='black', linewidths=0.8, zorder=5)
        ax_ki.annotate(labels_map.get(kp, str(kp)),
                       (np.log10(max(fki, 1e-6)), sim_fit_v[i]),
                       textcoords='offset points', xytext=(6, 3), fontsize=9)

    if not step0.empty:
        ax_ki.scatter(0.0, wt_fit, s=160, marker='D', color='#888888',
                      edgecolors='black', linewidths=0.8, zorder=5)
        ax_ki.annotate('WT', (0.0, wt_fit),
                       textcoords='offset points', xytext=(6, 3), fontsize=9)

    ax_ki.axhline(1.0, color='grey', linestyle='--', linewidth=0.8)
    ax_ki.axvline(0.0, color='grey', linestyle='--', linewidth=0.8)
    ax_ki.set_xlabel('Oracle F_Ki_Pyr  (log₁₀)', fontsize=10)
    ax_ki.set_ylabel('Sim. step-1 fitness', fontsize=10)
    ax_ki.set_title(f'(B)  Drug Resistance Correlation\n{ki_str}', fontsize=10)

    # ── C: Oracle F_eff landscape ────────────────────────────────────────────
    plot_df = oracle.dropna(subset=['F_eff']).sort_values('n_muts')
    n_mut_cmap = {0: '#4dac26', 1: '#b8e186', 2: '#f1b6da',
                  3: '#d01c8b', 4: '#7b3294'}
    bar_colors = [n_mut_cmap.get(int(n), '#aaaaaa') for n in plot_df['n_muts']]
    clinical = {'WT', 'S99N', 'N42I+S99N', 'C50R+S99N',
                'N42I+C50R+S99N', 'N42I+C50R+S99N+I155L'}
    short_labs = [m.replace('+', '\n+') for m in plot_df['mutations']]

    x_land = range(len(plot_df))
    land_bars = ax_land.bar(x_land, plot_df['F_eff'], color=bar_colors,
                            edgecolor='black', linewidth=0.6)
    for i, (_, row) in enumerate(plot_df.iterrows()):
        if row['mutations'] in clinical:
            land_bars[i].set_linewidth(2.0)

    ax_land.set_xticks(list(x_land))
    ax_land.set_xticklabels(short_labs, fontsize=6.5, ha='center')
    ax_land.set_yscale('log')
    ax_land.axhline(1.0, color='black', linestyle='--', linewidth=0.9)
    ax_land.set_ylabel('F_eff (log scale)', fontsize=10)
    ax_land.set_title('(C)  Oracle Haplotype Landscape\n'
                      'bold = clinical pathway', fontsize=10)
    patches_land = [mpatches.Patch(color=c, label=f'{n} muts')
                    for n, c in sorted(n_mut_cmap.items())]
    ax_land.legend(handles=patches_land, fontsize=7, loc='upper right')

    # ── D: KP99 co-recruitment trajectories ─────────────────────────────────
    sub99 = df[(df['key_position'] == 99) & (df['rep'] != 0)].copy()
    if not sub99.empty:
        track_pos = sorted(resistance_kps - {99})
        steps99   = sorted(sub99['step'].unique())
        pos_cols99 = {42: '#d7191c', 50: '#1a9641', 155: '#fd8d3c'}
        pos_lab99  = {42: 'N42I', 50: 'C50R', 155: 'I155L'}

        for p in track_pos:
            fracs = []
            for s in steps99:
                ss   = sub99[sub99['step'] == s]
                nr   = ss['rep'].nunique()
                cnt  = sum(
                    any(mp == p for mp, _, _ in parse_mutations(r['mutations']))
                    for _, r in ss.iterrows()
                )
                fracs.append(cnt / nr if nr else 0.0)
            ax_corecr.plot(steps99, fracs,
                           label=pos_lab99.get(p, str(p)),
                           color=pos_cols99.get(p, 'grey'),
                           linewidth=2.0, marker='o', markersize=3)

        ax_corecr.axhline(0.25, color='grey', linestyle='--', linewidth=0.8)
        ax_corecr.set_ylim(-0.02, 1.05)
        ax_corecr.set_xlabel('Mutation step', fontsize=10)
        ax_corecr.set_ylabel('Fraction of walks', fontsize=10)
        ax_corecr.set_title(
            f'(D)  KP99 Co-recruitment Dynamics\n'
            f'n={sub99["rep"].nunique()} walks', fontsize=10)
        ax_corecr.legend(fontsize=8)

    # ── E: Oracle F_eff — KP99 clinical haplotype pathway ───────────────────
    ora_hap = {row['mutations']: row['F_eff']
               for _, row in oracle.iterrows()
               if not np.isnan(row.get('F_eff', np.nan))}
    kp99_haps   = ['S99N', 'N42I+S99N', 'C50R+S99N',
                   'N42I+C50R+S99N', 'N42I+C50R+S99N+I155L']
    kp99_feff   = [ora_hap.get(h, np.nan) for h in kp99_haps]
    kp99_colors = ['#2c7bb6', '#d7191c', '#1a9641', '#8c510a', '#8073ac']
    valid_kp99  = [(h, f, c) for h, f, c in zip(kp99_haps, kp99_feff, kp99_colors)
                   if not np.isnan(f)]
    if valid_kp99:
        vh, vf, vc = zip(*valid_kp99)
        sl  = [h.replace('+', '\n+') for h in vh]
        hb  = ax_hap.bar(range(len(vf)), vf, color=vc,
                         edgecolor='black', linewidth=0.7)
        ax_hap.set_xticks(range(len(vf)))
        ax_hap.set_xticklabels(sl, fontsize=7.5)
        ax_hap.set_yscale('log')
        ax_hap.set_ylim(0.01, 5)
        ax_hap.axhline(1.0, color='black', linestyle='--', linewidth=0.9)
        ax_hap.set_ylabel('Oracle F_eff (log scale)', fontsize=10)
        ax_hap.set_title('(E)  KP99 Clinical Pathway Efficiency\n'
                         'low F_eff = where ridge compensators are needed',
                         fontsize=10)

        # Annotate final-step co-recruitment
        if not sub99.empty:
            ms   = int(sub99['step'].max())
            fs99 = sub99[sub99['step'] == ms]
            nr99 = fs99['rep'].nunique()
            for ip, p in enumerate(track_pos):
                cnt = sum(
                    any(mp == p for mp, _, _ in parse_mutations(r['mutations']))
                    for _, r in fs99.iterrows()
                )
                pn = {42: 'N42I', 50: 'C50R', 155: 'I155L'}.get(p, str(p))
                ax_hap.text(
                    0.02, 0.98 - ip * 0.11,
                    f'Sim {pn}: {100*cnt/nr99:.0f}%' if nr99 else '',
                    transform=ax_hap.transAxes, fontsize=7.5, va='top',
                    color=pos_cols99.get(p, 'grey'))

    # ── F: Lethality avoidance ───────────────────────────────────────────────
    if 99 in resistance_kps:
        total99  = len(sub99)
        lethal99 = sum(
            any(mp == 7 for mp, _, _ in parse_mutations(r['mutations']))
            for _, r in sub99.iterrows()
        )
        cats = ['All KP99 steps', 'Steps with A7V']
        vals = [total99, lethal99]
        bl   = ax_lethal.bar(cats, vals, color=['#4dac26', '#d01c8b'],
                             edgecolor='black', width=0.45)
        ax_lethal.bar_label(bl, padding=4, fontsize=11)
        ax_lethal.set_ylabel('Walk steps', fontsize=10)
        ax_lethal.set_title(
            '(F)  A7V Lethality Avoidance\n'
            'Oracle: A7V+S99N = no DHFR activity\n'
            f'Sim KP99: {lethal99}/{total99} steps = '
            f'{100*lethal99/total99:.1f}% A7V',
            fontsize=10
        )

    fig.suptitle('Oracle Validation: PfPATH Simulator vs Experimental Data',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(out_dir, 'oracle_validation_panel.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def print_oracle_summary(df: pd.DataFrame, oracle: pd.DataFrame,
                         resistance_kps: set, candidates: pd.DataFrame):
    """Console report for oracle validation metrics."""
    W = 72
    def sep(c='─'): print(c * W)

    sep('═')
    print('  ORACLE VALIDATION SUMMARY')
    sep('═')

    kp_to_aa  = {42: 'I', 50: 'R', 99: 'N', 155: 'L'}
    step1     = _step1_fitness(df, resistance_kps)
    singles   = _oracle_single(oracle, kp_to_aa)
    ora_hap   = {row['mutations']: row['F_eff']
                 for _, row in oracle.iterrows()
                 if not np.isnan(row.get('F_eff', np.nan))}

    print('  Step-1 walk fitness vs oracle single-mutant data:')
    print(f"  {'KP':>5}  {'Mutation':>8}  {'Sim step-1':>12}  "
          f"{'Oracle F_eff':>14}  {'Oracle F_Ki':>12}")
    sep()
    sim_v, eff_v = [], []
    for kp in sorted(resistance_kps):
        mut = {42: 'N42I', 50: 'C50R', 99: 'S99N', 155: 'I155L'}.get(kp, '?')
        sf  = step1.get(kp, np.nan)
        fe  = singles.get(kp, {}).get('F_eff', np.nan)
        fk  = singles.get(kp, {}).get('F_Ki_Pyr', np.nan)
        print(f"  {kp:>5}  {mut:>8}  {sf:>12.4f}  "
              f"{fe if not np.isnan(fe) else 'N/A':>14}  "
              f"{fk if not np.isnan(fk) else 'N/A':>12}")
        if not np.isnan(sf) and not np.isnan(fe):
            sim_v.append(sf)
            eff_v.append(fe)

    if len(sim_v) >= 2:
        r, p = scipy_stats.pearsonr(
            np.log10(np.clip(eff_v, 1e-6, None)), sim_v)
        rsp, psp = scipy_stats.spearmanr(eff_v, sim_v)
        print()
        print(f'  Pearson r (log F_eff vs sim fitness):  {r:.3f}  (p={p:.3f})')
        print(f'  Spearman ρ:                            {rsp:.3f}  (p={psp:.3f})')

    sep()
    print('  Oracle F_eff — KP99 clinical pathway:')
    kp99_haps = ['S99N', 'N42I+S99N', 'C50R+S99N',
                 'N42I+C50R+S99N', 'N42I+C50R+S99N+I155L']
    for h in kp99_haps:
        fv = ora_hap.get(h, np.nan)
        print(f"    {h:<30}  F_eff = "
              f"{fv:.4f}" if not np.isnan(fv) else f"    {h:<30}  F_eff = N/A")

    sep()
    if 99 in resistance_kps:
        sub99 = df[(df['key_position'] == 99) & (df['rep'] != 0)]
        total  = len(sub99)
        lethal = sum(
            any(mp == 7 for mp, _, _ in parse_mutations(r['mutations']))
            for _, r in sub99.iterrows()
        )
        print(f'  A7V lethality avoidance (KP99 walks):')
        print(f'    Oracle A7V+S99N = enzymatically inactive')
        print(f'    Simulator: {lethal}/{total} steps contain A7V '
              f'({100*lethal/total:.2f}%)  '
              + ('✓ avoids lethal combo' if lethal == 0 else '! A7V present!'))

    sep()
    if not candidates.empty:
        print(f'  Ridge candidates vs oracle fitness valleys:')
        print(f'    {len(candidates)} ridge candidates predicted by simulator')
        print(f'    C50R+S99N F_eff = {ora_hap.get("C50R+S99N", np.nan):.4f} '
              f'(triple/quad backgrounds are primary targets for ridge compensation)')
        print(f'    Canonical ridge cluster: '
              + ', '.join(str(to_canonical(int(p)))
                          for p in list(candidates['position_shifted'])[:5]))
    sep('═')


# ─── ESM2 structural analysis ────────────────────────────────────────────────

def _esm_patch_torch_load():
    import torch
    _orig = torch.load
    def _load(*a, **kw):
        kw.setdefault('weights_only', False)
        return _orig(*a, **kw)
    torch.load = _load


def _esm_load_model(model_path: str, device: str):
    _esm_patch_torch_load()
    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    return model.to(device).eval(), alphabet


def _esm_apply_mutations(wt: str,
                         muts: List[Tuple[int, str, str]]) -> str:
    """Apply (shifted_pos, wt_aa, mut_aa) mutations to WT sequence."""
    seq = list(wt)
    for pos, wt_aa, mut_aa in muts:
        idx = pos - 1
        if seq[idx] != wt_aa:
            raise ValueError(
                f"Mismatch at shifted pos {pos}: expected {wt_aa}, got {seq[idx]}")
        seq[idx] = mut_aa
    return ''.join(seq)


def _esm_run_inference(model, alphabet, sequences: List[Tuple[str, str]],
                       device: str) -> Dict[str, Dict]:
    import torch, time
    from tqdm import tqdm
    batch_converter = alphabet.get_batch_converter()
    results: Dict[str, Dict] = {}
    bar = tqdm(sequences, desc='ESM2 inference', unit='seq',
               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    for name, seq in bar:
        bar.set_postfix_str(name, refresh=True)
        t0 = time.time()
        _, _, tokens = batch_converter([(name, seq)])
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[33],
                        need_head_weights=True, return_contacts=True)
        results[name] = {
            'contacts':   out['contacts'][0].cpu().numpy(),
            'embeddings': out['representations'][33][0, 1:-1].cpu().numpy(),
        }
        elapsed = time.time() - t0
        tqdm.write(f"  [{name}]  {elapsed:.1f}s")
    return results


def _esm_save_cache(results: Dict, path: str):
    arrays = {}
    for name, r in results.items():
        key = name.replace('+', '__').replace(' ', '_')
        arrays[f'{key}__contacts']   = r['contacts']
        arrays[f'{key}__embeddings'] = r['embeddings']
    np.savez_compressed(path, **arrays)


def _esm_load_cache(path: str) -> Dict[str, Dict]:
    data = np.load(path)
    results: Dict[str, Dict] = {}
    keys = {k.rsplit('__', 1)[0] for k in data.files}
    for k in keys:
        name = k.replace('__', '+')
        results[name] = {
            'contacts':   data[f'{k}__contacts'],
            'embeddings': data[f'{k}__embeddings'],
        }
    return results


def _esm_perturb(wt_emb: np.ndarray, mut_emb: np.ndarray) -> np.ndarray:
    return np.linalg.norm(mut_emb - wt_emb, axis=-1)


def _esm_single_keys(results: Dict) -> List[str]:
    """All single-mutant haplotype names present in results, in definition order."""
    return [name for name, muts in _ESM_HAPLOTYPES
            if len(muts) == 1 and name in results]


def _esm_panel_singles(results: Dict) -> List[str]:
    """Primary single mutants for per-panel figures (subset of _ESM_PANEL_SINGLES)."""
    return [k for k in _ESM_PANEL_SINGLES if k in results]


def _esm_sort_ridge_by_structure(wt_contacts: np.ndarray,
                                  ridge_pos: List[int],
                                  res_pos: List[int],
                                  L: int) -> List[int]:
    """Sort ridge positions by max ESM2 contact probability to any resistance position.
    Ensures the most structurally coupled ridge candidates appear first in figures."""
    vr = [p for p in ridge_pos if 1 <= p <= L]
    vres = [p for p in res_pos if 1 <= p <= L]
    scores = {rp: max(wt_contacts[rp-1, rsp-1] for rsp in vres) for rp in vr}
    return sorted(vr, key=lambda p: scores[p], reverse=True)


def _esm_add_lines(ax, res_pos, ridge_pos, L, axis='both'):
    for p in res_pos:
        idx = p - 1
        if 0 <= idx < L:
            if axis in ('x', 'both'):
                ax.axvline(idx, color='red',  lw=0.8, alpha=0.8, ls='--')
            if axis in ('y', 'both'):
                ax.axhline(idx, color='red',  lw=0.8, alpha=0.8, ls='--')
    for p in ridge_pos:
        idx = p - 1
        if 0 <= idx < L:
            if axis in ('x', 'both'):
                ax.axvline(idx, color='blue', lw=0.8, alpha=0.55, ls=':')
            if axis in ('y', 'both'):
                ax.axhline(idx, color='blue', lw=0.8, alpha=0.55, ls=':')


def fig_esm_contact_deltas(results: Dict, res_pos: List[int],
                            ridge_pos: List[int], out_dir: str):
    show = [k for k in ['S99N', 'C50R+S99N', 'N42I+C50R+S99N',
                         'N42I+C50R+S99N+I155L'] if k in results]
    if not show:
        return
    wt_c = results['WT']['contacts']
    L    = wt_c.shape[0]
    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.3)

    fig, axes = plt.subplots(1, len(show), figsize=(5.5 * len(show), 5.5))
    if len(show) == 1:
        axes = [axes]
    for ax, key in zip(axes, show):
        delta = results[key]['contacts'] - wt_c
        im = ax.imshow(delta, cmap='RdBu_r', norm=norm,
                       origin='lower', aspect='auto')
        _esm_add_lines(ax, res_pos, ridge_pos, L)
        plt.colorbar(im, ax=ax, shrink=0.72, label='ΔContact')
        ax.set_title(f'Δ{key} − WT', fontsize=10, fontweight='bold')
        ax.set_xlabel('Residue (shifted)', fontsize=9)
        ax.set_ylabel('Residue (shifted)', fontsize=9)

    res_p  = mpatches.Patch(color='red',  label='Resistance pos.')
    rdg_p  = mpatches.Patch(color='blue', label='Ridge candidate')
    fig.legend(handles=[res_p, rdg_p], loc='lower center', ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('ESM2 ΔContact Maps: Resistance Mutation Effects',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'esm_contact_deltas.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def fig_esm_embedding_perturbation(results: Dict, res_pos: List[int],
                                    ridge_pos: List[int], out_dir: str):
    wt_emb = results['WT']['embeddings']
    L = wt_emb.shape[0]

    # Panel singles: primary resistance mutations only (keeps figure readable).
    # All singles feed into the mean perturbation profile figure.
    singles = _esm_panel_singles(results)
    multis  = [k for k in ['C50R+S99N', 'N42I+C50R+S99N',
                             'N42I+C50R+S99N+I155L'] if k in results]
    if not singles:
        return

    nrows = max(len(singles), len(multis))
    fig, axes = plt.subplots(nrows, 2, figsize=(14, max(nrows * 2.8, 8)),
                              sharex=True)
    if nrows == 1:
        axes = np.array([axes])
    axes = np.array(axes)

    cmap_s = plt.cm.get_cmap('tab10', max(len(singles), 1))
    cols_s = [cmap_s(i) for i in range(len(singles))]
    cols_m = ['#7b3294', '#8c510a', '#3d3d3d']
    x = np.arange(1, L + 1)

    def _panel(ax, key, color):
        perturb = _esm_perturb(wt_emb, results[key]['embeddings'])
        ax.fill_between(x, perturb, alpha=0.45, color=color)
        ax.plot(x, perturb, color=color, lw=0.8)
        for p in res_pos:
            if 1 <= p <= L:
                ax.axvline(p, color='red', lw=1.0, alpha=0.8, ls='--')
        for p in ridge_pos:
            if 1 <= p <= L:
                ax.axvspan(p - 1, p + 1, color='blue', alpha=0.15)
        ax.set_ylabel('‖Δemb‖₂', fontsize=8)
        ax.set_title(key, fontsize=9, fontweight='bold')

    for i, (key, col) in enumerate(zip(singles, cols_s)):
        if i < nrows:
            _panel(axes[i, 0], key, col)
    for i, (key, col) in enumerate(zip(multis, cols_m)):
        if i < nrows:
            _panel(axes[i, 1], key, col)
    for i in range(len(singles), nrows):
        axes[i, 0].set_visible(False)
    for i in range(len(multis), nrows):
        axes[i, 1].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel('Residue position (shifted)', fontsize=9)

    r_p = mpatches.Patch(color='red',  alpha=0.8, label='Resistance pos.')
    b_p = mpatches.Patch(color='blue', alpha=0.3, label='Ridge candidates')
    fig.legend(handles=[r_p, b_p], loc='upper right', fontsize=9,
               bbox_to_anchor=(1.0, 1.0))
    fig.suptitle('ESM2 Embedding Perturbation: Structural Influence of Mutations',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'esm_embedding_perturbation.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def fig_esm_perturbation_profile(results: Dict, res_pos: List[int],
                                  ridge_pos: List[int], out_dir: str):
    wt_emb  = results['WT']['embeddings']
    L = wt_emb.shape[0]
    x = np.arange(1, L + 1)
    singles = _esm_single_keys(results)
    if not singles:
        return

    perturbs     = np.stack([_esm_perturb(wt_emb, results[k]['embeddings'])
                              for k in singles], axis=0)
    mean_perturb = perturbs.mean(axis=0)
    ranks        = np.argsort(mean_perturb)[::-1] + 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    cmap = plt.cm.get_cmap('tab10', max(len(singles), 1))
    cols = [cmap(i) for i in range(len(singles))]

    for i, key in enumerate(singles):
        ax1.plot(x, perturbs[i], color=cols[i], lw=0.9, alpha=0.8, label=key)
    for p in res_pos:
        if 1 <= p <= L:
            ax1.axvline(p, color='red', lw=1.2, ls='--', alpha=0.7)
    for p in ridge_pos:
        if 1 <= p <= L:
            ax1.axvspan(p - 0.8, p + 0.8, color='steelblue', alpha=0.18)
    ax1.set_ylabel('‖Δemb‖₂', fontsize=10)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_title('ESM2 Per-Residue Embedding Perturbation (single mutants vs WT)',
                  fontsize=11)

    q75 = np.percentile(mean_perturb, 75)
    ax2.fill_between(x, mean_perturb, alpha=0.4, color='#636363')
    ax2.plot(x, mean_perturb, color='#252525', lw=1.0, label='Mean')
    ax2.fill_between(x, mean_perturb, where=mean_perturb > q75,
                     alpha=0.6, color='#d73027', label='Top 25%')
    for p in res_pos:
        if 1 <= p <= L:
            ax2.axvline(p, color='red', lw=1.2, ls='--', alpha=0.9)
    for p in ridge_pos:
        if 1 <= p <= L:
            ax2.axvspan(p - 0.8, p + 0.8, color='steelblue', alpha=0.35)
            ax2.text(p, mean_perturb[p - 1] + mean_perturb.max() * 0.04,
                     f'Rd{p}\n#{ranks[p-1]}',
                     color='steelblue', fontsize=6, ha='center',
                     va='bottom', fontweight='bold')
    ax2.set_xlabel('Residue position (shifted)', fontsize=10)
    ax2.set_ylabel('Mean ‖Δemb‖₂', fontsize=10)
    rd_p = mpatches.Patch(color='steelblue', alpha=0.35, label='Ridge candidates')
    h, l = ax2.get_legend_handles_labels()
    ax2.legend(handles=h + [rd_p], fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'esm_perturbation_profile.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def fig_esm_ridge_coupling(results: Dict, res_pos: List[int],
                            ridge_pos: List[int], out_dir: str):
    if not ridge_pos:
        print("  [skip] fig_esm_ridge_coupling: no ridge positions — skipping.")
        return
    wt_c   = results['WT']['contacts']
    wt_emb = results['WT']['embeddings']
    L = wt_c.shape[0]
    vres = [p for p in res_pos if 1 <= p <= L]
    # Sort by structural relevance so most-coupled positions are at top
    vr = _esm_sort_ridge_by_structure(wt_c, ridge_pos, res_pos, L)

    if not vr or not vres:
        print("  [skip] fig_esm_ridge_coupling: empty ridge or resistance position list.")
        return

    contact_mat = np.array([[wt_c[rp-1, resp-1] for resp in vres]
                             for rp in vr])

    singles = _esm_single_keys(results)
    perturb_mat = np.array([
        [_esm_perturb(wt_emb, results[k]['embeddings'])[rp-1]
         for k in singles]
        for rp in vr
    ])

    res_labs   = [f'R{p}\n(c{to_canonical(p)})' for p in vres]
    ridge_labs = [f'Rd{p}\n(c{to_canonical(p)})' for p in vr]

    fig, (ax1, ax2) = plt.subplots(1, 2,
                                    figsize=(14, 0.65 * len(vr) + 3))
    im1 = ax1.imshow(contact_mat, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax1.set_xticks(range(len(vres)))
    ax1.set_xticklabels(res_labs, fontsize=9)
    ax1.set_yticks(range(len(vr)))
    ax1.set_yticklabels(ridge_labs, fontsize=9)
    for i in range(contact_mat.shape[0]):
        for j in range(contact_mat.shape[1]):
            v = contact_mat[i, j]
            ax1.text(j, i, f'{v:.2f}', ha='center', va='center',
                     fontsize=8, color='white' if v > 0.6 else 'black')
    plt.colorbar(im1, ax=ax1, shrink=0.75, label='ESM2 contact prob. (WT)')
    ax1.set_title('(A)  Contact Probability\nRidge × Resistance (WT)', fontsize=10)

    vp = max(perturb_mat.max(), 0.01)
    im2 = ax2.imshow(perturb_mat, cmap='Blues', vmin=0, vmax=vp, aspect='auto')
    ax2.set_xticks(range(len(singles)))
    ax2.set_xticklabels(singles, fontsize=9)
    ax2.set_yticks(range(len(vr)))
    ax2.set_yticklabels(ridge_labs, fontsize=9)
    for i in range(perturb_mat.shape[0]):
        for j in range(perturb_mat.shape[1]):
            v = perturb_mat[i, j]
            ax2.text(j, i, f'{v:.2f}', ha='center', va='center',
                     fontsize=8, color='white' if v > 0.6 * vp else 'black')
    plt.colorbar(im2, ax=ax2, shrink=0.75, label='‖Δemb‖₂')
    ax2.set_title('(B)  Embedding Perturbation at Ridge Positions\n'
                  'per Single Resistance Mutation', fontsize=10)

    fig.suptitle('ESM2 Structural Coupling: Ridge Candidates ↔ Resistance Positions',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'esm_ridge_coupling.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def fig_esm_ridge_contact_partners(results: Dict, res_pos: List[int],
                                    ridge_pos: List[int], out_dir: str,
                                    top_n: int = 20):
    wt_c = results['WT']['contacts']
    L = wt_c.shape[0]
    # Sort by structural relevance, show top 8
    vr = _esm_sort_ridge_by_structure(wt_c, ridge_pos, res_pos, L)[:8]

    ncols = 2
    nrows = (len(vr) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 4.0))
    if nrows == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for ax, rp in zip(axes, vr):
        row = wt_c[rp - 1].copy()
        row[rp - 1] = 0.0
        top_idx  = np.argsort(row)[::-1][:top_n]
        top_pos  = [i + 1 for i in top_idx]
        top_vals = row[top_idx]
        cols = ['#d73027' if p in res_pos
                else '#3288bd' if p in ridge_pos
                else '#4dac26'
                for p in top_pos]
        ax.bar(range(len(top_pos)), top_vals, color=cols,
               edgecolor='black', linewidth=0.4)
        ax.set_xticks(range(len(top_pos)))
        ax.set_xticklabels([str(p) for p in top_pos], fontsize=6.5, rotation=90)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color='grey', ls='--', lw=0.7)
        for i, (p, v) in enumerate(zip(top_pos, top_vals)):
            if p in res_pos:
                ax.text(i, v + 0.02, f'R{p}', color='red',
                        fontsize=6.5, ha='center', fontweight='bold')
            elif p in ridge_pos:
                ax.text(i, v + 0.02, f'Rd{p}', color='#3288bd',
                        fontsize=6.5, ha='center')
        ax.set_ylabel('Contact prob.', fontsize=9)
        ax.set_title(f'Ridge {rp} (can.{to_canonical(rp)})  — top-{top_n} contacts (WT)',
                     fontsize=9, fontweight='bold')

    for ax in axes[len(vr):]:
        ax.set_visible(False)

    r_p  = mpatches.Patch(color='#d73027', label='Resistance pos.')
    rd_p = mpatches.Patch(color='#3288bd', label='Another ridge pos.')
    o_p  = mpatches.Patch(color='#4dac26', label='Other pos.')
    fig.legend(handles=[r_p, rd_p, o_p], loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('ESM2 Top Contact Partners per Ridge Candidate (WT)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'esm_ridge_contact_partners.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def fig_esm_panel(results: Dict, res_pos: List[int],
                  ridge_pos: List[int], out_dir: str):
    """Combined publication ESM2 figure (5 panels)."""
    wt_c   = results['WT']['contacts']
    wt_emb = results['WT']['embeddings']
    L = wt_c.shape[0]
    vres = [p for p in res_pos if 1 <= p <= L]
    # Sort ridge by structural coupling so strongest contact comes first
    vr = _esm_sort_ridge_by_structure(wt_c, ridge_pos, res_pos, L)
    x = np.arange(1, L + 1)

    fig = plt.figure(figsize=(22, 16))
    gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.40, hspace=0.52)
    ax_da = fig.add_subplot(gs[0, 0])
    ax_db = fig.add_subplot(gs[0, 1])
    ax_cp = fig.add_subplot(gs[0, 2])
    ax_pf = fig.add_subplot(gs[1, :2])
    ax_rp = fig.add_subplot(gs[1, 2])

    norm = TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.3)

    # A: ΔContact S99N
    if 'S99N' in results:
        delta = results['S99N']['contacts'] - wt_c
        im = ax_da.imshow(delta, cmap='RdBu_r', norm=norm,
                           origin='lower', aspect='auto')
        _esm_add_lines(ax_da, vres, vr, L)
        plt.colorbar(im, ax=ax_da, shrink=0.80, label='ΔContact')
        ax_da.set_title('(A)  ΔContact: S99N − WT', fontsize=10, fontweight='bold')
        ax_da.set_xlabel('Residue (shifted)', fontsize=9)
        ax_da.set_ylabel('Residue (shifted)', fontsize=9)

    # B: ΔContact Triple
    if 'N42I+C50R+S99N' in results:
        delta = results['N42I+C50R+S99N']['contacts'] - wt_c
        im = ax_db.imshow(delta, cmap='RdBu_r', norm=norm,
                           origin='lower', aspect='auto')
        _esm_add_lines(ax_db, vres, vr, L)
        plt.colorbar(im, ax=ax_db, shrink=0.80, label='ΔContact')
        ax_db.set_title('(B)  ΔContact: Triple − WT', fontsize=10, fontweight='bold')
        ax_db.set_xlabel('Residue (shifted)', fontsize=9)

    # C: Contact coupling heatmap — top 8 by structural coupling (already sorted)
    cmat = np.array([[wt_c[rp-1, rsp-1] for rsp in vres] for rp in vr[:8]])
    im = ax_cp.imshow(cmat, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax_cp.set_xticks(range(len(vres)))
    ax_cp.set_xticklabels([f'R{p}\n(c{to_canonical(p)})' for p in vres], fontsize=8)
    ax_cp.set_yticks(range(len(vr[:8])))
    ax_cp.set_yticklabels([f'Rd{p} (c{to_canonical(p)})' for p in vr[:8]], fontsize=8)
    for i in range(cmat.shape[0]):
        for j in range(cmat.shape[1]):
            v = cmat[i, j]
            ax_cp.text(j, i, f'{v:.2f}', ha='center', va='center',
                       fontsize=7.5, color='white' if v > 0.6 else 'black')
    plt.colorbar(im, ax=ax_cp, shrink=0.80, label='Contact prob. (WT)')
    ax_cp.set_title('(C)  Ridge × Resistance\nContact Probability (WT)',
                    fontsize=10, fontweight='bold')

    # D: Mean perturbation profile
    singles = _esm_single_keys(results)
    if singles:
        perturbs     = np.stack([_esm_perturb(wt_emb, results[k]['embeddings'])
                                  for k in singles])
        mean_p       = perturbs.mean(axis=0)
        q75          = np.percentile(mean_p, 75)
        ranks        = np.argsort(mean_p)[::-1] + 1
        ax_pf.fill_between(x, mean_p, alpha=0.35, color='#636363')
        ax_pf.plot(x, mean_p, color='#252525', lw=0.9, label='Mean ‖Δemb‖₂')
        ax_pf.fill_between(x, mean_p, where=mean_p > q75,
                            alpha=0.55, color='#d73027', label='Top 25%')
        for p in vres:
            ax_pf.axvline(p, color='red', lw=1.3, ls='--', alpha=0.85)
        for p in vr:
            if 1 <= p <= L:
                ax_pf.axvspan(p - 0.8, p + 0.8, color='steelblue', alpha=0.35)
                ax_pf.text(p, mean_p[p-1] + mean_p.max() * 0.04,
                            f'Rd{p}\n#{ranks[p-1]}',
                            color='steelblue', fontsize=6, ha='center',
                            va='bottom', fontweight='bold')
        ax_pf.set_xlabel('Residue position (shifted)', fontsize=10)
        ax_pf.set_ylabel('Mean ‖Δemb‖₂', fontsize=10)
        rd_p = mpatches.Patch(color='steelblue', alpha=0.35, label='Ridge candidates')
        h, l = ax_pf.get_legend_handles_labels()
        ax_pf.legend(handles=h + [rd_p], fontsize=8)
        ax_pf.set_title('(D)  Mean Embedding Perturbation (single mutants vs WT)',
                         fontsize=10, fontweight='bold')

    # E: Top contact partners for the most structurally coupled ridge candidate
    if vr:
        rp = vr[0]  # already sorted by max contact to any resistance position
        row = wt_c[rp - 1].copy()
        row[rp - 1] = 0.0
        top_idx  = np.argsort(row)[::-1][:20]
        top_pos  = [i + 1 for i in top_idx]
        top_vals = row[top_idx]
        cols = ['#d73027' if p in vres
                else '#3288bd' if p in vr
                else '#4dac26'
                for p in top_pos]
        ax_rp.bar(range(len(top_pos)), top_vals, color=cols,
                  edgecolor='black', lw=0.4)
        ax_rp.set_xticks(range(len(top_pos)))
        ax_rp.set_xticklabels([str(p) for p in top_pos], fontsize=7, rotation=90)
        ax_rp.axhline(0.5, color='grey', ls='--', lw=0.7)
        ax_rp.set_ylim(0, 1.05)
        for i, (p, v) in enumerate(zip(top_pos, top_vals)):
            if p in vres:
                ax_rp.text(i, v + 0.03, f'R{p}', color='red',
                            fontsize=6.5, ha='center', fontweight='bold')
        r_p  = mpatches.Patch(color='#d73027', label='Resistance')
        rp_p = mpatches.Patch(color='#3288bd', label='Ridge')
        ax_rp.legend(handles=[r_p, rp_p], fontsize=7)
        ax_rp.set_ylabel('Contact prob.', fontsize=9)
        ax_rp.set_title(f'(E)  Ridge {rp} (can.{to_canonical(rp)})\nTop-20 contacts (WT)',
                         fontsize=10, fontweight='bold')

    fig.suptitle('ESM2 Structural Analysis: Ridge Candidates vs Resistance Mutations',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(out_dir, 'esm_panel.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def print_esm_report(results: Dict, res_pos: List[int], ridge_pos: List[int]):
    wt_c   = results['WT']['contacts']
    wt_emb = results['WT']['embeddings']
    L = wt_c.shape[0]
    vres = [p for p in res_pos if 1 <= p <= L]
    # Sort by structural relevance for the report
    vr = _esm_sort_ridge_by_structure(wt_c, ridge_pos, res_pos, L)

    W = 78
    def sep(c='─'): print(c * W)

    singles = _esm_single_keys(results)
    perturbs = {k: _esm_perturb(wt_emb, results[k]['embeddings']) for k in singles}
    ranks    = {k: np.argsort(perturbs[k])[::-1] + 1 for k in singles}

    sep('═')
    print('  ESM2 STRUCTURAL ANALYSIS SUMMARY  (ridge sorted by max contact to resistance)')
    sep('═')
    res_hdr = '  '.join(f'R{p:>3}' for p in vres)
    per_hdr = '  '.join(f'{k:>13}' for k in singles)
    print(f"  {'Ridge':>6}  {'Can':>4}  {res_hdr}  {'MaxC':>5}  {per_hdr}")
    sep()
    for rp in vr:
        c_vals   = [wt_c[rp-1, rsp-1] for rsp in vres]
        max_c    = max(c_vals)
        c_str    = '  '.join(f'{v:>5.3f}' for v in c_vals)
        flag     = ' ◄' if max_c >= 0.5 else ('  ~' if max_c >= 0.1 else '')
        if singles:
            p_str = '  '.join(
                f'{perturbs[k][rp-1]:>5.2f}(#{ranks[k][rp-1]:>3})'
                for k in singles)
        else:
            p_str = ''
        print(f"  Rd{rp:>4}  {to_canonical(rp):>4}  {c_str}  {max_c:>5.3f}{flag}  {p_str}")
    sep('═')


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="PfPATH walk evaluator — compensatory ridge analysis."
    )
    p.add_argument('--walks', required=True,
                   help='Path to walks.csv from a PfPATH run.')
    p.add_argument('--out', default=None,
                   help='Output directory (default: same directory as walks.csv).')
    p.add_argument('--resistance_kps', type=str, default='42,50,99,155',
                   help='Comma-separated shifted positions of primary resistance '
                        'mutations (default: 42,50,99,155).')
    p.add_argument('--min_freq', type=float, default=0.05,
                   help='Minimum frequency to include a position (default 0.05).')
    p.add_argument('--min_cross_kp', type=int, default=2,
                   help='Minimum cross-KP count for ridge_candidates.csv (default 2).')
    p.add_argument('--oracle', default=None,
                   help='Path to oracle_wetlab23.csv for experimental validation. '
                        'Enables oracle validation figures and console report.')
    p.add_argument('--wt', default=None,
                   help='WT PfDHFR FASTA file. Required for ESM2 analysis.')
    p.add_argument('--esm_model',
                   default='/srv/shared/esm2/esm2_t33_650M_UR50D.pt',
                   help='Path to ESM2 model weights. Pass this flag (or --esm_cache) '
                        'to enable ESM2 structural analysis.')
    p.add_argument('--esm_cache', default=None,
                   help='Path to a previously saved esm_results.npz. If provided and '
                        'the file exists, inference is skipped. Defaults to '
                        '<out>/esm_results.npz when --wt or --esm_model is given.')
    p.add_argument('--esm_device', default='auto',
                   help='Device for ESM2 inference: auto | cpu | cuda:0 | cuda:1 | '
                        'cuda:2 (default: auto → cuda:0 if available, else cpu).')
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = args.out or os.path.dirname(os.path.abspath(args.walks))
    os.makedirs(out_dir, exist_ok=True)

    resistance_kps = set(int(x) for x in args.resistance_kps.split(',') if x.strip())

    print(f"Loading  : {args.walks}")
    df = load_walks(args.walks)
    print(f"Loaded   : {len(df)} rows, "
          f"{df[df['rep']!=0]['rep'].nunique()} walks, "
          f"{df['key_position'].nunique()} key positions")
    print(f"Resistance KPs analysed: {sorted(resistance_kps)}")
    print(f"Output   : {out_dir}")
    print()

    print("Computing ridge statistics...")
    detail_df, freq_pivot = compute_ridge_stats(df, resistance_kps, args.min_freq)

    if detail_df.empty:
        print("No secondary positions found above min_freq threshold. "
              "Check that resistance_kps match key positions in walks.csv.")
        return

    print("Generating figures...")
    fig_heatmap(freq_pivot, resistance_kps, out_dir)
    fig_ranking(freq_pivot, resistance_kps, out_dir)
    fig_freq_fitness(detail_df, resistance_kps, out_dir)
    fig_per_kp(detail_df, resistance_kps, out_dir)
    fig_dynamics(df, detail_df, resistance_kps, out_dir)
    fig_master_panel(freq_pivot, detail_df, df, resistance_kps, out_dir)

    print("Writing tables...")
    full, candidates = write_tables(
        freq_pivot, detail_df, resistance_kps, out_dir, args.min_cross_kp
    )

    print_report(candidates, resistance_kps, full)

    # ── Oracle validation ────────────────────────────────────────────────────
    if args.oracle:
        print()
        print(f"Loading oracle: {args.oracle}")
        oracle = load_oracle(args.oracle)
        print(f"Oracle loaded: {len(oracle)} haplotypes "
              f"({oracle.dropna(subset=['F_eff']).shape[0]} with F_eff data)")
        print()

        print("Generating oracle validation figures...")
        fig_oracle_step1_correlation(df, oracle, resistance_kps, out_dir)
        fig_oracle_feff_landscape(oracle, resistance_kps, candidates, out_dir)
        fig_clinical_corecruitment(df, oracle, resistance_kps, out_dir)
        fig_lethality_avoidance(df, oracle, resistance_kps, out_dir)
        fig_oracle_validation_panel(
            df, oracle, resistance_kps, freq_pivot, candidates, out_dir)
        print()

        print_oracle_summary(df, oracle, resistance_kps, candidates)
        print()
        print("Oracle validation figures written:")
        for fn in ['oracle_step1_corr.png', 'oracle_feff_landscape.png',
                   'oracle_kp99_corecruitment.png', 'oracle_lethality_avoidance.png',
                   'oracle_validation_panel.png']:
            print(f"  {os.path.join(out_dir, fn)}")

    # ── ESM2 structural analysis ─────────────────────────────────────────────
    run_esm = bool(args.wt or args.esm_cache)
    if run_esm:
        import torch
        if args.esm_device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            device = args.esm_device
        print(f"ESM2 device: {device}"
              + (f"  ({torch.cuda.get_device_name(device)})"
                 if device.startswith('cuda') else ''))

        cache_path = (args.esm_cache
                      or os.path.join(out_dir, 'esm_results.npz'))

        print()
        if os.path.exists(cache_path):
            print(f"Loading ESM2 cache: {cache_path}")
            esm_results = _esm_load_cache(cache_path)
            missing = [(n, _esm_apply_mutations(
                            open(args.wt).read().split('\n', 1)[1].replace('\n','')
                            if args.wt else _WT_SEQ_BUILTIN, muts))
                       for n, muts in _ESM_HAPLOTYPES if n not in esm_results]
            if missing:
                print(f"  {len(missing)} sequences missing from cache — running inference...")
                model, alphabet = _esm_load_model(args.esm_model, device)
                new_res = _esm_run_inference(model, alphabet, missing, device)
                esm_results.update(new_res)
                _esm_save_cache(esm_results, cache_path)
        else:
            if not args.wt:
                print("ESM2 analysis skipped: --wt required when no cache exists.")
                run_esm = False

        if run_esm and 'esm_results' not in dir():
            # No cache — full inference
            wt_seq = ''
            with open(args.wt) as f:
                for line in f:
                    if not line.startswith('>'):
                        wt_seq += line.strip()
            print(f"Running ESM2 inference on {device} "
                  f"({len(_ESM_HAPLOTYPES)} sequences)...")
            sequences = []
            for name, muts in _ESM_HAPLOTYPES:
                try:
                    sequences.append((name, _esm_apply_mutations(wt_seq, muts)))
                except ValueError as e:
                    print(f"  WARNING: skipping {name}: {e}")
            model, alphabet = _esm_load_model(args.esm_model, device)
            esm_results = _esm_run_inference(model, alphabet, sequences, device)
            _esm_save_cache(esm_results, cache_path)
            print(f"Cache saved → {cache_path}")

    if run_esm and 'esm_results' in dir():
        res_pos   = sorted(resistance_kps)
        ridge_pos = [int(p) for p in candidates['position_shifted']] \
                    if not candidates.empty else []

        print()
        print("Generating ESM2 figures...")
        fig_esm_contact_deltas(esm_results, res_pos, ridge_pos, out_dir)
        fig_esm_embedding_perturbation(esm_results, res_pos, ridge_pos, out_dir)
        fig_esm_perturbation_profile(esm_results, res_pos, ridge_pos, out_dir)
        fig_esm_ridge_coupling(esm_results, res_pos, ridge_pos, out_dir)
        fig_esm_ridge_contact_partners(esm_results, res_pos, ridge_pos, out_dir)
        fig_esm_panel(esm_results, res_pos, ridge_pos, out_dir)
        print()
        print_esm_report(esm_results, res_pos, ridge_pos)
        print()
        print("ESM2 figures written:")
        for fn in ['esm_contact_deltas.png', 'esm_embedding_perturbation.png',
                   'esm_perturbation_profile.png', 'esm_ridge_coupling.png',
                   'esm_ridge_contact_partners.png', 'esm_panel.png']:
            print(f"  {os.path.join(out_dir, fn)}")

        # Free GPU memory and release CUDA threads so the process exits cleanly
        try:
            import torch
            if 'model' in dir():
                del model
            torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == '__main__':
    import sys
    main()
    sys.exit(0)
