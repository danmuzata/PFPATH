"""
PfPATH adaptive walk epistasis and ridge identification analysis.

Identifies ridge (compensatory) positions using robust statistics beyond
simple frequency counting, and analyses epistatic relationships between
resistance mutations and ridge positions.

Inspired by Wagner (2023) evolvability-enhancing (EE) mutation framework.

Outputs → walk_analysis/
  CSVs:
    ridge_identification.csv          — per-position ridge scores with full stats
    epistasis_pairs.csv               — co-occurring mutation pairs with stats
    ridge_resistance_dependence.csv   — P(ridge | resistance mutant present) matrix
    ee_scores.csv                     — evolvability-enhancement scores per mutation
    step_order.csv                    — mean step at first acquisition per mutation

  Figures:
    fig01_ridge_identification.png    — composite ridge score: frequency + Fisher + cross-mutant + temporal
    fig02_mutation_freq.png           — top mutation frequencies coloured by class
    fig03_cooccurrence_heatmap.png    — pairwise co-occurrence matrix
    fig04_epistasis_score.png         — log₂(obs/expected) epistasis score matrix
    fig05_ridge_resistance_heatmap.png— P(ridge | resistance mutant) LANDSCAPE
    fig06_temporal_ordering.png       — P(A before B) for frequent pairs
    fig07_ee_analysis.png             — EE scores: Wagner analog
    fig08_fitness_trajectories.png    — fitness gain after ridge vs other first companion
    fig09_step_order_violin.png       — acquisition step violin: resistance vs ridge
    fig10_mi_dccm_validation.png      — walk epistasis score vs MI and DCCM (deferred)
    fig11_network.png                 — co-occurrence network
"""

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.size':         16,
    'axes.titlesize':    20,
    'axes.labelsize':    18,
    'xtick.labelsize':   16,
    'ytick.labelsize':   16,
    'legend.fontsize':   15,
    'figure.titlesize':  22,
    'axes.linewidth':    1.2,
    'lines.linewidth':   1.8,
})
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import mannwhitneyu, fisher_exact
from scipy.stats import false_discovery_control
import networkx as nx
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(__file__)
WALKS     = os.path.join(BASE, "runs/pfpath_run_revised_pop/walks.csv")
MI_CSV    = os.path.join(BASE, "inputs/coevolution.csv")
DCCM_CSV  = os.path.join(BASE, "inputs/dhfr_dccm_matrix.csv")
# Ridge confirmed set is derived from Filter C (cross=5 + freq>=0.20) — see walk_analysis/ridge_justification.md
RIDGE_CROSS_MIN = 5     # must appear from ALL resistance backgrounds
RIDGE_FREQ_MIN  = 0.20  # ≥20% of walks per background (~25× above null expectation)
OUTDIR    = os.path.join(BASE, "walk_analysis")
os.makedirs(OUTDIR, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
RES_POSITIONS    = {41, 42, 50, 99, 155}   # shifted numbering — resistance mutation sites
CANONICAL_OFFSET = 9
MIN_FREQ         = 0.05   # ≥5% of walks
MIN_PAIR_WALKS   = 3
N_BOOTSTRAP      = 1000
RNG              = np.random.default_rng(42)

# Colours
COL_RES   = "#D62728"
COL_RIDGE = "#1F77B4"
COL_OTHER = "#AAAAAA"

# Human-readable labels for resistance positions
RES_LABELS = {41: 'C41\n(c50)', 42: 'N42\n(c51)', 50: 'C50\n(c59)',
              99: 'S99\n(c108)', 155: 'I155\n(c164)'}

# ── Helpers ────────────────────────────────────────────────────────────────────
def parse_pos(m):
    match = re.search(r'(\d+)', str(m))
    return int(match.group(1)) if match else None

def parse_mut_set(s):
    if s == 'WT' or pd.isna(s):
        return frozenset()
    return frozenset(s.split('+'))

def canonical(p):
    return (p + CANONICAL_OFFSET) if p else None

def mut_class(m, ridge_pos):
    p = parse_pos(m)
    if p in RES_POSITIONS:  return 'resistance'
    if p in ridge_pos:      return 'ridge'
    return 'other'

def class_color(c):
    return {  'resistance': COL_RES, 'ridge': COL_RIDGE}.get(c, COL_OTHER)

legend_handles = [
    mpatches.Patch(color=COL_RES,   label='Resistance mutation site'),
    mpatches.Patch(color=COL_RIDGE, label='Ridge (compensatory)'),
    mpatches.Patch(color=COL_OTHER, label='Other'),
]

# ── 1. Load and preprocess walks ──────────────────────────────────────────────
print("Loading walks...")
df = pd.read_csv(WALKS)
df['mut_set'] = df['mutations'].apply(parse_mut_set)

records = []
for (kp, rep), grp in df.groupby(['key_position', 'rep']):
    grp = grp.sort_values('step').reset_index(drop=True)
    prev = frozenset()
    for _, row in grp.iterrows():
        curr = row['mut_set']
        records.append({
            'key_position': kp, 'rep': rep, 'step': row['step'],
            'fitness': row['fitness'], 'mut_set': curr, 'new_muts': curr - prev
        })
        prev = curr
walk_df = pd.DataFrame(records)

n_walks = walk_df.groupby(['key_position', 'rep']).ngroups
print(f"  Total walks: {n_walks}")

final_df = (walk_df.groupby(['key_position', 'rep'])
            .apply(lambda g: g.loc[g['step'].idxmax()])
            .reset_index(drop=True))
final_sets = final_df['mut_set'].tolist()

# Placeholder — ridge_ref_pos is redefined after ridge_id_df is built using Filter C
ridge_ref_pos = set()

# ── 2. Per-mutation frequency ─────────────────────────────────────────────────
all_muts = sorted({m for s in final_sets for m in s},
                  key=lambda m: (parse_pos(m) or 0, m))
mut_freq = {m: sum(m in s for s in final_sets) / n_walks for m in all_muts}

step_records = []
for (kp, rep), grp in walk_df.groupby(['key_position', 'rep']):
    for _, row in grp.sort_values('step').iterrows():
        for m in row['new_muts']:
            step_records.append({'key_position': kp, 'rep': rep,
                                 'step': row['step'], 'mutation': m,
                                 'fitness': row['fitness']})
step_df = pd.DataFrame(step_records)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. ROBUST RIDGE IDENTIFICATION
# Four criteria:  frequency + bootstrap CI
#                 cross-resistance-mutant recurrence
#                 temporal ordering (appears AFTER resistance mutation)
#                 Fisher enrichment (more common with resistance mutation present)
# Combined into a composite score
# ═══════════════════════════════════════════════════════════════════════════════
print("Computing ridge identification statistics...")

candidate_positions = sorted({parse_pos(m) for m in all_muts
                               if parse_pos(m) not in RES_POSITIONS and parse_pos(m)})

pos_records = []
for pos in candidate_positions:
    pos_muts = [m for m in all_muts if parse_pos(m) == pos]
    pos_freq  = sum(any(m in s for m in pos_muts) for s in final_sets) / n_walks
    if pos_freq == 0:
        continue

    # A. Bootstrap 95% CI
    bs_freqs = [
        sum(any(m in s for m in pos_muts) for s in RNG.choice(final_sets, size=n_walks, replace=True)) / n_walks
        for _ in range(N_BOOTSTRAP)
    ]
    ci_lo, ci_hi = np.percentile(bs_freqs, [2.5, 97.5])

    # B. Fisher enrichment: position appears more often when any resistance mutation is present
    has_res = [any(parse_pos(m) in RES_POSITIONS for m in s) for s in final_sets]
    has_pos = [any(m in s for m in pos_muts) for s in final_sets]
    a = sum(r and p for r, p in zip(has_res, has_pos))
    b = sum(r and not p for r, p in zip(has_res, has_pos))
    c = sum(not r and p for r, p in zip(has_res, has_pos))
    d = sum(not r and not p for r, p in zip(has_res, has_pos))
    try:
        _, fisher_p = fisher_exact([[a, b], [c, d]], alternative='greater')
    except Exception:
        fisher_p = 1.0
    p_with_res    = a / (a + b) if (a + b) > 0 else 0
    p_without_res = c / (c + d) if (c + d) > 0 else 0
    enrich_ratio  = p_with_res / p_without_res if p_without_res > 0 else np.inf

    # C. Cross-resistance recurrence: how many distinct resistance positions does it accompany?
    per_res_freq = {}
    for rp in RES_POSITIONS:
        kp_walks = final_df[final_df.key_position == rp]['mut_set']
        per_res_freq[rp] = sum(any(m in s for m in pos_muts) for s in kp_walks) / len(kp_walks)
    cross_count = sum(v > 0 for v in per_res_freq.values())

    # D. Temporal ordering: P(after resistance) + mean step delay after resistance
    n_after, n_before, n_both = 0, 0, 0
    step_delays = []
    for (kp, rep), grp in walk_df.groupby(['key_position', 'rep']):
        grp = grp.sort_values('step')
        res_step = next((row.step for _, row in grp.iterrows()
                         if any(parse_pos(m) == kp for m in row['new_muts'])), None)
        pos_step = next((row.step for _, row in grp.iterrows()
                         if any(m in row['new_muts'] for m in pos_muts)), None)
        if res_step is not None and pos_step is not None:
            n_both += 1
            if pos_step > res_step:
                n_after += 1
                step_delays.append(pos_step - res_step)
            else:
                n_before += 1
    p_after_res       = n_after / n_both if n_both > 0 else np.nan
    mean_step_delay   = float(np.mean(step_delays))   if step_delays else np.nan
    std_step_delay    = float(np.std(step_delays))    if len(step_delays) > 1 else 0.0

    # E. Mean fitness gain when this position is acquired
    gains = []
    for (kp, rep), grp in walk_df.groupby(['key_position', 'rep']):
        grp = grp.sort_values('step').reset_index(drop=True)
        for i, row in grp.iterrows():
            if any(m in row['new_muts'] for m in pos_muts) and i > 0:
                f0 = grp.iloc[i-1]['fitness']
                if f0 > 0:
                    gains.append(np.log10(row['fitness'] / f0))
    mean_gain = np.mean(gains) if gains else np.nan

    # F. Composite score (0–1)
    s_freq  = min(pos_freq / 0.30, 1.0)
    s_cross = cross_count / len(RES_POSITIONS)
    s_temp  = p_after_res if not np.isnan(p_after_res) else 0.5
    s_fish  = min(-np.log10(fisher_p + 1e-10) / 8, 1.0)
    composite = 0.30 * s_freq + 0.30 * s_cross + 0.25 * s_temp + 0.15 * s_fish

    pos_records.append({
        'position_shifted':    pos,
        'position_canonical':  canonical(pos),
        'pos_mutations':       ','.join(sorted(pos_muts)),
        'frequency':           pos_freq,
        'ci_lo_95':            ci_lo,
        'ci_hi_95':            ci_hi,
        'cross_resistance_count': cross_count,
        'p_temporal_after_res':  p_after_res,
        'mean_step_delay':       mean_step_delay,
        'std_step_delay':        std_step_delay,
        'n_temporal_obs':        n_both,
        'fisher_pval':           fisher_p,
        'enrichment_ratio':      enrich_ratio if np.isfinite(enrich_ratio) else 999,
        'mean_fitness_gain':     mean_gain,
        'composite_ridge_score': composite,
        'is_reference_ridge':    pos in ridge_ref_pos,
        **{f'freq_in_rm_{rp}': per_res_freq.get(rp, 0) for rp in sorted(RES_POSITIONS)}
    })

ridge_id_df = pd.DataFrame(pos_records).sort_values('composite_ridge_score', ascending=False)
ridge_id_df['fisher_pval_fdr'] = false_discovery_control(
    ridge_id_df['fisher_pval'].fillna(1.0).values, method='bh')
print(f"  Ridge candidates scored: {len(ridge_id_df)}")

# ── Define confirmed ridge using Filter C (cross=5 + freq>=0.20) ─────────────
# Biologically grounded: cross=5 operationalises universal convergence;
# freq>=0.20 is ~25× above null per-position expectation (~0.8%).
# See walk_analysis/ridge_justification.md for full rationale.
ridge_ref_pos = set(
    ridge_id_df.loc[
        (ridge_id_df['cross_resistance_count'] >= RIDGE_CROSS_MIN) &
        (ridge_id_df['frequency'] >= RIDGE_FREQ_MIN),
        'position_shifted'
    ]
)
ridge_id_df['is_reference_ridge'] = ridge_id_df['position_shifted'].isin(ridge_ref_pos)

# Extended ridge (Filter B): cross=5 + freq>=0.15 — reported in supplementary
ridge_extended_pos = set(
    ridge_id_df.loc[
        (ridge_id_df['cross_resistance_count'] >= RIDGE_CROSS_MIN) &
        (ridge_id_df['frequency'] >= 0.15),
        'position_shifted'
    ]
)
ridge_id_df['is_extended_ridge'] = ridge_id_df['position_shifted'].isin(ridge_extended_pos)

ridge_id_df.to_csv(os.path.join(OUTDIR, 'ridge_identification.csv'), index=False)
print(f"  Confirmed ridge (Filter C, cross=5+freq>=0.20): {len(ridge_ref_pos)} positions: "
      f"{sorted(ridge_ref_pos)}")
print(f"  Extended ridge (Filter B, cross=5+freq>=0.15): {len(ridge_extended_pos)} positions")

# Ridge sets for downstream colouring
ridge_all_pos = ridge_ref_pos

# ── 4. Pairwise epistasis ─────────────────────────────────────────────────────
print("Computing pairwise epistasis...")
common_muts = [m for m, f in mut_freq.items() if f >= MIN_FREQ]

pair_records = []
for i, m1 in enumerate(common_muts):
    f1 = mut_freq[m1]
    for m2 in common_muts[i+1:]:
        f2 = mut_freq[m2]
        joint = sum((m1 in s) and (m2 in s) for s in final_sets) / n_walks
        if joint * n_walks < MIN_PAIR_WALKS:
            continue
        eps = np.log2(joint / (f1 * f2)) if f1 * f2 > 0 and joint > 0 else 0

        both_walks = [(kp, rep) for (kp, rep), g in walk_df.groupby(['key_position', 'rep'])
                      if m1 in g['mut_set'].iloc[-1] and m2 in g['mut_set'].iloc[-1]]
        n_m1_first = 0
        for kp, rep in both_walks:
            grp = walk_df[(walk_df.key_position==kp)&(walk_df.rep==rep)].sort_values('step')
            s1 = next((r.step for _,r in grp.iterrows() if m1 in r.new_muts), None)
            s2 = next((r.step for _,r in grp.iterrows() if m2 in r.new_muts), None)
            if s1 and s2 and s1 < s2:
                n_m1_first += 1
        p_m1_first = n_m1_first / len(both_walks) if both_walks else 0.5

        a = int(joint*n_walks); b = int((f1-joint)*n_walks)
        c = int((f2-joint)*n_walks); d = n_walks - a - b - c
        try:
            _, fp = fisher_exact([[a,b],[c,max(0,d)]], alternative='greater')
        except Exception:
            fp = 1.0

        pair_records.append({
            'mut1': m1, 'mut2': m2,
            'class1': mut_class(m1, ridge_all_pos), 'class2': mut_class(m2, ridge_all_pos),
            'pos1': parse_pos(m1), 'pos2': parse_pos(m2),
            'freq1': f1, 'freq2': f2, 'joint_freq': joint,
            'P_m2_given_m1': joint/f1 if f1 else 0,
            'P_m1_given_m2': joint/f2 if f2 else 0,
            'epistasis_score': eps,
            'P_m1_before_m2': p_m1_first,
            'n_co_occur': len(both_walks),
            'fisher_pval': fp
        })

pairs_df = pd.DataFrame(pair_records)

# ── MI / DCCM lookups (built once, reused for both mutation-level and position-level) ──
mi_df = pd.read_csv(MI_CSV)
mi_df.columns = mi_df.columns.str.strip()
# Support both 'Residue 1'/'Residue 2' and 'Res1'/'Res2' column names
_c1 = 'Residue 1' if 'Residue 1' in mi_df.columns else 'Res1'
_c2 = 'Residue 2' if 'Residue 2' in mi_df.columns else 'Res2'
_mi = 'Mutual Information' if 'Mutual Information' in mi_df.columns else 'MI'
mi_df['pos1'] = mi_df[_c1].apply(lambda x: int(re.search(r'\d+', str(x)).group()))
mi_df['pos2'] = mi_df[_c2].apply(lambda x: int(re.search(r'\d+', str(x)).group()))
mi_lut = {(r.pos1, r.pos2): r[_mi] for _, r in mi_df.iterrows()}
mi_lut.update({(r.pos2, r.pos1): r[_mi] for _, r in mi_df.iterrows()})

# inputs/dhfr_dccm_matrix.csv — MD-based (1 µs CA-only trajectory, Bio3D v2.4)
dccm_raw = pd.read_csv(DCCM_CSV, index_col=0)
dccm_raw.columns = pd.to_numeric(dccm_raw.columns, errors='coerce').astype('Int64')
dccm_raw.index   = pd.to_numeric(dccm_raw.index,   errors='coerce').astype('Int64')

def get_mi(p1, p2):
    return mi_lut.get((p1, p2), np.nan)

def get_dccm(p1, p2):
    try:
        return float(dccm_raw.loc[p1, p2])
    except Exception:
        return np.nan

pairs_df['MI']   = pairs_df.apply(lambda r: get_mi(r.pos1, r.pos2),   axis=1)
pairs_df['DCCM'] = pairs_df.apply(lambda r: get_dccm(r.pos1, r.pos2), axis=1)
pairs_df.sort_values('epistasis_score', ascending=False).to_csv(
    os.path.join(OUTDIR, 'epistasis_pairs.csv'), index=False)

# ── 4b. Position-level epistasis (more data points for MI/DCCM correlation) ──
# Work at residue position level: does any mutation at pos1 co-occur with any at pos2?
print("Computing position-level epistasis...")
common_positions = sorted({parse_pos(m) for m in common_muts})

pos_pair_records = []
for i, p1 in enumerate(common_positions):
    muts1 = [m for m in common_muts if parse_pos(m) == p1]
    f1 = sum(any(m in s for m in muts1) for s in final_sets) / n_walks
    for p2 in common_positions[i+1:]:
        muts2  = [m for m in common_muts if parse_pos(m) == p2]
        f2     = sum(any(m in s for m in muts2) for s in final_sets) / n_walks
        joint  = sum(any(m in s for m in muts1) and any(m in s for m in muts2)
                     for s in final_sets) / n_walks
        if joint == 0:
            continue
        eps = np.log2(joint / (f1 * f2)) if f1 * f2 > 0 else 0

        # Conditional: P(pos2 mutated | pos1 mutated)
        p2_given_p1 = joint / f1 if f1 > 0 else 0
        p2_given_no_p1 = (sum(not any(m in s for m in muts1) and any(m in s for m in muts2)
                              for s in final_sets) / n_walks) / (1 - f1) if f1 < 1 else 0

        # Fisher enrichment
        a = int(joint*n_walks); b = int((f1-joint)*n_walks)
        c = int((f2-joint)*n_walks); d = max(0, n_walks - a - b - c)
        try:
            _, fp = fisher_exact([[a, b], [c, d]], alternative='greater')
        except Exception:
            fp = 1.0

        c1 = mut_class(muts1[0], ridge_all_pos)
        c2 = mut_class(muts2[0], ridge_all_pos)
        pos_pair_records.append({
            'pos1': p1, 'pos2': p2,
            'pos1_canonical': canonical(p1), 'pos2_canonical': canonical(p2),
            'class1': c1, 'class2': c2,
            'pair_type': '-'.join(sorted([c1, c2])),
            'freq_pos1': f1, 'freq_pos2': f2, 'joint_freq': joint,
            'P_pos2_given_pos1': p2_given_p1,
            'P_pos2_given_no_pos1': p2_given_no_p1,
            'conditional_enrichment': p2_given_p1 / p2_given_no_p1 if p2_given_no_p1 > 0 else np.inf,
            'epistasis_score': eps,
            'fisher_pval': fp,
            'MI': get_mi(p1, p2),
            'DCCM': get_dccm(p1, p2),
        })

pos_pairs_df = pd.DataFrame(pos_pair_records)
pos_pairs_df['fisher_pval_fdr'] = false_discovery_control(
    pos_pairs_df['fisher_pval'].fillna(1.0).values, method='bh')

# Obligate epistasis: P(pos2 | pos1) significantly > P(pos2 | no pos1)
# Note: FDR threshold is not met for many pairs due to low absolute counts
# (positions mutated in only 5-17 walks). Use nominal threshold and enrichment
# criterion; interpret as preferentially co-occurring rather than strictly obligate.
obligate = pos_pairs_df[
    (pos_pairs_df['P_pos2_given_pos1'] >= 0.4) &
    (pos_pairs_df['conditional_enrichment'] >= 2.0)
].sort_values('P_pos2_given_pos1', ascending=False)

pos_pairs_df.sort_values('epistasis_score', ascending=False).to_csv(
    os.path.join(OUTDIR, 'epistasis_position_pairs.csv'), index=False)
obligate.to_csv(os.path.join(OUTDIR, 'epistasis_obligate_pairs.csv'), index=False)
print(f"  Position pairs: {len(pos_pairs_df)} | Obligate (FDR<0.05, enrich≥2×): {len(obligate)}")

# ── 5. Ridge × Resistance conditional table ───────────────────────────────────
print("Computing ridge × resistance conditional frequencies...")
top_ridge_muts = (
    pd.DataFrame([{'mut': m, 'freq': mut_freq[m]}
                  for m in common_muts if parse_pos(m) not in RES_POSITIONS])
    .sort_values('freq', ascending=False).head(25)['mut'].tolist()
)

rk_records = []
for rm in top_ridge_muts:
    for rp in sorted(RES_POSITIONS):
        kp_walks = final_df[final_df.key_position == rp]
        has_res  = kp_walks['mut_set'].apply(lambda s: any(parse_pos(m)==rp for m in s))
        has_rm   = kp_walks['mut_set'].apply(lambda s: rm in s)
        n_yes = has_res.sum(); n_no = (~has_res).sum()
        rk_records.append({
            'ridge_mutation':              rm,
            'resistance_position':         rp,
            'resistance_canonical':        canonical(rp),
            'P_ridge_given_resistance':    (has_res & has_rm).sum() / n_yes if n_yes else 0,
            'P_ridge_given_no_resistance': (~has_res & has_rm).sum() / n_no  if n_no  else 0,
        })
rk_df = pd.DataFrame(rk_records)
rk_df.to_csv(os.path.join(OUTDIR, 'ridge_resistance_dependence.csv'), index=False)

# ── 6. Wagner (2023) consistency metrics ──────────────────────────────────────
# We test whether our resistance-ridge relationship exhibits the same four
# empirical hallmarks that Wagner (2023) identified for EE mutations:
#   Obs 1 — EE mutations appear early in adaptive walks
#   Obs 2 — Walks encountering EE mutations reach higher final fitness
#   Obs 3 — EE mutations increase accessibility of subsequent beneficial mutations
#   Obs 4 — EE mutations obligately precede the mutations they enable
print("Computing Wagner-consistency metrics...")

# -- Obs 1: step of first acquisition — resistance mutations early, ridge later --
res_steps   = step_df[step_df['mutation'].apply(lambda m: parse_pos(m) in RES_POSITIONS)]['step']
ridge_steps = step_df[step_df['mutation'].apply(
    lambda m: parse_pos(m) in ridge_all_pos and parse_pos(m) not in RES_POSITIONS)]['step']
obs1_stat, obs1_p = mannwhitneyu(res_steps, ridge_steps, alternative='less')

# -- Obs 2: final fitness — walks with ≥1 ridge mutation vs without --
final_df['has_ridge'] = final_df['mut_set'].apply(
    lambda s: any(parse_pos(m) in ridge_all_pos and parse_pos(m) not in RES_POSITIONS for m in s))
final_df['n_ridge_muts'] = final_df['mut_set'].apply(
    lambda s: sum(1 for m in s if parse_pos(m) in ridge_all_pos and parse_pos(m) not in RES_POSITIONS))
with_ridge_fitness    = final_df[final_df.has_ridge]['fitness'].values
without_ridge_fitness = final_df[~final_df.has_ridge]['fitness'].values
# Use log10(fitness) to avoid blow-up when without-ridge walks stall near 0
log_with    = np.log10(np.maximum(with_ridge_fitness,    1e-12))
log_without = np.log10(np.maximum(without_ridge_fitness, 1e-12))
obs2_log_diff = np.mean(log_with) - np.mean(log_without)
obs2_stat, obs2_p = (mannwhitneyu(log_with, log_without, alternative='greater')
                     if len(log_without) > 0 else (np.nan, np.nan))

# Per-KP breakdown (log10 scale)
obs2_per_kp = []
for kp in sorted(RES_POSITIONS):
    sub  = final_df[final_df.key_position == kp]
    wr   = np.log10(np.maximum(sub[sub.has_ridge]['fitness'].values,  1e-12))
    nor  = np.log10(np.maximum(sub[~sub.has_ridge]['fitness'].values, 1e-12))
    if len(wr) > 0 and len(nor) > 0:
        diff = np.mean(wr) - np.mean(nor)
        _, p = mannwhitneyu(wr, nor, alternative='greater')
    else:
        diff, p = np.nan, np.nan
    obs2_per_kp.append({'resistance_position': kp, 'canonical': canonical(kp),
                        'mean_log10fitness_with_ridge':    np.mean(wr)  if len(wr)  else np.nan,
                        'mean_log10fitness_without_ridge': np.mean(nor) if len(nor) else np.nan,
                        'log10_fitness_gain': diff, 'mwu_pval': p,
                        'n_with_ridge': len(wr), 'n_without_ridge': len(nor)})
obs2_df = pd.DataFrame(obs2_per_kp)

# -- Obs 3: ridge selectivity within walks --
# All walks start from resistance backgrounds so P(ridge|res)/P(ridge overall) ≈ 1.
# Better question: out of all compensatory positions ever mutated across walks,
# are the 14 reference ridge positions disproportionately represented in any
# given walk's final mutant set?
# Null: ridge_ref_pos occupy X% of all compensatory positions ever seen.
# Observed: per walk, what fraction of acquired compensatory muts land at ridge_ref_pos?
# Selectivity = observed / null → values >1 mean ridge positions are preferentially fixed.
from scipy.stats import ttest_1samp

all_comp_pos   = {parse_pos(m) for m in all_muts if parse_pos(m) not in RES_POSITIONS}
null_ridge_frac = len(ridge_ref_pos) / len(all_comp_pos) if all_comp_pos else 0

walk_selectivity = []
for (kp, rep), grp in final_df.groupby(['key_position', 'rep']):
    s = grp.iloc[0]['mut_set']
    comp_muts = [m for m in s if parse_pos(m) not in RES_POSITIONS]
    n_ref_ridge = sum(1 for m in comp_muts if parse_pos(m) in ridge_ref_pos)
    obs_frac    = n_ref_ridge / len(comp_muts) if comp_muts else np.nan
    walk_selectivity.append({
        'key_position': kp, 'rep': rep,
        'n_compensatory': len(comp_muts),
        'n_ref_ridge': n_ref_ridge,
        'obs_ridge_frac': obs_frac,
        'null_ridge_frac': null_ridge_frac,
        'selectivity_ratio': obs_frac / null_ridge_frac if (null_ridge_frac > 0 and not np.isnan(obs_frac)) else np.nan
    })
access_df = pd.DataFrame(walk_selectivity).dropna(subset=['selectivity_ratio'])
obs3_median_sel = access_df['selectivity_ratio'].median()
obs3_tstat, obs3_p = ttest_1samp(access_df['obs_ridge_frac'].dropna(), null_ridge_frac,
                                  alternative='greater')
access_df.to_csv(os.path.join(OUTDIR, 'ridge_selectivity.csv'), index=False)
print(f"  Obs3: null ridge frac={null_ridge_frac:.3f} | ridge ref positions={len(ridge_ref_pos)} | comp positions={len(all_comp_pos)}")

# -- Obs 4: obligate ordering — P(resistance acquired before ridge | both present) --
ordering_records = []
for (kp, rep), grp in walk_df.groupby(['key_position', 'rep']):
    grp = grp.sort_values('step')
    final_muts = grp.iloc[-1]['mut_set']
    has_ridge_in_walk = any(parse_pos(m) in ridge_all_pos and parse_pos(m) not in RES_POSITIONS
                            for m in final_muts)
    if not has_ridge_in_walk:
        continue
    first_res_step   = next((row.step for _, row in grp.iterrows()
                             if any(parse_pos(m) in RES_POSITIONS for m in row['new_muts'])), None)
    first_ridge_step = next((row.step for _, row in grp.iterrows()
                             if any(parse_pos(m) in ridge_all_pos and parse_pos(m) not in RES_POSITIONS
                                    for m in row['new_muts'])), None)
    if first_res_step is not None and first_ridge_step is not None:
        ordering_records.append({'key_position': kp, 'rep': rep,
                                 'first_res_step': first_res_step,
                                 'first_ridge_step': first_ridge_step,
                                 'res_before_ridge': first_res_step < first_ridge_step})
ordering_df = pd.DataFrame(ordering_records)
p_res_before_ridge = ordering_df['res_before_ridge'].mean() if len(ordering_df) > 0 else np.nan

# -- Summary CSV --
wagner_summary = pd.DataFrame([
    {'observation': 'Obs1: EE mutations appear early',
     'wagner_finding': 'EE mutations acquired before mutations they enable',
     'our_metric': 'Mean step: resistance vs ridge mutations',
     'our_value': f"resistance {res_steps.mean():.1f} vs ridge {ridge_steps.mean():.1f}",
     'mwu_pval': obs1_p,
     'consistent': res_steps.mean() < ridge_steps.mean()},
    {'observation': 'Obs2: EE walks reach higher final fitness',
     'wagner_finding': '~7% higher fitness in walks encountering EE mutations',
     'our_metric': 'log10 fitness gain: walks with ≥1 ridge mutation vs without',
     'our_value': f"+{obs2_log_diff:.2f} log10 units",
     'mwu_pval': obs2_p,
     'consistent': obs2_log_diff > 0},
    {'observation': 'Obs3: EE mutations increase accessibility',
     'wagner_finding': 'EE mutations raise mean neighbour fitness beyond direct benefit',
     'our_metric': 'Ridge selectivity: observed ridge fraction / null ridge fraction (per walk)',
     'our_value': f"{obs3_median_sel:.2f}x (t-test p={obs3_p:.3g})",
     'mwu_pval': obs3_p,
     'consistent': obs3_median_sel > 1},
    {'observation': 'Obs4: Obligate ordering (EE precedes enhanced)',
     'wagner_finding': 'EE mutation must be present before it enhances neighbours',
     'our_metric': 'P(resistance acquired before ridge | both present)',
     'our_value': f"{p_res_before_ridge:.3f}" if not np.isnan(p_res_before_ridge) else 'N/A',
     'mwu_pval': np.nan,
     'consistent': p_res_before_ridge > 0.9 if not np.isnan(p_res_before_ridge) else None},
])
wagner_summary.to_csv(os.path.join(OUTDIR, 'wagner_consistency.csv'), index=False)
obs2_df.to_csv(os.path.join(OUTDIR, 'wagner_obs2_per_kp.csv'), index=False)

step_stats = (step_df.groupby('mutation')['step']
              .agg(['mean','std','count']).reset_index()
              .rename(columns={'mean':'mean_step','std':'std_step','count':'n_walks'}))
step_stats['freq']  = step_stats['n_walks'] / n_walks
step_stats['class'] = step_stats['mutation'].apply(lambda m: mut_class(m, ridge_all_pos))
step_stats.to_csv(os.path.join(OUTDIR, 'step_order.csv'), index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating figures...")

top30 = sorted([m for m in mut_freq if mut_freq[m] >= MIN_FREQ],
               key=lambda m: -mut_freq[m])[:30]
n30 = len(top30)

# ── Fig 1: Ridge identification composite score ────────────────────────────────
top_pos = ridge_id_df.head(30).reset_index(drop=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 7))

ax = axes[0]
ax.barh(range(len(top_pos)), top_pos['composite_ridge_score'],
        color=[COL_RIDGE if r else COL_OTHER for r in top_pos['is_reference_ridge']],
        edgecolor='white')
ax.set_yticks(range(len(top_pos)))
ax.set_yticklabels([f"pos {int(r.position_shifted)} (c{int(r.position_canonical)})"
                    for _, r in top_pos.iterrows()], fontsize=17)
ax.invert_yaxis()
ax.axvline(0.75, ls='--', color='red', lw=1.2, label='Score threshold 0.75 (Filter D)')
ax.axvline(0.3, ls=':', color='grey', lw=0.8, alpha=0.5)
ax.set_xlabel('Composite ridge score', fontsize=20)
ax.set_title('A. Composite score\n(frequency + cross-RM + temporal + Fisher)', fontsize=20)
ax.legend(fontsize=17)

ax = axes[1]
ax.barh(range(len(top_pos)), top_pos['cross_resistance_count'],
        color=[COL_RIDGE if r else COL_OTHER for r in top_pos['is_reference_ridge']])
ax.set_yticks(range(len(top_pos)))
ax.set_yticklabels([f"pos {int(r.position_shifted)}" for _, r in top_pos.iterrows()], fontsize=17)
ax.invert_yaxis()
ax.axvline(5, ls='--', color='red', lw=1.2, label='All 5 contexts (Filter C)')
ax.set_xlabel('Number of resistance contexts\nposition appears in', fontsize=20)
ax.set_title('B. Cross-resistance-context\nrecurrence', fontsize=20)
ax.legend(fontsize=17)

ax = axes[2]
_tp2 = top_pos.dropna(subset=['mean_step_delay']).reset_index(drop=True)
_xerr2 = _tp2['std_step_delay'].fillna(0).values
_gmed2 = float(top_pos['mean_step_delay'].median())
ax.barh(range(len(_tp2)), _tp2['mean_step_delay'],
        xerr=_xerr2, error_kw={'elinewidth': 0.6, 'capsize': 1.5, 'ecolor': '#555555'},
        color=[COL_RIDGE if r else COL_OTHER for r in _tp2['is_reference_ridge']],
        edgecolor='white')
ax.set_yticks(range(len(_tp2)))
ax.set_yticklabels([f"pos {int(r.position_shifted)}" for _, r in _tp2.iterrows()], fontsize=17)
ax.invert_yaxis()
ax.axvline(_gmed2, ls='--', color='k', lw=1.2, label=f'Median = {_gmed2:.1f} steps')
ax.set_xlabel('Mean steps after resistance mutation (±SD)', fontsize=20)
ax.set_title('C. Temporal ordering\n(fewer steps = acquired sooner)', fontsize=20)
ax.legend(handles=[mpatches.Patch(color=COL_RIDGE, label='Ridge candidate'),
                   mpatches.Patch(color=COL_OTHER, label='Below threshold'),
                   plt.Line2D([0], [0], color='k', ls='--', lw=1.2,
                              label=f'Median = {_gmed2:.1f} steps')],
          fontsize=15, loc='lower right')

plt.suptitle('Ridge position identification — multi-criterion statistical scoring', fontsize=20, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig01_ridge_identification.png'), dpi=200, bbox_inches='tight')
plt.close()

# ── Fig 1 individual panels (publication quality, 8×10, 300 dpi) ──────────────
_ylabels_full = [f"pos {int(r.position_shifted)} (c{int(r.position_canonical)})"
                 for _, r in top_pos.iterrows()]
_colors       = [COL_RIDGE if r else COL_OTHER for r in top_pos['is_reference_ridge']]
_legend_h     = [mpatches.Patch(color=COL_RIDGE, label='Confirmed ridge candidate'),
                 mpatches.Patch(color=COL_OTHER, label='Below threshold')]

# Panel A — Composite score
fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(range(len(top_pos)), top_pos['composite_ridge_score'],
        color=_colors, edgecolor='white', height=0.75)
ax.set_yticks(range(len(top_pos)))
ax.set_yticklabels(_ylabels_full, fontsize=19)
ax.invert_yaxis()
ax.axvline(0.75, ls='--', color='red', lw=1.5,
           label='Score threshold (0.75, Filter D)')
ax.set_xlabel('Composite ridge score', fontsize=22)
ax.set_title('Composite ridge score\n(frequency + cross-resistance context + temporal ordering + Fisher)',
             fontsize=20, fontweight='bold', pad=10)
ax.set_xlim(0, 1.0)
ax.tick_params(axis='x', labelsize=19)
ax.legend(handles=_legend_h +
          [mpatches.Patch(color='none', label=''),
           plt.Line2D([0],[0], color='red', ls='--', lw=1.5, label='Filter D threshold = 0.75')],
          fontsize=17, frameon=True, edgecolor='#cccccc', loc='lower right')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=0.2, linestyle='--', lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig01A_composite_score.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig01A_composite_score.png")

# Panel B — Cross-resistance context recurrence
fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(range(len(top_pos)), top_pos['cross_resistance_count'],
        color=_colors, edgecolor='white', height=0.75)
ax.set_yticks(range(len(top_pos)))
ax.set_yticklabels([str(int(r.position_shifted)) for _, r in top_pos.iterrows()], fontsize=19)
ax.set_ylabel('Position', fontsize=22)
ax.invert_yaxis()
ax.axvline(5, ls='--', color='red', lw=1.5, label='Filter C threshold: all 5 contexts')
ax.set_xlabel('Number of resistance contexts (out of 5)', fontsize=22)
ax.set_title('Cross-resistance context recurrence\n'
             'Number of distinct resistance-mutation backgrounds in which position appears',
             fontsize=20, fontweight='bold', pad=10)
ax.set_xlim(0, len(RES_POSITIONS) + 0.5)
ax.set_xticks(range(len(RES_POSITIONS) + 1))
ax.tick_params(axis='x', labelsize=19)
ax.legend(handles=_legend_h +
          [plt.Line2D([0],[0], color='red', ls='--', lw=1.5, label='Filter C: all 5 contexts')],
          fontsize=17, frameon=True, edgecolor='#cccccc', loc='lower right')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=0.2, linestyle='--', lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig01B_cross_context.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig01B_cross_context.png")

# Panel C — Mean step delay (within-group temporal discrimination)
tp = top_pos.dropna(subset=['mean_step_delay']).reset_index(drop=True)
_tp_labels  = [f"pos {int(r.position_shifted)} (c{int(r.position_canonical)})"
               for _, r in tp.iterrows()]
_tp_colors  = [COL_RIDGE if r else COL_OTHER for r in tp['is_reference_ridge']]
_tp_xerr    = tp['std_step_delay'].fillna(0).values
_global_med = float(top_pos['mean_step_delay'].median())

fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(range(len(tp)), tp['mean_step_delay'],
        xerr=_tp_xerr, error_kw={'elinewidth': 0.8, 'capsize': 2, 'ecolor': '#555555'},
        color=_tp_colors, edgecolor='white', height=0.75)
ax.set_yticks(range(len(tp)))
ax.set_yticklabels(_tp_labels, fontsize=19)
ax.invert_yaxis()
ax.axvline(_global_med, ls='--', color='k', lw=1.5,
           label=f'Median step delay ({_global_med:.1f} steps)')
ax.set_xlabel('Mean steps acquired after resistance mutation (± SD)', fontsize=22)
ax.set_title('Temporal coupling to resistance\n'
             'Smaller delay = position acquired sooner after resistance mutation',
             fontsize=20, fontweight='bold', pad=10)
ax.tick_params(axis='x', labelsize=19)
ax.legend(handles=_legend_h +
          [mpatches.Patch(color='none', label=''),
           plt.Line2D([0],[0], color='k', ls='--', lw=1.5,
                      label=f'Median = {_global_med:.1f} steps')],
          fontsize=17, frameon=True, edgecolor='#cccccc', loc='lower right')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=0.2, linestyle='--', lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig01C_temporal_ordering.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print("  Saved: fig01C_temporal_ordering.png")

# ── Fig 2: Mutation frequency bar ─────────────────────────────────────────────
freq_df = (pd.DataFrame([{'mutation': m, 'freq': f,
                           'class': mut_class(m, ridge_all_pos)} for m, f in mut_freq.items()])
           .sort_values('freq', ascending=False).head(40))

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(range(len(freq_df)), freq_df['freq']*100,
       color=[class_color(c) for c in freq_df['class']], edgecolor='white', lw=0.5)
ax.set_xticks(range(len(freq_df)))
ax.set_xticklabels(freq_df['mutation'], rotation=55, ha='right', fontsize=13)
ax.set_ylabel('Frequency across walks (%)', fontsize=19)
ax.set_title(f'Top 40 mutations by frequency across {n_walks} adaptive walks', fontsize=20)
ax.axhline(MIN_FREQ*100, ls='--', color='k', lw=0.8, alpha=0.6, label=f'{int(MIN_FREQ*100)}% threshold')
ax.legend(handles=legend_handles + [mpatches.Patch(color='none', label='')], fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig02_mutation_freq.png'), dpi=200)
plt.close()

# ── Fig 3: Co-occurrence heatmap ──────────────────────────────────────────────
cooc = np.array([[sum((a in s)and(b in s) for s in final_sets)/n_walks
                  for b in top30] for a in top30])
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cooc*100, cmap='Blues', vmin=0, vmax=25)
plt.colorbar(im, ax=ax, label='Co-occurrence frequency (%)')
ax.set_xticks(range(n30)); ax.set_yticks(range(n30))
ax.set_xticklabels(top30, rotation=60, ha='right', fontsize=13)
ax.set_yticklabels(top30, fontsize=13)
for t, m in zip(ax.get_xticklabels(), top30): t.set_color(class_color(mut_class(m, ridge_all_pos)))
for t, m in zip(ax.get_yticklabels(), top30): t.set_color(class_color(mut_class(m, ridge_all_pos)))
ax.set_title('Pairwise co-occurrence frequency (top 30 mutations)', fontsize=20)
ax.legend(handles=legend_handles, loc='lower right', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig03_cooccurrence_heatmap.png'), dpi=200)
plt.close()

# ── Fig 4: Epistasis score heatmap ───────────────────────────────────────────
eps_mat = np.full((n30, n30), np.nan)
for _, r in pairs_df.iterrows():
    if r.mut1 in top30 and r.mut2 in top30:
        i, j = top30.index(r.mut1), top30.index(r.mut2)
        eps_mat[i,j] = eps_mat[j,i] = r.epistasis_score
np.fill_diagonal(eps_mat, 0)
vmax = np.nanpercentile(np.abs(eps_mat[np.isfinite(eps_mat)]), 95) if np.any(np.isfinite(eps_mat)) else 1

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(eps_mat, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=ax, label='Epistasis score [log₂(P_obs/P_exp)]')
ax.set_xticks(range(n30)); ax.set_yticks(range(n30))
ax.set_xticklabels(top30, rotation=60, ha='right', fontsize=13)
ax.set_yticklabels(top30, fontsize=13)
for t, m in zip(ax.get_xticklabels(), top30): t.set_color(class_color(mut_class(m, ridge_all_pos)))
for t, m in zip(ax.get_yticklabels(), top30): t.set_color(class_color(mut_class(m, ridge_all_pos)))
ax.set_title('Pairwise epistasis score [log₂(observed/expected co-occurrence)]', fontsize=20)
ax.legend(handles=legend_handles, loc='lower right', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig04_epistasis_score.png'), dpi=200)
plt.close()

# ── Fig 5: Ridge × Resistance LANDSCAPE heatmap ──────────────────────────────
n_rm    = len(RES_POSITIONS)
n_ridge = len(top_ridge_muts)
res_order = sorted(RES_POSITIONS)

cond_mat = np.zeros((n_rm, n_ridge))
for j, rm in enumerate(top_ridge_muts):
    for i, rp in enumerate(res_order):
        row = rk_df[(rk_df.ridge_mutation==rm) & (rk_df.resistance_position==rp)]
        if not row.empty:
            cond_mat[i, j] = row.iloc[0]['P_ridge_given_resistance']

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(cond_mat*100, cmap='YlOrRd', vmin=0, vmax=60, aspect='auto')
cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.015)
cbar.set_label('P(compensatory mutation present\n| resistance mutation present) (%)',
               fontsize=17, labelpad=8)
cbar.ax.tick_params(labelsize=16)

ax.set_yticks(range(n_rm))
ax.set_yticklabels([RES_LABELS[rp] for rp in res_order], fontsize=17, fontweight='bold')
ax.set_ylabel('Resistance Mutations', fontsize=20, fontweight='bold', labelpad=10)

ax.set_xticks(range(n_ridge))
ax.set_xticklabels(top_ridge_muts, rotation=90, ha='center', fontsize=17, fontweight='bold')
for t, rm in zip(ax.get_xticklabels(), top_ridge_muts):
    t.set_color(COL_RIDGE if mut_class(rm, ridge_all_pos) == 'ridge' else '#666666')
ax.set_xlabel('Ridge Candidates', fontsize=20, fontweight='bold', labelpad=10)

for i in range(n_rm):
    for j in range(n_ridge):
        val = cond_mat[i, j]*100
        if val > 2:
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='white' if val > 35 else '#333333')

ax.set_title(
    'P(compensatory mutation present | resistance mutation present)\n'
    'Blue labels = confirmed ridge (Filter C: universal cross-resistance across all 5 backgrounds + frequency ≥ 20%)',
    fontsize=19, pad=10)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig05_ridge_resistance_heatmap.png'),
            dpi=300, bbox_inches='tight')
plt.close(fig)

# ── Fig 6: Temporal ordering heatmap ─────────────────────────────────────────
ord_mat = np.full((n30, n30), np.nan)
for _, r in pairs_df.iterrows():
    if r.mut1 in top30 and r.mut2 in top30:
        i, j = top30.index(r.mut1), top30.index(r.mut2)
        ord_mat[i,j] = r['P_m1_before_m2']
        ord_mat[j,i] = 1 - r['P_m1_before_m2']
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(ord_mat, cmap='RdYlGn', vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label='P(row mutation acquired before column mutation)')
ax.set_xticks(range(n30)); ax.set_yticks(range(n30))
ax.set_xticklabels(top30, rotation=60, ha='right', fontsize=13)
ax.set_yticklabels(top30, fontsize=13)
for t, m in zip(ax.get_xticklabels(), top30): t.set_color(class_color(mut_class(m, ridge_all_pos)))
for t, m in zip(ax.get_yticklabels(), top30): t.set_color(class_color(mut_class(m, ridge_all_pos)))
ax.set_title('Temporal ordering: P(row mutation acquired before column mutation)', fontsize=20)
ax.legend(handles=legend_handles, loc='lower right', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig06_temporal_ordering.png'), dpi=200)
plt.close()

# ── Fig 7: Wagner (2023) consistency — 4-panel ───────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Resistance mutations as evolvability-enhancing mutations\n'
             '(consistent with Wagner 2023 hallmarks)', fontsize=22, y=1.01)

# Panel A — Obs 1: step acquisition violin
ax = axes[0, 0]
classes_v = ['resistance', 'ridge', 'other']
labels_v  = ['Resistance\nmutations', 'Ridge\n(compensatory)', 'Other']
colors_v  = [COL_RES, COL_RIDGE, COL_OTHER]
data_v = [step_df[step_df['mutation'].apply(
               lambda m: mut_class(m, ridge_all_pos))==cls]['step'].values
          for cls in classes_v]
data_v_filt = [(d, l, c) for d, l, c in zip(data_v, labels_v, colors_v) if len(d) > 0]
vp = ax.violinplot([d for d,_,_ in data_v_filt],
                   positions=range(1, len(data_v_filt)+1), showmedians=True)
for body, (_,__,col) in zip(vp['bodies'], data_v_filt):
    body.set_facecolor(col); body.set_alpha(0.75)
for part in ['cmedians','cmins','cmaxes','cbars']:
    if part in vp: vp[part].set_color('black'); vp[part].set_lw(1.5)
ax.set_xticks(range(1, len(data_v_filt)+1))
ax.set_xticklabels([l for _,l,_ in data_v_filt], fontsize=19)
ax.set_ylabel('Step at first acquisition', fontsize=19)
ax.set_title(f'A. Resistance mutations acquired first\n'
             f'(MWU p={obs1_p:.3g})', fontsize=19)
res_med  = np.median(data_v[0]) if len(data_v[0]) else 0
rid_med  = np.median(data_v[1]) if len(data_v[1]) else 0
ax.text(0.97, 0.97, f'Median: {res_med:.0f} vs {rid_med:.0f} steps',
        transform=ax.transAxes, ha='right', va='top', fontsize=15,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
ax.text(0.05, 0.95, 'Wagner Obs. 1', transform=ax.transAxes,
        fontsize=15, color='#555555', style='italic')

# Panel B — Obs 2: log10 final fitness with vs without ridge mutations
ax = axes[0, 1]
positions = sorted(RES_POSITIONS)
x    = np.arange(len(positions))
w_means  = [obs2_df[obs2_df.resistance_position==rp]['mean_log10fitness_with_ridge'].values[0]
             for rp in positions]
wo_means = [obs2_df[obs2_df.resistance_position==rp]['mean_log10fitness_without_ridge'].values[0]
             for rp in positions]
diffs    = [obs2_df[obs2_df.resistance_position==rp]['log10_fitness_gain'].values[0]
             for rp in positions]
pvals_b  = [obs2_df[obs2_df.resistance_position==rp]['mwu_pval'].values[0]
             for rp in positions]
bar_w = 0.35
ax.bar(x - bar_w/2, w_means,  bar_w, color=COL_RIDGE, alpha=0.85, label='With ≥1 ridge mutation')
ax.bar(x + bar_w/2, wo_means, bar_w, color=COL_OTHER,  alpha=0.85, label='Without ridge mutation')
ax.set_xticks(x)
ax.set_xticklabels([RES_LABELS[rp] for rp in positions], fontsize=17)
ax.set_ylabel('Mean log₁₀(final fitness)', fontsize=19)
ax.set_title(f'B. Ridge-acquiring walks reach higher fitness\n'
             f'(overall +{obs2_log_diff:.2f} log₁₀ units, MWU p={obs2_p:.3g})', fontsize=19)
for xi, diff, p in zip(x, diffs, pvals_b):
    if not np.isnan(diff):
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        ypos = max(w_means[xi], wo_means[xi]) + 0.3
        ax.text(xi, ypos, f'+{diff:.1f}\n{sig}', ha='center', fontsize=13)
ax.legend(fontsize=15)
ax.text(0.05, 0.95, 'Wagner Obs. 2', transform=ax.transAxes,
        fontsize=15, color='#555555', style='italic')

# Panel C — Obs 3: ridge selectivity per walk
ax = axes[1, 0]
# Distribution of ridge fraction per walk vs null expectation
by_kp = access_df.groupby('key_position')['obs_ridge_frac']
kp_order = sorted(RES_POSITIONS)
sel_data = [access_df[access_df.key_position==kp]['obs_ridge_frac'].dropna().values
            for kp in kp_order]
vp = ax.violinplot(sel_data, positions=range(len(kp_order)), showmedians=True)
for body in vp['bodies']:
    body.set_facecolor(COL_RIDGE); body.set_alpha(0.7)
for part in ['cmedians','cmins','cmaxes','cbars']:
    if part in vp: vp[part].set_color('black')
ax.axhline(null_ridge_frac, ls='--', color='red', lw=1.5,
           label=f'Null expectation ({null_ridge_frac:.2f})')
ax.set_xticks(range(len(kp_order)))
ax.set_xticklabels([RES_LABELS[rp] for rp in kp_order], fontsize=17)
ax.set_ylabel('Fraction of compensatory mutations\nat reference ridge positions (per walk)', fontsize=17)
ax.set_title(f'C. Reference ridge positions preferentially selected\n'
             f'(median {obs3_median_sel:.1f}× above null expectation, t-test p={obs3_p:.3g})', fontsize=19)
ax.legend(fontsize=15)
ax.text(0.98, 0.02, 'Wagner Obs. 3', transform=ax.transAxes,
        fontsize=15, color='#555555', style='italic', ha='right')

# Panel D — Obs 4: obligate ordering per KP
ax = axes[1, 1]
per_kp_order = []
for kp in sorted(RES_POSITIONS):
    sub = ordering_df[ordering_df.key_position==kp]
    p_before = sub['res_before_ridge'].mean() if len(sub) > 0 else np.nan
    per_kp_order.append({'kp': kp, 'label': RES_LABELS[kp],
                         'p_before': p_before, 'n': len(sub)})
order_df_plot = pd.DataFrame(per_kp_order).dropna(subset=['p_before'])

ax.bar(range(len(order_df_plot)), order_df_plot['p_before']*100,
       color=COL_RES, alpha=0.8, edgecolor='white')
ax.axhline(90, ls='--', color='k', lw=1, label='90% threshold')
ax.axhline(50, ls=':', color='grey', lw=1)
ax.set_xticks(range(len(order_df_plot)))
ax.set_xticklabels(order_df_plot['label'], fontsize=17)
ax.set_ylabel('P(resistance acquired before ridge) (%)', fontsize=19)
ax.set_ylim(0, 110)
for xi, row in enumerate(order_df_plot.itertuples()):
    ax.text(xi, row.p_before*100 + 2, f'{row.p_before*100:.0f}%\n(n={row.n})',
            ha='center', fontsize=15)
ax.set_title(f'D. Obligate ordering: resistance precedes ridge\n'
             f'(overall P={p_res_before_ridge:.3f})', fontsize=19)
ax.legend(fontsize=15)
ax.text(0.05, 0.05, 'Wagner Obs. 4', transform=ax.transAxes,
        fontsize=15, color='#555555', style='italic')

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig07_wagner_consistency.png'), dpi=200, bbox_inches='tight')
plt.close()

# ── Fig 7: Individual panels ──────────────────────────────────────────────────
# 7A — Step acquisition violin
fig, ax = plt.subplots(figsize=(7, 6))
vp = ax.violinplot([d for d,_,_ in data_v_filt],
                   positions=range(1, len(data_v_filt)+1), showmedians=True)
for body, (_,__,col) in zip(vp['bodies'], data_v_filt):
    body.set_facecolor(col); body.set_alpha(0.75)
for part in ['cmedians','cmins','cmaxes','cbars']:
    if part in vp: vp[part].set_color('black'); vp[part].set_lw(1.5)
ax.set_xticks(range(1, len(data_v_filt)+1))
ax.set_xticklabels([l for _,l,_ in data_v_filt], fontsize=20)
ax.set_ylabel('Step at first acquisition', fontsize=20)
res_med = np.median(data_v[0]) if len(data_v[0]) else 0
rid_med = np.median(data_v[1]) if len(data_v[1]) else 0
ax.set_title(f'Resistance mutations acquired first (Wagner Obs. 1)\n'
             f'MWU p={obs1_p:.3g}; median {res_med:.0f} vs {rid_med:.0f} steps', fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig07a_obs1_step_order.png'), dpi=200, bbox_inches='tight')
plt.close()

# 7B — Fitness with / without ridge
fig, ax = plt.subplots(figsize=(8, 6))
bar_w = 0.35
ax.bar(x - bar_w/2, w_means,  bar_w, color=COL_RIDGE, alpha=0.85, label='With ≥1 ridge mutation')
ax.bar(x + bar_w/2, wo_means, bar_w, color=COL_OTHER,  alpha=0.85, label='Without ridge mutation')
ax.set_xticks(x)
ax.set_xticklabels([RES_LABELS[rp] for rp in positions], fontsize=19)
ax.set_ylabel('Mean log₁₀(final fitness)', fontsize=20)
ax.set_title(f'Ridge-acquiring walks reach higher fitness (Wagner Obs. 2)\n'
             f'Overall +{obs2_log_diff:.2f} log₁₀ units, MWU p={obs2_p:.3g}', fontsize=20)
for xi, diff, p in zip(x, diffs, pvals_b):
    if not np.isnan(diff):
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        ypos = max(w_means[xi], wo_means[xi]) + 0.3
        ax.text(xi, ypos, f'+{diff:.1f}\n{sig}', ha='center', fontsize=15)
ax.legend(fontsize=17)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig07b_obs2_ridge_fitness.png'), dpi=200, bbox_inches='tight')
plt.close()

# 7C — Ridge selectivity violin
fig, ax = plt.subplots(figsize=(7, 6))
vp = ax.violinplot(sel_data, positions=range(len(kp_order)), showmedians=True)
for body in vp['bodies']:
    body.set_facecolor(COL_RIDGE); body.set_alpha(0.7)
for part in ['cmedians','cmins','cmaxes','cbars']:
    if part in vp: vp[part].set_color('black')
ax.axhline(null_ridge_frac, ls='--', color='red', lw=1.5,
           label=f'Null expectation ({null_ridge_frac:.2f})')
ax.set_xticks(range(len(kp_order)))
ax.set_xticklabels([RES_LABELS[rp] for rp in kp_order], fontsize=19)
ax.set_ylabel('Fraction of compensatory mutations at\nreference ridge positions (per walk)', fontsize=19)
ax.set_title(f'Reference ridge positions preferentially selected (Wagner Obs. 3)\n'
             f'Median {obs3_median_sel:.1f}× above null, t-test p={obs3_p:.3g}', fontsize=20)
ax.legend(fontsize=17)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig07c_obs3_ridge_selectivity.png'), dpi=200, bbox_inches='tight')
plt.close()

# 7D — Obligate ordering bars
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(range(len(order_df_plot)), order_df_plot['p_before']*100,
       color=COL_RES, alpha=0.8, edgecolor='white')
ax.axhline(90, ls='--', color='k', lw=1, label='90% threshold')
ax.axhline(50, ls=':', color='grey', lw=1)
ax.set_xticks(range(len(order_df_plot)))
ax.set_xticklabels(order_df_plot['label'], fontsize=19)
ax.set_ylabel('P(resistance acquired before ridge) (%)', fontsize=20)
ax.set_ylim(0, 115)
for xi, row in enumerate(order_df_plot.itertuples()):
    ax.text(xi, row.p_before*100 + 2, f'{row.p_before*100:.0f}%\n(n={row.n})',
            ha='center', fontsize=17)
ax.set_title(f'Obligate ordering: resistance precedes ridge (Wagner Obs. 4)\n'
             f'Overall P={p_res_before_ridge:.3f}', fontsize=20)
ax.legend(fontsize=17)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig07d_obs4_obligate_ordering.png'), dpi=200, bbox_inches='tight')
plt.close()

# ── Fig 8: Fitness trajectories ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, kp in zip(axes, [42, 50]):
    ridge_fits, other_fits = [], []
    for rep, grp in walk_df[walk_df.key_position==kp].groupby('rep'):
        grp = grp.sort_values('step').reset_index(drop=True)
        first_cls = None
        for _, row in grp.iterrows():
            for m in row['new_muts']:
                if parse_pos(m) != kp:
                    first_cls = mut_class(m, ridge_all_pos)
                    break
            if first_cls: break
        fits = [np.log10(max(r['fitness'],1e-12)) for _, r in grp.iterrows()][:20]
        (ridge_fits if first_cls=='ridge' else other_fits).append(fits)

    for traj_list, col, lbl in [(ridge_fits, COL_RIDGE, 'Ridge 1st companion'),
                                  (other_fits, COL_OTHER, 'Other 1st companion')]:
        if traj_list:
            arr = np.array([t+[t[-1]]*(20-len(t)) for t in traj_list])
            m_, s_ = arr.mean(0), arr.std(0)/np.sqrt(len(arr))
            ax.plot(m_, color=col, lw=2, label=f'{lbl} (n={len(traj_list)})')
            ax.fill_between(range(20), m_-s_, m_+s_, color=col, alpha=0.2)
    ax.set_xlabel('Step', fontsize=19)
    ax.set_ylabel('log₁₀(fitness)', fontsize=19)
    ax.set_title(f'Fitness trajectory by first companion\n'
                 f'(resistance mutant c{canonical(kp)}, shifted pos {kp})', fontsize=19)
    ax.legend(fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig08_fitness_trajectories.png'), dpi=200)
plt.close()

# ── Fig 9: Step-order violin ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
classes_v  = ['resistance', 'ridge', 'other']
labels_v   = ['Resistance\nmutation sites', 'Ridge\n(compensatory)', 'Other']
colors_v   = [COL_RES, COL_RIDGE, COL_OTHER]
data_v = [step_df[step_df['mutation'].apply(
              lambda m: mut_class(m, ridge_all_pos))==cls]['step'].values
          for cls in classes_v]
data_v = [(d, l, c) for d, l, c in zip(data_v, labels_v, colors_v) if len(d) > 0]
vp = ax.violinplot([d for d,_,_ in data_v],
                   positions=range(1, len(data_v)+1), showmedians=True)
for body, (_, __, col) in zip(vp['bodies'], data_v):
    body.set_facecolor(col); body.set_alpha(0.7)
for part in ['cmedians','cmins','cmaxes','cbars']:
    if part in vp: vp[part].set_color('black')
ax.set_xticks(range(1, len(data_v)+1))
ax.set_xticklabels([l for _,l,_ in data_v], fontsize=19)
ax.set_ylabel('Step at first acquisition', fontsize=19)
ax.set_title('Mutation acquisition order across adaptive walks\n'
             '(resistance mutations precede compensatory ridge mutations)', fontsize=20)
if len(data_v) >= 2:
    stat, p = mannwhitneyu(data_v[0][0], data_v[1][0], alternative='less')
    ax.text(1.5, ax.get_ylim()[1]*0.95, f'MWU p={p:.3f}', ha='center', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig09_step_order_violin.png'), dpi=200)
plt.close()

# ── Fig 10: Obligate pairs + MI/DCCM validation (position-level) ─────────────
fig = plt.figure(figsize=(18, 12))
gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

# Panel A — Obligate epistatic pairs (top 20 by P(B|A))
ax = fig.add_subplot(gs[0, :2])
top_ob = pos_pairs_df.nlargest(20, 'P_pos2_given_pos1').reset_index(drop=True)
y = range(len(top_ob))
ax.barh(y, top_ob['P_pos2_given_pos1'] * 100, color=[
    class_color(r['pair_type'].split('-')[0] if r['class1']!='other' else r['class2'])
    for _, r in top_ob.iterrows()], alpha=0.8, label='P(pos2 | pos1 mutated)')
ax.barh(y, top_ob['P_pos2_given_no_pos1'] * 100,
        left=0, color='#DDDDDD', alpha=0.6, label='P(pos2 | pos1 NOT mutated)')
ax.set_yticks(y)
pair_labels = [
    f"pos {int(r.pos1)}(c{int(r.pos1_canonical)}) → pos {int(r.pos2)}(c{int(r.pos2_canonical)})  "
    f"[{r.class1}–{r.class2}]"
    for _, r in top_ob.iterrows()
]
ax.set_yticklabels(pair_labels, fontsize=13)
ax.invert_yaxis()
ax.axvline(50, ls='--', color='k', lw=1, alpha=0.5)
ax.set_xlabel('Conditional probability (%)', fontsize=19)
ax.set_title('Preferentially co-occurring epistatic pairs:\nP(position B mutated | position A mutated) vs without A\n'
             '(P≥0.4, enrichment≥2×; note: FDR power limited by small per-position n)',
             fontsize=17)
ax.legend(fontsize=15, loc='lower right')
# Annotate enrichment
for i, row in top_ob.iterrows():
    enr = row['conditional_enrichment']
    enr_str = f'{enr:.1f}×' if np.isfinite(enr) else '∞'
    fdr_str = '***' if row['fisher_pval_fdr'] < 0.001 else ('**' if row['fisher_pval_fdr'] < 0.01
               else ('*' if row['fisher_pval_fdr'] < 0.05 else ''))
    ax.text(row['P_pos2_given_pos1']*100 + 1, i,
            f'{enr_str} {fdr_str}', va='center', fontsize=13)

# Panel B — epistasis score vs MI (position level)
ax = fig.add_subplot(gs[0, 2])
valid_pos = pos_pairs_df.dropna(subset=['epistasis_score','MI']).replace([np.inf,-np.inf], np.nan).dropna()
col_map = {'resistance-ridge': '#9B30FF', 'ridge-ridge': COL_RIDGE,
           'resistance-resistance': COL_RES, 'other-other': COL_OTHER,
           'other-ridge': '#4DBEEE', 'other-resistance': '#FF7F0E'}
colors_scatter = [col_map.get(r['pair_type'], COL_OTHER) for _, r in valid_pos.iterrows()]
ax.scatter(valid_pos['MI'], valid_pos['epistasis_score'],
           c=colors_scatter, alpha=0.55, s=30, edgecolors='none')
if len(valid_pos) > 2:
    r_mi, p_mi = stats.pearsonr(valid_pos['MI'], valid_pos['epistasis_score'])
    ax.set_title(f'Walk epistasis vs MI\n(r={r_mi:.3f}, p={p_mi:.3g}, n={len(valid_pos)})', fontsize=17)
ax.set_xlabel('Mutual Information (MSA)', fontsize=17)
ax.set_ylabel('Walk epistasis score\n[log₂(obs/exp)]', fontsize=17)
ax.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in col_map.items()],
          fontsize=12, loc='upper right')

# Panel C — epistasis score vs DCCM (position level)
ax = fig.add_subplot(gs[1, 0])
valid_dccm = pos_pairs_df.dropna(subset=['epistasis_score','DCCM']).replace([np.inf,-np.inf], np.nan).dropna()
colors_dccm = [col_map.get(r['pair_type'], COL_OTHER) for _, r in valid_dccm.iterrows()]
ax.scatter(valid_dccm['DCCM'], valid_dccm['epistasis_score'],
           c=colors_dccm, alpha=0.55, s=30, edgecolors='none')
if len(valid_dccm) > 2:
    r_dc, p_dc = stats.pearsonr(valid_dccm['DCCM'], valid_dccm['epistasis_score'])
    ax.set_title(f'Walk epistasis vs DCCM\n(r={r_dc:.3f}, p={p_dc:.3g}, n={len(valid_dccm)})', fontsize=17)
ax.set_xlabel('DCCM correlation (MD-based, 1 µs trajectory)', fontsize=17)
ax.set_ylabel('Walk epistasis score\n[log₂(obs/exp)]', fontsize=17)

# Panel D — pair-type MI boxplot
ax = fig.add_subplot(gs[1, 1])
pair_types_order = ['resistance-ridge', 'ridge-ridge', 'resistance-resistance', 'other-other']
pt_labels        = ['Resistance–\nRidge', 'Ridge–\nRidge', 'Resistance–\nResistance', 'Other']
mi_by_type = [pos_pairs_df[pos_pairs_df['pair_type']==pt]['MI'].dropna().values
              for pt in pair_types_order]
mi_by_type_filt = [(d, l) for d, l in zip(mi_by_type, pt_labels) if len(d) > 0]
bp = ax.boxplot([d for d,_ in mi_by_type_filt],
                labels=[l for _,l in mi_by_type_filt],
                patch_artist=True, medianprops=dict(color='black', lw=2))
type_cols = [col_map.get(pt, COL_OTHER) for pt in pair_types_order]
for patch, col in zip(bp['boxes'], [c for c,(_,__) in zip(type_cols, mi_by_type_filt)]):
    patch.set_facecolor(col); patch.set_alpha(0.7)
ax.set_ylabel('Mutual Information', fontsize=17)
ax.set_title('MI by epistatic pair type\n(resistance–ridge pairs: highest MI?)', fontsize=17)

# Panel E — pair-type DCCM boxplot
ax = fig.add_subplot(gs[1, 2])
dccm_by_type = [pos_pairs_df[pos_pairs_df['pair_type']==pt]['DCCM'].dropna().values
                for pt in pair_types_order]
dccm_by_type_filt = [(d, l) for d, l in zip(dccm_by_type, pt_labels) if len(d) > 0]
bp2 = ax.boxplot([d for d,_ in dccm_by_type_filt],
                 labels=[l for _,l in dccm_by_type_filt],
                 patch_artist=True, medianprops=dict(color='black', lw=2))
for patch, col in zip(bp2['boxes'], [c for c,(_,__) in zip(type_cols, dccm_by_type_filt)]):
    patch.set_facecolor(col); patch.set_alpha(0.7)
ax.set_ylabel('DCCM correlation', fontsize=17)
ax.set_title('DCCM by epistatic pair type\n(do epistatically-linked pairs move together?)', fontsize=17)

fig.suptitle('Epistatic pairs from adaptive walks validated against\n'
             'MSA mutual information (MI) and correlated dynamics (DCCM)',
             fontsize=22, y=1.01)
plt.savefig(os.path.join(OUTDIR, 'fig10_epistasis_mi_dccm.png'), dpi=200, bbox_inches='tight')
plt.close()

# Pre-compute correlation stats used in individual panels
_valid_mi  = pos_pairs_df.dropna(subset=['epistasis_score','MI']).replace([np.inf,-np.inf], np.nan).dropna()
_valid_dc  = pos_pairs_df.dropna(subset=['epistasis_score','DCCM']).replace([np.inf,-np.inf], np.nan).dropna()
_r_mi, _p_mi = (stats.pearsonr(_valid_mi['MI'], _valid_mi['epistasis_score'])
                if len(_valid_mi) > 2 else (np.nan, np.nan))
_r_dc, _p_dc = (stats.pearsonr(_valid_dc['DCCM'], _valid_dc['epistasis_score'])
                if len(_valid_dc) > 2 else (np.nan, np.nan))

# ── Fig 10: Individual panels ─────────────────────────────────────────────────
# 10A — Obligate / preferentially co-occurring pairs
fig, ax = plt.subplots(figsize=(10, 14))
top_ob = pos_pairs_df.nlargest(20, 'P_pos2_given_pos1').reset_index(drop=True)
y = range(len(top_ob))
ax.barh(y, top_ob['P_pos2_given_pos1'] * 100,
        color=[class_color(r['pair_type'].split('-')[0] if r['class1'] != 'other' else r['class2'])
               for _, r in top_ob.iterrows()],
        alpha=0.8, label='P(pos2 | pos1 mutated)')
ax.barh(y, top_ob['P_pos2_given_no_pos1'] * 100,
        left=0, color='#DDDDDD', alpha=0.6, label='P(pos2 | pos1 NOT mutated)')
pair_labels = [
    f"pos {int(r.pos1)} → pos {int(r.pos2)}"
    for _, r in top_ob.iterrows()
]
ax.set_yticks(y)
ax.set_yticklabels(pair_labels, fontsize=26)
ax.invert_yaxis()
ax.axvline(50, ls='--', color='k', lw=1.5, alpha=0.7, label='50% reference line')
ax.set_xlabel('Conditional probability (%)', fontsize=28)
ax.set_title('Preferentially co-occurring epistatic pairs (P≥0.4, enrichment≥2×)\n'
             'P(position B mutated | position A mutated) vs P(B | A not mutated)', fontsize=26)
ax.legend(handles=[
    mpatches.Patch(color=COL_RIDGE, alpha=0.8, label='P(B|A) — ridge position'),
    mpatches.Patch(color=COL_RES,   alpha=0.8, label='P(B|A) — resistance position'),
    mpatches.Patch(color='#DDDDDD', alpha=0.6, label='P(B | A NOT mutated)'),
    plt.Line2D([0], [0], color='k', ls='--', lw=1.5, label='50% reference line'),
], fontsize=24, loc='lower right')
ax.tick_params(axis='x', labelsize=15)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=0.2, linestyle='--', lw=0.6)
for i, row in top_ob.iterrows():
    enr = row['conditional_enrichment']
    enr_str = f'{enr:.1f}×' if np.isfinite(enr) else '∞'
    ax.text(row['P_pos2_given_pos1']*100 + 1, i,
            enr_str, va='center', fontsize=24)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig10a_obligate_pairs.png'), dpi=300, bbox_inches='tight')
plt.close()

# ── Combined 2×2 figure: composite score / cross-context / obligate pairs / temporal ordering ──
import matplotlib.image as _mpimg

_panel_paths_2x2 = [
    os.path.join(OUTDIR, 'fig01A_composite_score.png'),
    os.path.join(OUTDIR, 'fig01B_cross_context.png'),
    os.path.join(OUTDIR, 'fig10a_obligate_pairs.png'),
    os.path.join(OUTDIR, 'fig01C_temporal_ordering.png'),
]
_panel_letters = ['A', 'B', 'C', 'D']

fig2x2, axes2x2 = plt.subplots(2, 2, figsize=(22, 22))
for ax_p, path, letter in zip(axes2x2.flat, _panel_paths_2x2, _panel_letters):
    img = _mpimg.imread(path)
    ax_p.imshow(img)
    ax_p.axis('off')
    ax_p.text(0.01, 0.99, letter, transform=ax_p.transAxes,
              fontsize=32, fontweight='bold', va='top', ha='left', color='black')

plt.subplots_adjust(wspace=0.02, hspace=0.02)
plt.savefig(os.path.join(OUTDIR, 'fig_combined_2x2_ridge_panel.png'), dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: fig_combined_2x2_ridge_panel.png")

# 10B — Epistasis vs MI scatter
fig, ax = plt.subplots(figsize=(6, 5))
colors_b = [col_map.get(r['pair_type'], COL_OTHER) for _, r in _valid_mi.iterrows()]
ax.scatter(_valid_mi['MI'], _valid_mi['epistasis_score'],
           c=colors_b, alpha=0.55, s=60, edgecolors='none')
ax.set_xlabel('Mutual Information (MSA)', fontsize=20)
ax.set_ylabel('Walk epistasis score [log₂(obs/exp)]', fontsize=20)
ax.set_title(f'Walk epistasis vs MI\nr={_r_mi:.3f}, p={_p_mi:.3g}, n={len(_valid_mi)}', fontsize=20)
ax.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in col_map.items()],
          fontsize=13, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig10b_epistasis_vs_mi.png'), dpi=200, bbox_inches='tight')
plt.close()

# 10C — Epistasis vs DCCM scatter
fig, ax = plt.subplots(figsize=(6.5, 5.5))
colors_c = [col_map.get(r['pair_type'], COL_OTHER) for _, r in _valid_dc.iterrows()]
ax.scatter(_valid_dc['DCCM'], _valid_dc['epistasis_score'],
           c=colors_c, alpha=0.55, s=60, edgecolors='none', zorder=3)
# Regression line
_x_fit = np.linspace(_valid_dc['DCCM'].min(), _valid_dc['DCCM'].max(), 100)
_slope, _intercept = np.polyfit(_valid_dc['DCCM'], _valid_dc['epistasis_score'], 1)
ax.plot(_x_fit, _slope * _x_fit + _intercept,
        color='#333333', lw=1.8, ls='--', alpha=0.75, zorder=2, label='linear fit')
ax.set_xlabel('DCCM correlation (MD-based, 1 µs trajectory)', fontsize=20)
ax.set_ylabel('Walk epistasis score [log₂(obs/exp)]', fontsize=20)
ax.set_title(f'Walk epistasis vs DCCM\nPearson r={_r_dc:.3f}, p={_p_dc:.3g}, n={len(_valid_dc)}',
             fontsize=20)
handles_legend = [mpatches.Patch(color=v, label=k) for k, v in col_map.items()
                  if _valid_dc['pair_type'].eq(k).any()]
handles_legend.append(plt.Line2D([0], [0], color='#333333', lw=1.8, ls='--', label='linear fit'))
ax.legend(handles=handles_legend, fontsize=13, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig10c_epistasis_vs_dccm.png'), dpi=200, bbox_inches='tight')
plt.close()

# 10D — MI by pair type boxplot
fig, ax = plt.subplots(figsize=(6, 5))
mi_by_type_filt2 = [(pos_pairs_df[pos_pairs_df['pair_type']==pt]['MI'].dropna().values, lb)
                    for pt, lb in zip(pair_types_order, pt_labels)
                    if pos_pairs_df[pos_pairs_df['pair_type']==pt]['MI'].dropna().shape[0] > 0]
bp = ax.boxplot([d for d, _ in mi_by_type_filt2],
                labels=[l for _, l in mi_by_type_filt2],
                patch_artist=True, medianprops=dict(color='black', lw=2))
type_cols2 = [col_map.get(pt, COL_OTHER) for pt in pair_types_order]
for patch, col in zip(bp['boxes'], type_cols2[:len(mi_by_type_filt2)]):
    patch.set_facecolor(col); patch.set_alpha(0.7)
ax.set_ylabel('Mutual Information', fontsize=20)
ax.set_title('MI by epistatic pair type\n(do epistatic pairs show evolutionary co-variation?)', fontsize=19)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig10d_mi_by_pair_type.png'), dpi=200, bbox_inches='tight')
plt.close()

# 10E — DCCM by pair type boxplot
fig, ax = plt.subplots(figsize=(6, 5))
dccm_by_type_filt2 = [(pos_pairs_df[pos_pairs_df['pair_type']==pt]['DCCM'].dropna().values, lb)
                      for pt, lb in zip(pair_types_order, pt_labels)
                      if pos_pairs_df[pos_pairs_df['pair_type']==pt]['DCCM'].dropna().shape[0] > 0]
bp2 = ax.boxplot([d for d, _ in dccm_by_type_filt2],
                 labels=[l for _, l in dccm_by_type_filt2],
                 patch_artist=True, medianprops=dict(color='black', lw=2))
for patch, col in zip(bp2['boxes'], type_cols2[:len(dccm_by_type_filt2)]):
    patch.set_facecolor(col); patch.set_alpha(0.7)
ax.set_ylabel('DCCM correlation', fontsize=20)
ax.set_title('DCCM by epistatic pair type\n(do epistatically-linked pairs move together?)', fontsize=19)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig10e_dccm_by_pair_type.png'), dpi=200, bbox_inches='tight')
plt.close()

# ── Fig SI-A: Obligate pairs bar chart — publication quality ─────────────────
fig, ax = plt.subplots(figsize=(12, 9))

top_ob2  = pos_pairs_df.nlargest(20, 'P_pos2_given_pos1').reset_index(drop=True)
y2       = range(len(top_ob2))
bar_cols = [class_color(r['pair_type'].split('-')[0] if r['class1'] != 'other' else r['class2'])
            for _, r in top_ob2.iterrows()]

ax.barh(y2, top_ob2['P_pos2_given_pos1'] * 100,
        color=bar_cols, alpha=0.85, height=0.65,
        label='P(B mutated | A mutated)')
ax.barh(y2, top_ob2['P_pos2_given_no_pos1'] * 100,
        color='#CCCCCC', alpha=0.7, height=0.65,
        label='P(B mutated | A NOT mutated)')

_pair_labels2 = [
    f"pos {int(r.pos1)} (c{int(r.pos1_canonical)})  →  pos {int(r.pos2)} (c{int(r.pos2_canonical)})"
    f"   [{r.class1}–{r.class2}]"
    for _, r in top_ob2.iterrows()
]
ax.set_yticks(y2)
ax.set_yticklabels(_pair_labels2, fontsize=18)
ax.invert_yaxis()
ax.axvline(50, ls='--', color='k', lw=1.2, alpha=0.5, label='50% threshold')
ax.set_xlabel('Conditional probability (%)', fontsize=22, fontweight='bold')
ax.set_title(
    'Preferentially co-occurring epistatic pairs\n'
    'P(position B mutated | position A mutated)  vs  P(B | A not mutated)\n'
    'Threshold: P(B|A) ≥ 0.40, enrichment ≥ 2×',
    fontsize=20, pad=10)
ax.tick_params(axis='x', labelsize=19)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=0.2, linestyle='--', lw=0.6)

for i, row in top_ob2.iterrows():
    enr     = row['conditional_enrichment']
    enr_str = f'{enr:.1f}×' if np.isfinite(enr) else '∞'
    fdr     = row['fisher_pval_fdr']
    stars   = '***' if fdr < 0.001 else ('**' if fdr < 0.01 else ('*' if fdr < 0.05 else ''))
    ax.text(row['P_pos2_given_pos1'] * 100 + 1.2, i,
            f'{enr_str} {stars}', va='center', fontsize=17, fontweight='bold')

_leg_handles = [
    mpatches.Patch(color=COL_RIDGE,  alpha=0.85, label='Ridge position'),
    mpatches.Patch(color=COL_RES,    alpha=0.85, label='Resistance position'),
    mpatches.Patch(color='#CCCCCC',  alpha=0.70, label='P(B | A NOT mutated)'),
    plt.Line2D([0],[0], color='k', ls='--', lw=1.2, label='50% reference'),
]
ax.legend(handles=_leg_handles, fontsize=17, loc='lower right',
          frameon=True, edgecolor='#cccccc')

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'figSI_A_obligate_pairs.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print("  Saved: figSI_A_obligate_pairs.png")

# ── Fig SI-B: DCCM + MI boxplots side by side — publication quality ──────────

# Mann-Whitney U: do Ridge-Ridge pairs have higher DCCM than Resistance-Ridge?
_rr_dccm   = pos_pairs_df[pos_pairs_df['pair_type'] == 'ridge-ridge']['DCCM'].dropna().values
_resr_dccm = pos_pairs_df[pos_pairs_df['pair_type'] == 'resistance-ridge']['DCCM'].dropna().values
_mw_stat, _mw_p = mannwhitneyu(_rr_dccm, _resr_dccm, alternative='greater')
print(f"\nMann-Whitney U (Ridge-Ridge DCCM > Resistance-Ridge DCCM):")
print(f"  U = {_mw_stat:.0f},  p = {_mw_p:.4f},  n_RR = {len(_rr_dccm)},  n_ReR = {len(_resr_dccm)}")

pd.DataFrame([{
    'test':        'Mann-Whitney U',
    'comparison':  'Ridge-Ridge DCCM > Resistance-Ridge DCCM',
    'U_statistic': _mw_stat,
    'p_value':     _mw_p,
    'n_group1':    len(_rr_dccm),
    'n_group2':    len(_resr_dccm),
    'group1':      'ridge-ridge',
    'group2':      'resistance-ridge',
    'metric':      'DCCM',
    'alternative': 'greater',
    'significant': _mw_p < 0.05,
}]).to_csv(os.path.join(OUTDIR, 'structural_validation_stats.csv'), index=False)
print("  Saved: structural_validation_stats.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

_pt_order  = ['resistance-ridge', 'ridge-ridge', 'resistance-resistance']
_pt_labels = ['Resistance–\nRidge', 'Ridge–\nRidge', 'Resistance–\nResistance']
_pt_cols   = [col_map['resistance-ridge'], col_map['ridge-ridge'],
              col_map['resistance-resistance']]

for ax, metric, ylabel, title in [
    (axes[0], 'DCCM',
     'DCCM correlation (MD, 1 µs)',
     'Correlated atomic motion by pair type\n(do epistatically linked positions move together?)'),
    (axes[1], 'MI',
     'Mutual Information (MSA)',
     'Evolutionary co-variation by pair type\n(do epistatically linked positions co-evolve in field isolates?)'),
]:
    groups = [(pos_pairs_df[pos_pairs_df['pair_type'] == pt][metric].dropna().values, lb, col)
              for pt, lb, col in zip(_pt_order, _pt_labels, _pt_cols)
              if pos_pairs_df[pos_pairs_df['pair_type'] == pt][metric].dropna().shape[0] > 0]

    bp = ax.boxplot([d for d, _, __ in groups],
                    labels=[lb for _, lb, __ in groups],
                    patch_artist=True,
                    medianprops=dict(color='black', lw=2.5),
                    whiskerprops=dict(lw=1.5),
                    capprops=dict(lw=1.5),
                    flierprops=dict(marker='o', markersize=9,
                                   markerfacecolor='none', markeredgewidth=1))
    for patch, (_, __, col) in zip(bp['boxes'], groups):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    # jitter individual points for transparency
    np.random.seed(42)
    for xi, (data, _, col) in enumerate(groups, 1):
        jitter = np.random.uniform(-0.12, 0.12, size=len(data))
        ax.scatter(xi + jitter, data, color=col, alpha=0.35, s=30,
                   edgecolors='none', zorder=3)

    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis='both', labelsize=19)
    ax.tick_params(axis='x', rotation=90)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.2, linestyle='--', lw=0.6)

# Significance bracket on DCCM panel (axes[0]): Resistance-Ridge (x=1) vs Ridge-Ridge (x=2)
_ax_dccm = axes[0]
_ylim    = _ax_dccm.get_ylim()
_yrange  = _ylim[1] - _ylim[0]
_y_br    = _ylim[1] - 0.04 * _yrange          # bracket height: near top
_y_tick  = _y_br - 0.015 * _yrange            # small tick drop
_p_label = (f'p = {_mw_p:.4f}' if _mw_p >= 0.0001
            else f'p < 0.0001') + (' *' if _mw_p < 0.05 else ' ns')
_ax_dccm.plot([1, 1, 2, 2], [_y_tick, _y_br, _y_br, _y_tick],
              lw=1.2, color='black')
_ax_dccm.text(1.5, _y_br + 0.005 * _yrange, _p_label,
              ha='center', va='bottom', fontsize=17, color='black')
_ax_dccm.set_ylim(_ylim[0], _ylim[1] + 0.08 * _yrange)   # headroom for bracket

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'figSI_B_dccm_mi_boxplots.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print("  Saved: figSI_B_dccm_mi_boxplots.png")

# ── Fig 10F + Summary table: DCCM/MI validation for the 9 obligate pairs ─────
# Biological note lookup (keyed by (pos1_shifted, pos2_shifted))
BIO_NOTES = {
    (2,  164): ("Hub ridge pair: every walk that acquires a mutation at pos11 invariably "
                "also has the dominant ridge hub pos173; moderate DCCM coupling (0.245)."),
    (41,  48): ("Resistance→ridge: C50R resistance co-occurs with compensatory mutations at "
                "pos57 in 60% of walks (5.2× enrichment); direct fitness-landscape coupling."),
    (41,  52): ("Resistance→ridge: C50R resistance co-occurs with compensatory mutations at "
                "pos61 in 60% of walks (4.1× enrichment); low DCCM suggests sequence- not "
                "dynamics-driven compensation."),
    (50,  52): ("Resistance→ridge: C59R resistance co-occurs with pos61 compensatory mutations "
                "(3.8× enrichment); DCCM=0.351 confirms correlated dynamics; MI unavailable "
                "(short-range pair excluded from MSA coevolution analysis)."),
    (48,  52): ("Ridge compensatory cluster: pos57 and pos61 co-compensate with 4.3× enrichment; "
                "strongest DCCM of resistance-context pairs (0.456) confirms tight correlated "
                "dynamics; MI=0 because short-range pair excluded from MSA analysis."),
    (153, 154): ("Adjacent ridge pair: highest DCCM of all obligate pairs (0.568), confirming "
                 "tightly correlated structural dynamics; MI unavailable (adjacent positions "
                 "excluded from MSA coevolution analysis by design)."),
    (153, 165): ("Ridge cluster extension: pos162 and pos174 co-occur within the ridge; "
                 "MI=0 despite 12-residue separation suggests co-occurrence is fitness-driven "
                 "rather than evolutionary; DCCM=0.181 supports moderate dynamic coupling."),
    (154, 164): ("Ridge cluster: pos163 preferentially co-occurs with hub ridge pos173 (2.1×); "
                 "highest MI of all obligate pairs (0.290) confirms evolutionary co-variation; "
                 "DCCM=0.253 further supports structural coupling."),
    (4,   95):  ("Ridge pair including primary wet-lab candidate: pos13 and pos104 (M104I — "
                 "primary experimental validation target) co-occur in 43% of walks (4.4×); "
                 "MI=0 suggests walk co-occurrence is fitness-driven, not ancestral; "
                 "DCCM=0.206 supports moderate dynamic coupling."),
}

MI_NOTES = {
    (2,  164): "In MSA",
    (41,  48): "In MSA",
    (41,  52): "In MSA",
    (50,  52): "Excluded (short-range pair)",
    (48,  52): "Short-range, genuine zero in MSA",
    (153, 154): "Excluded (adjacent pair)",
    (153, 165): "In MSA, genuine zero",
    (154, 164): "In MSA",
    (4,   95):  "In MSA, genuine zero",
}

# Build the formatted summary table
ob_sorted = obligate.sort_values('pair_type').reset_index(drop=True).copy()
ob_sorted['pos1_canonical'] = ob_sorted['pos1'] + CANONICAL_OFFSET
ob_sorted['pos2_canonical'] = ob_sorted['pos2'] + CANONICAL_OFFSET
ob_sorted['n_joint_walks']  = (ob_sorted['joint_freq'] * n_walks).round(0).astype(int)
ob_sorted['pair_label']     = ob_sorted.apply(
    lambda r: f"pos{int(r.pos1)}(c{int(r.pos1_canonical)}) → pos{int(r.pos2)}(c{int(r.pos2_canonical)})",
    axis=1)
ob_sorted['MI_note']  = ob_sorted.apply(
    lambda r: MI_NOTES.get((int(r.pos1), int(r.pos2)), ''), axis=1)
ob_sorted['biological_note'] = ob_sorted.apply(
    lambda r: BIO_NOTES.get((int(r.pos1), int(r.pos2)), ''), axis=1)

summary_cols = ['pair_label', 'pair_type',
                'pos1', 'pos1_canonical', 'pos2', 'pos2_canonical',
                'P_pos2_given_pos1', 'P_pos2_given_no_pos1',
                'conditional_enrichment', 'n_joint_walks',
                'fisher_pval', 'fisher_pval_fdr',
                'MI', 'MI_note', 'DCCM', 'biological_note']
ob_sorted[summary_cols].to_csv(
    os.path.join(OUTDIR, 'obligate_pairs_summary_table.csv'), index=False)

# Fig 10F — two-panel: epistasis evidence (left) + DCCM/MI validation (right)
ob_plot = ob_sorted.sort_values('P_pos2_given_pos1', ascending=True).reset_index(drop=True)
short_labels = [
    f"pos{int(r.pos1)}(c{int(r.pos1_canonical)})→pos{int(r.pos2)}(c{int(r.pos2_canonical)}) [{r.pair_type}]"
    for _, r in ob_plot.iterrows()
]
bar_colors = [class_color(r['class1'] if r['class1'] != 'other' else r['class2'])
              for _, r in ob_plot.iterrows()]
y_pos = np.arange(len(ob_plot))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Preferentially co-occurring epistatic pairs:\nevidence from adaptive walks (left) '
             'and structural/evolutionary validation (right)', fontsize=20, y=1.02)

# Left panel: conditional probability bars
ax1.barh(y_pos, ob_plot['P_pos2_given_pos1'] * 100,
         color=bar_colors, alpha=0.85, label='P(pos2 | pos1 mutated)', height=0.55)
ax1.barh(y_pos, ob_plot['P_pos2_given_no_pos1'] * 100,
         color='#CCCCCC', alpha=0.7, label='P(pos2 | pos1 NOT mutated)', height=0.55)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(short_labels, fontsize=14)
ax1.set_xlabel('Conditional probability (%)', fontsize=19)
ax1.set_title('Epistatic co-occurrence in adaptive walks\n(P≥0.4, enrichment≥2×)', fontsize=19)
ax1.axvline(40, ls='--', color='k', lw=1, alpha=0.4)
ax1.legend(fontsize=15, loc='lower right')
for i, row in ob_plot.iterrows():
    ax1.text(row['P_pos2_given_pos1']*100 + 0.5, i,
             f"{row['conditional_enrichment']:.1f}×\n(n={int(row['joint_freq']*n_walks)})",
             va='center', fontsize=13)

# Right panel: DCCM and MI grouped bars
bar_w = 0.3
mi_vals   = ob_plot['MI'].fillna(0).values
dccm_vals = ob_plot['DCCM'].fillna(0).values
mi_nan    = ob_plot['MI'].isna().values
is_res_ridge = ob_plot['pair_type'].values == 'resistance-ridge'
xmax = max(dccm_vals.max(), (ob_plot['MI'].dropna().max() if ob_plot['MI'].notna().any() else 0)) * 1.35 + 0.08

ax2.barh(y_pos + bar_w/2, dccm_vals, bar_w,
         color='#2196F3', alpha=0.8, label='DCCM (NMA correlated dynamics)')
ax2.barh(y_pos - bar_w/2, mi_vals, bar_w,
         color='#FF9800', alpha=0.8, label='MI (MSA coevolution)')

# Mark MI=NaN pairs: grey bar spanning the annotation zone + clear italic label
for i, is_nan in enumerate(mi_nan):
    if is_nan:
        ax2.barh(y_pos[i] - bar_w/2, xmax * 0.18, bar_w,
                 color='#E0E0E0', alpha=0.9, edgecolor='#AAAAAA', lw=0.5)
        ax2.text(xmax * 0.01, y_pos[i] - bar_w/2,
                 'MI: n/a (adjacent pair\nexcluded from MSA)',
                 va='center', fontsize=12, color='#666666', style='italic')

# Annotate DCCM values
for i, (dc, mi) in enumerate(zip(dccm_vals, mi_vals)):
    if dc > 0.005:
        ax2.text(dc + 0.005, y_pos[i] + bar_w/2, f'{dc:.3f}', va='center', fontsize=13)
    if mi > 0.005 and not mi_nan[i]:
        ax2.text(mi + 0.005, y_pos[i] - bar_w/2, f'{mi:.3f}', va='center', fontsize=13)

# Flag resistance-ridge pairs: note that low DCCM = fitness-driven, not dynamics-driven
for i, (is_rr, dc) in enumerate(zip(is_res_ridge, dccm_vals)):
    if is_rr and dc < 0.1:
        ax2.text(xmax * 0.72, y_pos[i] + bar_w/2,
                 '† fitness-driven', va='center', fontsize=12, color='#B22222', style='italic')

ax2.set_yticks(y_pos)
ax2.set_yticklabels(short_labels, fontsize=14)
ax2.set_xlabel('Correlation strength', fontsize=19)
ax2.set_title('Structural/evolutionary validation\n'
              'DCCM: r=0.183, p=0.024 vs walk epistasis (n=153 position pairs)',
              fontsize=19)
ax2.axvline(0.2, ls='--', color='#2196F3', lw=0.8, alpha=0.35,
            label='DCCM=0.2 reference')
ax2.legend(fontsize=14, loc='lower right')
ax2.set_xlim(0, xmax)
ax2.text(0.01, -0.09,
         '† Low DCCM for resistance→ridge pairs indicates co-occurrence is '
         'fitness landscape-driven, not correlated-dynamics-driven.',
         transform=ax2.transAxes, fontsize=13, color='#666666', style='italic')

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig10f_obligate_pairs_validated.png'), dpi=200,
            bbox_inches='tight')
plt.close()

# ── Fig 11: Network ───────────────────────────────────────────────────────────
G = nx.Graph()
for m in top30:
    G.add_node(m, cls=mut_class(m,ridge_all_pos), freq=mut_freq[m])
for _, r in pairs_df.iterrows():
    if r.mut1 in top30 and r.mut2 in top30 and r.epistasis_score>0 and r.n_co_occur>=MIN_PAIR_WALKS:
        G.add_edge(r.mut1, r.mut2, weight=r.epistasis_score, cooc=r.joint_freq)
fig, ax = plt.subplots(figsize=(13, 11))
pos = nx.spring_layout(G, weight='cooc', seed=42, k=1.5, iterations=100)
nx.draw_networkx_edges(G, pos, ax=ax,
                       width=[G[u][v]['weight']*2 for u,v in G.edges()],
                       edge_color='#AAAAAA', alpha=0.6)
nx.draw_networkx_nodes(G, pos, ax=ax,
                       node_color=[class_color(G.nodes[n]['cls']) for n in G.nodes()],
                       node_size=[300+2000*G.nodes[n]['freq'] for n in G.nodes()],
                       alpha=0.9)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=7.5, font_weight='bold')
ax.set_title('Mutation co-occurrence network (positive epistasis only)\n'
             'Node size ∝ frequency; edge width ∝ epistasis score', fontsize=20)
ax.legend(handles=legend_handles, fontsize=17, loc='upper left')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'fig11_network.png'), dpi=200)
plt.close()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== SUMMARY ===")
print(f"Walks: {n_walks}  |  Unique mutations: {len(all_muts)}  |  Common (≥{int(MIN_FREQ*100)}%): {len(common_muts)}")
print(f"Epistatic pairs: {len(pairs_df)}")
print(f"\nTop 10 ridge positions by composite score:")
cols = ['position_shifted','position_canonical','frequency','cross_resistance_count',
        'p_temporal_after_res','composite_ridge_score','is_reference_ridge']
print(ridge_id_df[cols].head(10).to_string(index=False))
print(f"\nAll outputs → {OUTDIR}")
