#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Date: 01012026
PfPATH — Population-Genetic & Adaptive-Walk Simulator for PfDHFR

This script models PfDHFR evolution under biologically informed constraints.

Inputs (optional except WT):
- DMS fitness landscape                       (--dms)
- SPIRED thermodynamic ddG                    (--thermo)
- Rosetta ddG (single, double, triple, quad)  (--ddg_s, --ddg_combo ...)
- Co-evolution mutual information             (--mi)
- DCCM contact strength                       (--dccm)

Biology-aware defaults:
- Mutation rate μ ~1e-5 per site per generation (effective population rate)
- Population size N default 10000
- Selection strength β default 2.0
- ΔΔG-based stability cutoff using SPIRED ddG (default cutoff 4.0 kcal/mol)
- Drug mode with haplotype-specific selection coefficients:
    * Triple: N51I+C59R+S108N
    * Quad:   N51I+C59R+S108N+I164L

Modes:
- Population (Wright–Fisher) simulation  [default]
- Adaptive walks around key positions    [--walks]
"""

import argparse
import csv
import warnings
import random
import re
from collections import defaultdict, Counter
from typing import Dict, Tuple, List, Optional
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# colour palette for adaptive walks
TAB20 = list(plt.get_cmap("tab20").colors)


# ========================= Utility helpers =========================

def warn(msg: str) -> None:
    warnings.warn(msg)


def normalise_array(values: List[float]) -> np.ndarray:
    """Z-score normalise then squash with tanh(z/2) → [-1, 1]."""
    arr = np.array(values, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-8:
        z = arr - mean
    else:
        z = (arr - mean) / std
    return np.tanh(z / 2.0)


def load_wt_fasta(path: str) -> str:
    with open(path) as f:
        lines = f.read().strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"WT FASTA {path} seems malformed.")
    return lines[1].strip()


# ========================= Loaders =========================

def load_dms(path: Optional[str]) -> Dict[Tuple[int, str], float]:
    """Load DMS epistasis single-mutant fitness; returns dict[(pos, aa)] = normalised score."""
    if not path:
        warn("DMS file not provided; DMS term will be omitted.")
        return {}
    data = {}
    vals = []
    with open(path) as f:
        reader = csv.DictReader(f)
        if 'pos' not in reader.fieldnames or 'subs' not in reader.fieldnames:
            warn("DMS file missing columns 'pos' or 'subs'; skipping DMS.")
            return {}
        # Try common score column names
        score_col = None
        for candidate in ('prediction_epistatic', 'score', 'fitness'):
            if candidate in reader.fieldnames:
                score_col = candidate
                break
        if score_col is None:
            warn("DMS file has no recognized score column; skipping DMS.")
            return {}

        for row in reader:
            try:
                pos = int(row['pos'])
                aa = row['subs'].strip()
                if not aa:
                    continue
                val = float(row[score_col])
            except Exception:
                continue
            data[(pos, aa)] = val
            vals.append(val)

    norm = normalise_array(vals)
    for key, nv in zip(data.keys(), norm):
        data[key] = nv
    return data


def load_thermo(path: Optional[str]) -> Tuple[Dict[Tuple[int, str], float],
                                             Dict[Tuple[int, str], float]]:
    """Load SPIRED ddG; returns (thermo_norm, thermo_raw) dicts keyed by (pos, aa)."""
    if not path:
        warn("Thermo file not provided; thermo term and stability cutoff will be omitted.")
        return {}, {}

    data_raw = {}
    ddg_vals = []
    with open(path) as f:
        reader = csv.DictReader(f)
        if 'mutant' not in reader.fieldnames or 'ddG' not in reader.fieldnames:
            warn("Thermo file missing 'mutant' or 'ddG' columns; skipping thermo.")
            return {}, {}
        for row in reader:
            mut = row['mutant'].strip()
            if not mut:
                continue
            try:
                pos = int(mut[1:-1])
                aa = mut[-1]
            except Exception:
                continue
            ddg = float(row['ddG'])
            data_raw[(pos, aa)] = ddg
            ddg_vals.append(ddg)

    if not ddg_vals:
        warn("No valid thermo entries found; skipping thermo.")
        return {}, {}

    norm = normalise_array(ddg_vals)
    thermo_norm = {}
    for key, nv in zip(data_raw.keys(), norm):
        # Flip sign so stabilizing (negative ddG) becomes positive contribution
        thermo_norm[key] = -nv
    return thermo_norm, data_raw


def parse_mutation_label(label: str, offset: int = -9) -> Tuple[int, str]:
    """
    Parse Rosetta-style label like N108I, apply offset to map to PfDHFR indexing.
    """
    label = label.strip()
    pos = int(label[1:-1]) + offset
    aa = label[-1]
    return pos, aa


def load_ddg_single(path: Optional[str]) -> Dict[Tuple[int, str], float]:
    """Load Rosetta single-mutant ddG; applies -9 offset to map canonical→shifted numbering."""
    if not path:
        warn("Single ddG file not provided; Rosetta single ddG term will be omitted.")
        return {}

    raw = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        if 'case_name' not in reader.fieldnames or 'total_score' not in reader.fieldnames:
            warn("Single ddG file missing 'case_name' or 'total_score'; skipping.")
            return {}
        for row in reader:
            first_label = row['case_name'].split(',')[0]
            try:
                pos, aa = parse_mutation_label(first_label)
            except Exception:
                continue
            raw[(pos, aa)].append(float(row['total_score']))

    if not raw:
        warn("No valid entries in single ddG file; skipping.")
        return {}

    data = {k: float(np.mean(v)) for k, v in raw.items()}
    norm = normalise_array(list(data.values()))
    out = {}
    for key, nv in zip(data.keys(), norm):
        # Rosetta total_score: more negative = more stable → flip sign
        out[key] = -nv
    return out


def load_ddg_combos(paths: Optional[List[str]]) -> Dict[frozenset, float]:
    """Load Rosetta ddG for double/triple/quad combos; returns dict[frozenset[(pos, aa)]]."""
    if not paths:
        warn("No ddg_combo files provided; higher-order ddG term will be omitted.")
        return {}

    raw = defaultdict(list)
    for file in paths:
        with open(file) as f:
            reader = csv.DictReader(f)
            if 'case_name' not in reader.fieldnames or 'total_score' not in reader.fieldnames:
                warn(f"Combo ddG file {file} missing 'case_name' or 'total_score'; skipping.")
                continue
            for row in reader:
                muts_str = row['case_name'].split(',')[0]
                labels = muts_str.split('_')
                try:
                    muts = [parse_mutation_label(m) for m in labels]
                except Exception:
                    continue
                key = frozenset(muts)
                raw[key].append(float(row['total_score']))

    if not raw:
        warn("No valid entries in combo ddG files; skipping ddG combos.")
        return {}

    data = {k: float(np.mean(v)) for k, v in raw.items()}
    norm = normalise_array(list(data.values()))
    out = {}
    for key, nv in zip(data.keys(), norm):
        out[key] = -nv
    return out


def load_mi(path: Optional[str]) -> Dict[Tuple[int, int], float]:
    """Load MI co-evolution pairs; returns symmetric dict[(i, j)] = MI in [0, 1]."""
    if not path:
        warn("MI file not provided; MI term will be omitted.")
        return {}

    # Accept both short ('Res1','Res2','MI') and long ('Residue 1','Residue 2','Mutual Information') headers
    COL_ALIASES = {
        'Res1': ('Res1', 'Residue 1', 'res1', 'residue1'),
        'Res2': ('Res2', 'Residue 2', 'res2', 'residue2'),
        'MI':   ('MI', 'Mutual Information', 'mi', 'mutual_information'),
    }

    out = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        def resolve(key):
            for alias in COL_ALIASES[key]:
                if alias in fields:
                    return alias
            return None

        col1 = resolve('Res1')
        col2 = resolve('Res2')
        colmi = resolve('MI')

        if not col1 or not col2 or not colmi:
            warn(f"MI file missing required columns (checked aliases); found: {fields}; skipping MI.")
            return {}

        for row in reader:
            try:
                i = int(re.sub(r'\D', '', row[col1]))
                j = int(re.sub(r'\D', '', row[col2]))
                v = float(np.clip(float(row[colmi]), 0.0, 1.0))
            except Exception:
                continue
            out[(i, j)] = v
            out[(j, i)] = v

    if not out:
        warn("No valid MI entries found; skipping MI.")
        return {}

    return out


def load_blosum(path: Optional[str]) -> Dict[Tuple[int, str], float]:
    """Load position-specific BLOSUM substitution scores; returns normalised dict[(pos, aa)]."""
    if not path:
        return {}

    STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")
    data: Dict[Tuple[int, str], float] = {}
    vals: List[float] = []

    try:
        df = pd.read_csv(path)
    except Exception as e:
        warn(f"Failed to read BLOSUM file {path}: {e}")
        return {}

    if 'Position' not in df.columns or 'WT_AA' not in df.columns:
        warn("BLOSUM file missing 'Position' or 'WT_AA' columns; skipping BLOSUM.")
        return {}

    aa_cols = [c for c in df.columns if c in STANDARD_AAS]
    for _, row in df.iterrows():
        try:
            pos = int(row['Position'])
            wt_aa = str(row['WT_AA']).strip()
        except Exception:
            continue
        for aa in aa_cols:
            if aa == wt_aa:
                continue
            try:
                score = float(row[aa])
            except Exception:
                continue
            data[(pos, aa)] = score
            vals.append(score)

    if not vals:
        warn("No valid BLOSUM entries found; skipping BLOSUM.")
        return {}

    norm = normalise_array(vals)
    out: Dict[Tuple[int, str], float] = {}
    for key, nv in zip(data.keys(), norm):
        out[key] = nv
    print(f"[BLOSUM] Loaded {len(out)} position-specific substitution scores "
          f"({len(set(k[0] for k in out))} positions).")
    return out


def load_pocket(path: Optional[str]) -> Dict[int, Tuple[str, float]]:
    """Load binding-pocket position weights (pos, role, weight CSV); returns dict[pos] = (role, weight)."""
    if not path:
        return {}
    out: Dict[int, Tuple[str, float]] = {}
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                pos    = int(row['pos'])
                role   = str(row.get('role', 'structural')).strip()
                weight = float(row.get('weight', 1.0))
                out[pos] = (role, weight)
    except Exception as e:
        warn(f"Failed to read pocket file {path}: {e}")
        return {}
    if out:
        print(f"[Pocket] Loaded {len(out)} binding-pocket positions: "
              f"{sorted(out.keys())}")
    return out


def load_dccm(path: Optional[str]) -> Dict[Tuple[int, int], float]:
    """Load DCCM matrix CSV; returns normalised off-diagonal dict[(i, j)] = score."""
    if not path:
        warn("DCCM file not provided; DCCM term will be omitted.")
        return {}

    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as e:
        warn(f"Failed to read DCCM file {path}: {e}")
        return {}

    data = {}
    vals = []
    for i in df.index:
        for j in df.columns:
            try:
                ii = int(i)
                jj = int(j)
            except Exception:
                continue
            if ii == jj:
                continue
            v = float(df.loc[i, j])
            data[(ii, jj)] = v
            vals.append(v)

    if not vals:
        warn("No valid DCCM values found; skipping DCCM.")
        return {}

    norm_vals = normalise_array(vals)
    out = {}
    for key, nv in zip(data.keys(), norm_vals):
        out[key] = nv
    return out


# ========================= Fitness model =========================

class FitnessModel:
    """Composite fitness model: exp(β·(S1 + S2 + penalty)) × stability_factor."""

    def __init__(
        self,
        wt_seq: str,
        dms: Dict[Tuple[int, str], float],
        thermo_norm: Dict[Tuple[int, str], float],
        thermo_raw: Dict[Tuple[int, str], float],
        ddg_single: Dict[Tuple[int, str], float],
        ddg_highorder: Dict[frozenset, float],
        mi: Dict[Tuple[int, int], float],
        dccm: Dict[Tuple[int, int], float],
        weights: Dict[str, float],
        ddg_cutoff_kcal: Optional[float] = 4.0,
        ddg_soft_start: Optional[float] = 2.0,
        mi_lone_threshold: float = 0.5,
        blosum: Optional[Dict[Tuple[int, str], float]] = None,
        pocket_weights: Optional[Dict[int, Tuple[str, float]]] = None,
    ):
        self.wt = wt_seq
        self.dms = dms
        self.thermo_norm = thermo_norm
        self.thermo_raw = thermo_raw
        self.ddg_s = ddg_single
        self.ddg_c = ddg_highorder
        self.mi = mi
        self.dccm = dccm
        self.w = weights
        self.blosum = blosum or {}
        self.pocket_weights = pocket_weights or {}

        self.ddg_cutoff_kcal = ddg_cutoff_kcal
        self.ddg_soft_start = ddg_soft_start

        # Precompute strongly-coupled MI partners (above threshold) for lone-pair penalty.
        # Stored as position → [(partner_pos, mi_value), ...].
        self._mi_partners: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        seen_pairs: set = set()
        for (i, j), v in mi.items():
            pair = (min(i, j), max(i, j))
            if pair not in seen_pairs and v >= mi_lone_threshold:
                seen_pairs.add(pair)
                self._mi_partners[i].append((j, v))
                self._mi_partners[j].append((i, v))

    def _mutations(self, seq: str) -> List[Tuple[int, str]]:
        """Return list of (pos, aa) where seq differs from WT (1-based positions)."""
        muts = []
        for i, (a, b) in enumerate(zip(self.wt, seq)):
            if a != b:
                muts.append((i + 1, b))
        return muts

    def _stability_penalty_factor(self, muts: List[Tuple[int, str]]) -> float:
        """Apply soft/hard stability cutoff based on SPIRED ddG_raw."""
        if not self.thermo_raw or self.ddg_cutoff_kcal is None:
            return 1.0
        if not muts:
            return 1.0

        ddgs = [self.thermo_raw.get((p, aa), 0.0) for p, aa in muts]
        max_ddg = max(ddgs) if ddgs else 0.0

        if max_ddg <= 0.0:
            return 1.0

        if max_ddg >= self.ddg_cutoff_kcal:
            return 1e-12

        if self.ddg_soft_start is not None and max_ddg > self.ddg_soft_start:
            span = max(self.ddg_cutoff_kcal - self.ddg_soft_start, 1e-6)
            x = (max_ddg - self.ddg_soft_start) / span  # in (0,1)
            return float(np.exp(-2.0 * x))

        return 1.0

    def compute_fitness(self, seq: str) -> float:
        muts = self._mutations(seq)

        # Hard cap: sequences with more mutations than biologically plausible are inviable
        if len(muts) > int(self.w.get('max_mut_load', 12)):
            return 1e-12

        # Lethality guard: A7V + S99N is experimentally inactive (no DHFR activity)
        _pos2aa = dict(muts)
        if _pos2aa.get(7) == 'V' and _pos2aa.get(99) == 'N':
            return 1e-12

        # Stability gate (ΔΔG_max filter per methodology)
        stability_factor = self._stability_penalty_factor(muts)
        if stability_factor <= 1e-12:
            return 1e-12

        # S1: coverage-weighted single-site contributions.
        # pocket_weights scale up functionally critical positions (binding pocket).
        # BLOSUM fills in evolutionary substitution cost for DMS-dark positions.
        # Unknown penalty fires only when ALL data sources are absent.
        S1 = 0.0
        for (p, aa) in muts:
            dms_val    = self.dms.get((p, aa), None)
            thermo_val = self.thermo_norm.get((p, aa), None)
            ddg_val    = self.ddg_s.get((p, aa), None)
            blosum_val = self.blosum.get((p, aa), None)

            pocket_w = self.pocket_weights.get(p, ('none', 1.0))[1]

            scored = (
                self.w['w_dms']              * (dms_val    if dms_val    is not None else 0.0) +
                self.w['w_thermo']           * (thermo_val if thermo_val is not None else 0.0) +
                self.w['w_ddg']              * (ddg_val    if ddg_val    is not None else 0.0) +
                self.w.get('w_blosum', 0.0)  * (blosum_val if blosum_val is not None else 0.0)
            )
            S1 += pocket_w * scored

            if dms_val is None and thermo_val is None and ddg_val is None:
                S1 -= self.w.get('w_unknown', 0.1)

        # S2: pairwise epistasis contributions
        S2 = 0.0
        mutated_positions = {p for p, _ in muts}

        for i, m1 in enumerate(muts):
            p1, _ = m1
            for m2 in muts[i + 1:]:
                p2, _ = m2
                key_pair = frozenset([m1, m2])
                S2 += self.w['w_ddg_pair'] * self.ddg_c.get(key_pair, 0.0)
                S2 += self.w['w_mi']        * self.mi.get((p1, p2), 0.0)
                S2 += self.w['w_dccm']      * self.dccm.get((p1, p2), 0.0)

        # Lone-pair penalty: mutating one residue of a strongly MI-coupled pair
        # without its partner incurs a small penalty proportional to coupling strength.
        w_lone = self.w.get('w_lone', 0.05)
        for (p, _) in muts:
            for (partner, mi_val) in self._mi_partners.get(p, []):
                if partner not in mutated_positions:
                    S2 -= w_lone * mi_val

        # Linear purifying selection: small cost per mutation models the average
        # deleterious background effect of random amino acid changes. Applied to
        # ALL mutations (unlike w_count which only kicks in above 6). Default 0
        # for adaptive walks; set via --wf_purify in WF oracle-only mode to
        # prevent neutral sequence drift that would slow the simulation.
        w_purify = self.w.get('w_purify', 0.0)
        S1 -= w_purify * len(muts)

        # Quadratic mutation-load penalty — grows fast enough to cap walk depth
        # while staying gentle for sequences with ≤6 mutations.
        n_excess = max(0, len(muts) - 6)
        penalty = -self.w['w_count'] * (n_excess + 0.1 * n_excess ** 2)

        total_score = S1 + S2 + penalty
        if not np.isfinite(total_score):
            return 1e-12

        beta = self.w.get('beta', 2.0)
        x = float(np.clip(beta * total_score, -50.0, 50.0))
        intrinsic = max(float(np.exp(x)), 1e-12)

        fitness = intrinsic * stability_factor
        return fitness if (np.isfinite(fitness) and fitness > 0.0) else 1e-12


# ========================= Drug / haplotype logic =========================

def classify_pf_dhfr_haplotype(seq: str) -> str:
    """Classify PfDHFR haplotype (shifted numbering): lethal/cyc_double/cyc_single/quad/triple/double/single/partial/wt_like."""
    if len(seq) < 155:
        return 'other'

    has_7V  = seq[6]  == 'V'
    has_99N = seq[98] == 'N'
    has_99T = seq[98] == 'T'

    # Cycloguanil pathway — checked before Pyr markers
    if has_7V and has_99N:
        return 'lethal'
    if has_7V:
        return 'cyc_double' if has_99T else 'cyc_single'

    # Pyrimethamine pathway
    mut42I  = seq[41]  == 'I'
    mut50R  = seq[49]  == 'R'
    mut155L = seq[154] == 'L'

    if mut42I and mut50R and has_99N and mut155L:
        return 'quad'
    if mut42I and mut50R and has_99N:
        return 'triple'
    if mut42I and has_99N:
        return 'double'
    if has_99N and not mut42I and not mut50R and not mut155L:
        return 'single'
    if mut42I or mut50R or mut155L:
        return 'partial'
    return 'wt_like'


def drug_state_for_generation(mode: str, gen: int, cycle_pattern: Tuple[int, int]) -> str:
    """Return 'on' or 'off' drug state for a given generation under none/on/off/cycle mode."""
    if mode == 'on':
        return 'on'
    if mode in ('off', 'none'):
        return 'off'
    if mode == 'cycle':
        on_gens, off_gens = cycle_pattern
        total = on_gens + off_gens
        if total <= 0:
            return 'off'
        if (gen % total) < on_gens:
            return 'on'
        return 'off'
    return 'off'


def haplotype_fitness_factor(
    hap: str,
    drug_state: str,
    single_cost_off: float,
    double_cost_off: float,
    triple_cost_off: float,
    quad_cost_off: float,
    single_adv_on: float,
    double_adv_on: float,
    triple_adv_on: float,
    quad_adv_on: float,
    cyc_drug_state: str = 'off',
    cyc_single_cost_off: float = 0.0,
    cyc_double_cost_off: float = 0.0,
    cyc_single_adv_on: float = 0.0,
    cyc_double_adv_on: float = 0.0,
) -> float:
    """Apply haplotype-specific Pyr and Cyc selection coefficients; returns multiplicative fitness factor."""
    if hap == 'lethal':
        return 0.0

    _pyr_costs = {
        'single': single_cost_off,
        'double': double_cost_off,
        'triple': triple_cost_off,
        'quad':   quad_cost_off,
    }
    _pyr_advs = {
        'single': single_adv_on,
        'double': double_adv_on,
        'triple': triple_adv_on,
        'quad':   quad_adv_on,
    }
    _cyc_costs = {'cyc_single': cyc_single_cost_off, 'cyc_double': cyc_double_cost_off}
    _cyc_advs  = {'cyc_single': cyc_single_adv_on,   'cyc_double': cyc_double_adv_on}

    # Pyrimethamine factor (only for Pyr haplotypes)
    pyr_factor = 1.0
    if hap in _pyr_costs:
        if drug_state == 'off':
            pyr_factor = max(1.0 - _pyr_costs[hap], 0.0)
        elif drug_state == 'on':
            pyr_factor = 1.0 + _pyr_advs.get(hap, 0.0)

    # Cycloguanil factor (only for Cyc haplotypes)
    cyc_factor = 1.0
    if hap in _cyc_costs:
        if cyc_drug_state == 'off':
            cyc_factor = max(1.0 - _cyc_costs[hap], 0.0)
        elif cyc_drug_state == 'on':
            cyc_factor = 1.0 + _cyc_advs.get(hap, 0.0)

    return pyr_factor * cyc_factor


# ========================= Mutation & simulation =========================

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def mutate(seq: str, mu: float, alphabet=AMINO_ACIDS) -> str:
    """Per-site point mutation with probability mu."""
    chars = list(seq)
    for i, aa in enumerate(chars):
        if random.random() < mu:
            choices = [a for a in alphabet if a != aa]
            chars[i] = random.choice(choices)
    return ''.join(chars)


def run_population(
    model: FitnessModel,
    wt: str,
    pop: int,
    gens: int,
    mu: float,
    out_dir: str,
    key_positions: List[int],
    drug_mode: str,
    drug_cycle: Tuple[int, int],
    single_cost_off: float,
    double_cost_off: float,
    triple_cost_off: float,
    quad_cost_off: float,
    single_adv_on: float,
    double_adv_on: float,
    triple_adv_on: float,
    quad_adv_on: float,
    drug_mode_cyc: str = 'none',
    drug_cycle_cyc: Tuple[int, int] = (3, 1),
    cyc_single_cost_off: float = 0.0,
    cyc_double_cost_off: float = 0.0,
    cyc_single_adv_on: float = 0.0,
    cyc_double_adv_on: float = 0.0,
    log_every: int = 100,
) -> None:
    """Wright–Fisher simulation; writes freqs.csv and haplotypes.csv to out_dir."""
    population = [wt] * pop

    freq_path  = os.path.join(out_dir, 'freqs.csv')
    haplo_path = os.path.join(out_dir, 'haplotypes.csv')

    with open(freq_path, 'w') as f_freq, open(haplo_path, 'w') as f_hap:
        f_freq.write("generation,mut_position,count\n")
        f_hap.write("generation,wt_like,single,double,triple,quad,partial,"
                    "cyc_single,cyc_double,lethal,other\n")

        pbar = tqdm(range(gens), desc="  WF simulation", unit="gen",
                    bar_format="{l_bar}{bar}| gen {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                    file=sys.stdout, dynamic_ncols=True)

        for g in pbar:
            mutated_pop = [mutate(seq, mu) for seq in population]
            drug_state     = drug_state_for_generation(drug_mode,     g, drug_cycle)
            cyc_drug_state = drug_state_for_generation(drug_mode_cyc, g, drug_cycle_cyc)

            unique_seqs  = list(set(mutated_pop))
            final_fitness = {}

            for seq in unique_seqs:
                base = model.compute_fitness(seq)
                if not np.isfinite(base) or base <= 0.0:
                    base = 1e-12

                hap    = classify_pf_dhfr_haplotype(seq)
                factor = haplotype_fitness_factor(
                    hap, drug_state,
                    single_cost_off, double_cost_off,
                    triple_cost_off, quad_cost_off,
                    single_adv_on,  double_adv_on,
                    triple_adv_on,  quad_adv_on,
                    cyc_drug_state=cyc_drug_state,
                    cyc_single_cost_off=cyc_single_cost_off,
                    cyc_double_cost_off=cyc_double_cost_off,
                    cyc_single_adv_on=cyc_single_adv_on,
                    cyc_double_adv_on=cyc_double_adv_on,
                )

                val = base * factor
                final_fitness[seq] = val if (np.isfinite(val) and val > 0.0) else 1e-12

            # Wright–Fisher sampling
            weights = [final_fitness[s] for s in mutated_pop]
            total_w = sum(weights)
            probs   = [w / total_w for w in weights] if total_w > 0.0 else None
            population = random.choices(mutated_pop, probs, k=pop)

            if (g % log_every) == 0:
                pos_counts = Counter()
                for seq in population:
                    for p in key_positions:
                        if p <= len(seq) and p <= len(wt) and seq[p - 1] != wt[p - 1]:
                            pos_counts[p] += 1
                for p in key_positions:
                    f_freq.write(f"{g},{p},{pos_counts.get(p, 0)}\n")

                hap_counts = Counter(classify_pf_dhfr_haplotype(seq) for seq in population)
                f_hap.write(
                    f"{g},"
                    f"{hap_counts.get('wt_like',    0)},"
                    f"{hap_counts.get('single',     0)},"
                    f"{hap_counts.get('double',     0)},"
                    f"{hap_counts.get('triple',     0)},"
                    f"{hap_counts.get('quad',       0)},"
                    f"{hap_counts.get('partial',    0)},"
                    f"{hap_counts.get('cyc_single', 0)},"
                    f"{hap_counts.get('cyc_double', 0)},"
                    f"{hap_counts.get('lethal',     0)},"
                    f"{hap_counts.get('other',      0)}\n"
                )

                triple_f  = hap_counts.get('triple',     0) / pop
                quad_f    = hap_counts.get('quad',       0) / pop
                cyc_d_f   = hap_counts.get('cyc_double', 0) / pop
                pbar.set_postfix(
                    pyr=drug_state,
                    cyc=cyc_drug_state,
                    triple=f"{triple_f:.3f}",
                    quad=f"{quad_f:.3f}",
                    cyc2=f"{cyc_d_f:.3f}",
                )


# ========================= Adaptive walks =========================

def describe_mutations(wt: str, seq: str) -> str:
    """Return compact mutation string relative to WT, e.g. 'N42I+C50R+S99N'."""
    muts = []
    for i, (a, b) in enumerate(zip(wt, seq), start=1):
        if a != b:
            muts.append(f"{a}{i}{b}")
    return "+".join(muts) if muts else "WT"


def run_adaptive_walks(
    model: FitnessModel,
    wt: str,
    key_positions: List[int],
    out_dir: str,
    reps: int = 20,
    steps: int = 50,
    tries_per_step: int = 20
) -> None:
    """Run stochastic adaptive walks from each key position; writes walks.csv to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "walks.csv")

    wt_fit = model.compute_fitness(wt)

    total_walks = sum(
        min(reps, len([a for a in AMINO_ACIDS if a != wt[kp - 1]]))
        for kp in key_positions if 1 <= kp <= len(wt)
    )

    with open(out_path, "w") as f:
        # kp_present: whether the forced step-1 mutation is still in the sequence at this step
        f.write("key_position,rep,step,fitness,mutations,kp_present\n")

        walk_num = 0
        pbar = tqdm(total=total_walks, desc="  Adaptive walks", unit="walk",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} walks [{elapsed}<{remaining}]",
                    file=sys.stdout, dynamic_ncols=True)

        for kp in key_positions:
            if kp < 1 or kp > len(wt):
                continue

            wt_aa   = wt[kp - 1]
            alt_aas = [a for a in AMINO_ACIDS if a != wt_aa]
            n_reps  = min(reps, len(alt_aas))

            # Write WT baseline once per key position (not per rep — it's identical)
            f.write(f"{kp},0,0,{wt_fit:.6f},WT,False\n")

            for idx in range(n_reps):
                rep_id   = idx + 1
                start_aa = alt_aas[idx]

                walk_num += 1
                pbar.set_postfix(pos=kp, start=f"{wt_aa}{kp}{start_aa}", walk=walk_num)

                # Step 1: force mutation at kp
                seq_list = list(wt)
                seq_list[kp - 1] = start_aa
                current_seq = "".join(seq_list)
                current_fit = model.compute_fitness(current_seq)
                muts_str    = describe_mutations(wt, current_seq)
                f.write(f"{kp},{rep_id},1,{current_fit:.6f},{muts_str},True\n")

                # Steps 2..steps: stochastic hill-climb over single-mutant neighbours
                for step in range(2, steps + 1):
                    best_seq = current_seq
                    best_fit = current_fit

                    for _ in range(tries_per_step):
                        i        = random.randrange(len(wt))
                        aa_here  = current_seq[i]
                        cand_aa  = random.choice([a for a in AMINO_ACIDS if a != aa_here])

                        cand_list    = list(current_seq)
                        cand_list[i] = cand_aa
                        cand_seq     = "".join(cand_list)

                        f_cand = model.compute_fitness(cand_seq)
                        if f_cand > best_fit:
                            best_fit = f_cand
                            best_seq = cand_seq

                    current_seq = best_seq
                    current_fit = best_fit
                    muts_str    = describe_mutations(wt, current_seq)
                    kp_present  = current_seq[kp - 1] != wt_aa
                    f.write(f"{kp},{rep_id},{step},{current_fit:.6f},{muts_str},{kp_present}\n")

                pbar.update(1)

        pbar.close()

# ========================= Oracle / wet-lab loader =========================

def _parse_mutation_set(mut_str: str):
    """Parse 'N51I+C59R+S108N' → list of (pos, wt_aa, mut_aa); WT/empty returns []."""
    mut_str = str(mut_str).strip()
    if not mut_str:
        return []
    if mut_str.upper() in {"WT", "WILD-TYPE", "WILD TYPE"}:
        return []

    muts = []
    for tok in mut_str.replace(" ", "").split("+"):
        m = re.match(r"([A-Z])(\d+)([A-Z])", tok)
        if not m:
            continue
        wt_aa = m.group(1)
        pos = int(m.group(2))
        mut_aa = m.group(3)
        muts.append((pos, wt_aa, mut_aa))
    return muts


def _haplotype_from_mutations(muts) -> str:
    """Classify a mutation list into haplotype classes (shifted numbering); mirrors classify_pf_dhfr_haplotype."""
    if not muts:
        return "wt_like"

    pos2aa = {pos: mut_aa for (pos, _, mut_aa) in muts}

    has_7V   = pos2aa.get(7)   == "V"
    has_99N  = pos2aa.get(99)  == "N"
    has_99T  = pos2aa.get(99)  == "T"
    has_42I  = pos2aa.get(42)  == "I"
    has_50R  = pos2aa.get(50)  == "R"
    has_155L = pos2aa.get(155) == "L"

    # Cycloguanil pathway
    if has_7V and has_99N:
        return "lethal"
    if has_7V:
        return "cyc_double" if has_99T else "cyc_single"

    # Pyrimethamine pathway
    if has_42I and has_50R and has_99N and has_155L:
        return "quad"
    if has_42I and has_50R and has_99N:
        return "triple"
    if has_42I and has_99N:
        return "double"
    if has_99N and not has_42I and not has_50R and not has_155L:
        return "single"
    return "other"

def load_oracle(path: Optional[str], mode: str = "combo"):
    """Load wet-lab oracle CSV; derives Pyr and Cyc drug-selection parameters (ki/eff/combo modes)."""
    if not path:
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        warn(f"Failed to read oracle file {path}: {e}")
        return None

    required = {"mutations", "F_Ki_Pyr", "F_eff"}
    if not required.issubset(df.columns):
        warn(f"Oracle file missing required columns {required}; ignoring.")
        return None

    # ---- WT ----
    wt_row = df[df["mutations"].str.upper() == "WT"]
    if wt_row.empty:
        warn("Oracle: WT row not found; ignoring.")
        return None

    wt_row = wt_row.iloc[0]
    phi_wt  = float(wt_row["F_Ki_Pyr"])
    eff_wt  = float(wt_row["F_eff"])

    # classify all rows
    df = df.copy()
    df["__muts"] = df["mutations"].apply(_parse_mutation_set)
    df["__hap"]  = df["__muts"].apply(_haplotype_from_mutations)

    triple = df[df["__hap"] == "triple"]
    quad   = df[df["__hap"] == "quad"]

    if triple.empty or quad.empty:
        warn("Oracle: missing triple/quad rows; ignoring.")
        return None

    triple = triple.sort_values("F_Ki_Pyr", ascending=False).iloc[0]
    quad   = quad.sort_values("F_Ki_Pyr",   ascending=False).iloc[0]

    phi_triple = float(triple["F_Ki_Pyr"])
    phi_quad   = float(quad["F_Ki_Pyr"])
    eff_triple = float(triple["F_eff"])
    eff_quad   = float(quad["F_eff"])

    # Single and double — use oracle rows if present, else interpolate geometrically
    single_rows = df[df["__hap"] == "single"]
    double_rows = df[df["__hap"] == "double"]

    if not single_rows.empty:
        sr = single_rows.sort_values("F_Ki_Pyr", ascending=False).iloc[0]
        phi_single = float(sr["F_Ki_Pyr"])
        eff_single = float(sr["F_eff"])
    else:
        phi_single = max(phi_wt, phi_triple ** (1.0 / 3.0))
        eff_single = eff_wt
        warn("Oracle: no single-mutant row found; interpolating single-mutant parameters.")

    if not double_rows.empty:
        dr = double_rows.sort_values("F_Ki_Pyr", ascending=False).iloc[0]
        phi_double = float(dr["F_Ki_Pyr"])
        eff_double = float(dr["F_eff"])
    else:
        phi_double = max(phi_wt, phi_triple ** (2.0 / 3.0))
        eff_double = eff_wt
        warn("Oracle: no double-mutant row found; interpolating double-mutant parameters.")

    # Scaling factor for Ki advantage (log10-proportional, bounded)
    alpha = 0.2

    def adv_from_Ki(phi: float) -> float:
        return max(0.0, alpha * np.log10(max(phi / phi_wt, 1.0)))

    adv_single = adv_from_Ki(phi_single)
    adv_double = adv_from_Ki(phi_double)
    adv_triple = adv_from_Ki(phi_triple)
    adv_quad   = adv_from_Ki(phi_quad)

    def cost_from_eff(eff: float) -> float:
        return max(0.0, min(1.0, 1.0 - eff / eff_wt))

    cost_single = cost_from_eff(eff_single)
    cost_double = cost_from_eff(eff_double)
    cost_triple = cost_from_eff(eff_triple)
    cost_quad   = cost_from_eff(eff_quad)

    if mode == "ki":
        single_adv_on, double_adv_on  = adv_single,  adv_double
        triple_adv_on, quad_adv_on    = adv_triple,  adv_quad
        single_cost_off = double_cost_off = triple_cost_off = quad_cost_off = 0.0

    elif mode == "eff":
        single_adv_on = double_adv_on = triple_adv_on = quad_adv_on = 0.0
        single_cost_off = cost_single
        double_cost_off = cost_double
        triple_cost_off = cost_triple
        quad_cost_off   = cost_quad

    else:  # combo (recommended): 70% Ki advantage + 30% efficiency cost
        w_ki, w_eff     = 0.7, 0.3
        single_adv_on   = w_ki * adv_single
        double_adv_on   = w_ki * adv_double
        triple_adv_on   = w_ki * adv_triple
        quad_adv_on     = w_ki * adv_quad
        single_cost_off = w_eff * cost_single
        double_cost_off = w_eff * cost_double
        triple_cost_off = w_eff * cost_triple
        quad_cost_off   = w_eff * cost_quad

    print(
        f"[Oracle Pyr mode={mode}] "
        f"single_adv={single_adv_on:.4f}, double_adv={double_adv_on:.4f}, "
        f"triple_adv={triple_adv_on:.4f}, quad_adv={quad_adv_on:.4f} | "
        f"single_cost={single_cost_off:.4f}, double_cost={double_cost_off:.4f}, "
        f"triple_cost={triple_cost_off:.4f}, quad_cost={quad_cost_off:.4f}"
    )

    # ---- Cycloguanil pathway (A7V / A7V+S99T) ----
    cyc_single_adv_on    = 0.0
    cyc_double_adv_on    = 0.0
    cyc_single_cost_off  = 0.0
    cyc_double_cost_off  = 0.0

    if "F_Ki_Cyc" in df.columns:
        phi_wt_cyc = float(wt_row["F_Ki_Cyc"])

        def adv_from_Ki_cyc(phi: float) -> float:
            return max(0.0, alpha * np.log10(max(phi / phi_wt_cyc, 1.0)))

        cyc_s_rows = df[df["__hap"] == "cyc_single"]
        cyc_d_rows = df[df["__hap"] == "cyc_double"]

        if not cyc_s_rows.empty:
            csr = cyc_s_rows.sort_values("F_Ki_Cyc", ascending=False).iloc[0]
            phi_cyc_s = float(csr["F_Ki_Cyc"])
            eff_cyc_s = float(csr["F_eff"])
        else:
            phi_cyc_s = phi_wt_cyc
            eff_cyc_s = eff_wt
            warn("Oracle: no cyc_single (A7V) row found; cycloguanil pathway inactive.")

        if not cyc_d_rows.empty:
            cdr = cyc_d_rows.sort_values("F_Ki_Cyc", ascending=False).iloc[0]
            phi_cyc_d = float(cdr["F_Ki_Cyc"])
            eff_cyc_d = float(cdr["F_eff"])
        else:
            phi_cyc_d = phi_cyc_s
            eff_cyc_d = eff_cyc_s
            warn("Oracle: no cyc_double (A7V+S99T) row found; copying cyc_single.")

        adv_cyc_s  = adv_from_Ki_cyc(phi_cyc_s)
        adv_cyc_d  = adv_from_Ki_cyc(phi_cyc_d)
        cost_cyc_s = cost_from_eff(eff_cyc_s)
        cost_cyc_d = cost_from_eff(eff_cyc_d)

        if mode == "ki":
            cyc_single_adv_on = adv_cyc_s
            cyc_double_adv_on = adv_cyc_d
        elif mode == "eff":
            cyc_single_cost_off = cost_cyc_s
            cyc_double_cost_off = cost_cyc_d
        else:  # combo
            cyc_single_adv_on   = w_ki * adv_cyc_s
            cyc_double_adv_on   = w_ki * adv_cyc_d
            cyc_single_cost_off = w_eff * cost_cyc_s
            cyc_double_cost_off = w_eff * cost_cyc_d

        print(
            f"[Oracle Cyc mode={mode}] "
            f"cyc_single_adv={cyc_single_adv_on:.4f}, cyc_double_adv={cyc_double_adv_on:.4f} | "
            f"cyc_single_cost={cyc_single_cost_off:.4f}, cyc_double_cost={cyc_double_cost_off:.4f}"
        )

    return dict(
        single_adv_on        = single_adv_on,
        double_adv_on        = double_adv_on,
        triple_adv_on        = triple_adv_on,
        quad_adv_on          = quad_adv_on,
        single_cost_off      = single_cost_off,
        double_cost_off      = double_cost_off,
        triple_cost_off      = triple_cost_off,
        quad_cost_off        = quad_cost_off,
        cyc_single_adv_on    = cyc_single_adv_on,
        cyc_double_adv_on    = cyc_double_adv_on,
        cyc_single_cost_off  = cyc_single_cost_off,
        cyc_double_cost_off  = cyc_double_cost_off,
    )

# ========================= Simple plotting helpers =========================

def plot_population_outputs(out_dir: str, pop: int, key_positions: List[int]) -> None:
    """Plot allele and haplotype frequency trajectories from freqs.csv and haplotypes.csv."""
    freq_path = os.path.join(out_dir, 'freqs.csv')
    haplo_path = os.path.join(out_dir, 'haplotypes.csv')

    if os.path.exists(freq_path):
        df_f = pd.read_csv(freq_path)
        plt.figure(figsize=(7, 6))
        for p in key_positions:
            sub = df_f[df_f['mut_position'] == p]
            if sub.empty:
                continue
            freq = sub['count'] / float(pop)
            plt.plot(sub['generation'], freq, marker='o', label=f"pos {p}")
        plt.xlabel("Generation", fontsize=14)
        plt.ylabel("Frequency (mutated)", fontsize=14)
        plt.title("Key-position mutation frequencies", fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "freqs_plot.png"), dpi=300)
        plt.close()

    if os.path.exists(haplo_path):
        df_h = pd.read_csv(haplo_path)
        plt.figure(figsize=(7, 6))
        _hap_cols = ['wt_like', 'single', 'double', 'triple', 'quad', 'partial',
                     'cyc_single', 'cyc_double', 'lethal', 'other']
        for col in _hap_cols:
            if col in df_h.columns:
                freq = df_h[col] / float(pop)
                plt.plot(df_h['generation'], freq, marker='o', label=col)
        plt.xlabel("Generation", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title("Haplotype frequencies", fontsize=16)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "haplotypes_plot.png"), dpi=300)
        plt.close()


def _pos_from_token(tok: str) -> Optional[int]:
    """Extract residue index from a mutation token like 'A7S' or 'N42Y'."""
    m = re.match(r"[A-Z](\d+)[A-Z]", tok)
    if m:
        return int(m.group(1))
    return None

        
def plot_walks_outputs(out_dir: str) -> None:
    """Plot adaptive-walk fitness trajectories per key position from walks.csv."""
    path = os.path.join(out_dir, "walks.csv")
    if not os.path.exists(path):
        return

    df = pd.read_csv(path)
    # Exclude the per-position WT baseline row (rep == 0) from trajectory plots
    df_walks = df[df['rep'] != 0].copy()

    for kp in sorted(df_walks['key_position'].unique()):
        sub = df_walks[df_walks['key_position'] == kp]

        plt.figure(figsize=(7, 7))

        reps = sorted(sub['rep'].unique())
        for idx, rep in enumerate(reps):
            srep = sub[sub['rep'] == rep].sort_values('step')

            steps   = srep['step'].values
            fit_log = np.log10(np.clip(srep['fitness'].values, 1e-12, None))

            # Start mutation label from step 1
            start_row     = srep[srep['step'] == 1].iloc[0]
            start_mut_str = start_row['mutations']
            start_token   = start_mut_str.split('+')[0] if start_mut_str not in ('WT', '') else "WT"

            # Final step: pick the last mutation NOT at the key position for the label
            final_row     = srep.iloc[-1]
            final_mut_str = final_row['mutations']
            tokens        = [t for t in final_mut_str.split('+') if t]

            kp_int    = int(kp)
            end_token = tokens[-1] if tokens else start_token
            for t in reversed(tokens):
                pos_t = _pos_from_token(t)
                if pos_t is not None and pos_t != kp_int:
                    end_token = t
                    break

            label = f"{start_token} → {end_token}"

            c = TAB20[idx % len(TAB20)]
            plt.plot(
                steps,
                fit_log,
                marker="o",
                markersize=9,
                markerfacecolor=c,
                markeredgecolor="black",
                markeredgewidth=0.8,
                linewidth=2.5,
                color=c,
                label=label
            )

        plt.xlabel("Mutation step", fontsize=16)
        plt.ylabel("Fitness (log10)", fontsize=16)
        plt.title(f"Stochastic bundle — position {kp}", fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

# ---------------- LEGEND (clean 2-column version) ----------------
        # Extract handles + labels BEFORE creating legend
        handles, labels = plt.gca().get_legend_handles_labels()

        # Create inside-plot legend
        leg = plt.legend(
            handles,
            labels,
            fontsize=12,
            loc="lower right",
            ncol=2,                 # <<<<<<<<<<  TWO columns
            frameon=True,
            framealpha=0.9,
            borderpad=0.6,
            labelspacing=0.4,
        )

        # Increase marker & line size inside legend
        for h in leg.legend_handles:
            try:
                h.set_markersize(10)
                h.set_linewidth(2.5)
            except:
                pass

        # leave space at bottom for legend
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        plt.savefig(os.path.join(out_dir, f"walks_pos{kp}.png"), dpi=300)
        plt.close()


# ========================= CLI =========================

def parse_args():
    p = argparse.ArgumentParser(
        description="PfPATH: PfDHFR evolution simulator with DMS + ddG + MI + DCCM + drug selection."
    )
    p.add_argument('--wt', required=True, help="WT PfDHFR FASTA file.")
    p.add_argument('--dms', help="DMS epistasis single mutant CSV.")
    p.add_argument('--thermo', help="SPIRED or similar ddG CSV with 'mutant' and 'ddG'.")
    p.add_argument('--ddg_s', help="Rosetta singles ddG CSV.")
    p.add_argument('--ddg_combo', nargs='*', help="Rosetta combo ddG CSVs (double/triple/quad).")
    p.add_argument('--mi', help="Co-evolution CSV with Res1, Res2, MI.")
    p.add_argument('--dccm', help="DCCM matrix CSV.")
    p.add_argument('--blosum', help="Position-specific BLOSUM CSV (Position, WT_AA, A..Y). "
                   "Covers all positions including DMS-dark ones; normalized like other terms.")
    p.add_argument('--pocket', help="Binding-pocket weight CSV (pos, role, weight). "
                   "Mutations at pocket positions get their S1 score multiplied by weight. "
                   "Roles: drug / substrate / both / structural.")
    p.add_argument('--key_positions', type=str, default='7,41,42,50,99,155',
                   help="Comma-separated list of key positions to track (default: 7,41,42,50,99,155).")

    # Population-genetic parameters
    p.add_argument('--pop', type=int, default=10000,
                   help="Population size (default 10000).")
    p.add_argument('--gens', type=int, default=20000,
                   help="Number of generations (default 20000).")
    p.add_argument('--mu', type=float, default=1e-5,
                   help="Per-site mutation rate (default 1e-5).")

    p.add_argument('--out', default=None,
                   help="Output directory. Defaults to 'runs/pfpath_run_YYYYMMDD_HHMMSS' "
                        "inside the current working directory.")

    # Weights for different components
    p.add_argument('--w_dms',     type=float, default=1.0)
    p.add_argument('--w_thermo',  type=float, default=1.0)
    p.add_argument('--w_ddg',     type=float, default=1.0)
    p.add_argument('--w_ddg_pair',type=float, default=1.0)
    p.add_argument('--w_blosum',  type=float, default=0.3,
                   help="Weight for BLOSUM substitution score term (default 0.3). "
                        "Only active when --blosum file is also provided.")
    p.add_argument('--w_mi',      type=float, default=0.5)
    p.add_argument('--w_dccm',    type=float, default=0.5)
    p.add_argument('--w_count',   type=float, default=0.05,
                   help="Mutation-load penalty coefficient (quadratic, default 0.05).")
    p.add_argument('--w_unknown', type=float, default=0.1,
                   help="Penalty per mutation absent from all data sources (default 0.1).")
    p.add_argument('--w_lone',    type=float, default=0.05,
                   help="Lone-pair epistatic penalty coefficient per MI unit (default 0.05).")
    p.add_argument('--mi_lone_threshold', type=float, default=0.5,
                   help="MI threshold above which lone-pair penalty applies (default 0.5).")
    p.add_argument('--max_mut_load', type=int, default=12,
                   help="Hard cap on number of simultaneous mutations (default 12).")
    p.add_argument('--beta', type=float, default=2.0,
                   help="Selection strength parameter β (default 2.0).")

    # Stability cutoff parameters
    p.add_argument('--ddg_cutoff', type=float, default=4.0,
                   help="SPIRED ddG cutoff (kcal/mol) for near-lethal mutants (default 4.0).")
    p.add_argument('--ddg_soft_start', type=float, default=2.0,
                   help="Soft stability penalty starts at this ddG (default 2.0).")

    # Drug selection mode and haplotype fitness
    p.add_argument('--oracle_mode', choices=['ki', 'eff', 'combo'],
                    default='combo',
                    help="How to use oracle_wetlab.csv: "
                            "ki = only resistance (F_Ki_Pyr), "
                            "eff = only catalytic efficiency (F_eff), "
                            "combo = weighted mix (default).")
    p.add_argument('--oracle',
                   help="Wet-lab oracle CSV (mutations, F_Ki_Pyr, F_eff) to derive drug selection.")
    p.add_argument('--drug_mode', choices=['none', 'on', 'off', 'cycle'],
                   default='none',
                   help="Pyrimethamine drug selection mode: none, on, off, cycle (default none).")
    p.add_argument('--drug_cycle', type=str, default='3,1',
                   help="Pyr cycle pattern 'on,off' generations for drug_mode=cycle (default '3,1').")
    p.add_argument('--drug_mode_cyc', choices=['none', 'on', 'off', 'cycle'],
                   default='none',
                   help="Cycloguanil drug selection mode: none, on, off, cycle (default none).")
    p.add_argument('--drug_cycle_cyc', type=str, default='3,1',
                   help="Cyc cycle pattern 'on,off' generations for drug_mode_cyc=cycle (default '3,1').")
    p.add_argument('--single_cost_off', type=float, default=0.00,
                   help="Fitness cost of single mutant off-drug (default 0.00).")
    p.add_argument('--double_cost_off', type=float, default=0.01,
                   help="Fitness cost of double mutant off-drug (default 0.01).")
    p.add_argument('--triple_cost_off', type=float, default=0.02,
                   help="Fitness cost of triple mutant off-drug (default 0.02).")
    p.add_argument('--quad_cost_off',   type=float, default=0.10,
                   help="Fitness cost of quadruple mutant off-drug (default 0.10).")
    p.add_argument('--single_adv_on',   type=float, default=0.02,
                   help="Fitness advantage of single mutant on-drug (default 0.02).")
    p.add_argument('--double_adv_on',   type=float, default=0.04,
                   help="Fitness advantage of double mutant on-drug (default 0.04).")
    p.add_argument('--triple_adv_on',   type=float, default=0.08,
                   help="Fitness advantage of triple mutant on-drug (default 0.08).")
    p.add_argument('--quad_adv_on',     type=float, default=0.12,
                   help="Fitness advantage of quadruple mutant on-drug (default 0.12).")
    p.add_argument('--cyc_single_cost_off', type=float, default=0.30,
                   help="Fitness cost of A7V (cyc_single) off-drug (default 0.30).")
    p.add_argument('--cyc_double_cost_off', type=float, default=0.27,
                   help="Fitness cost of A7V+S99T (cyc_double) off-drug (default 0.27).")
    p.add_argument('--cyc_single_adv_on',   type=float, default=0.33,
                   help="Fitness advantage of A7V (cyc_single) on cycloguanil (default 0.33).")
    p.add_argument('--cyc_double_adv_on',   type=float, default=0.41,
                   help="Fitness advantage of A7V+S99T (cyc_double) on cycloguanil (default 0.41).")

    p.add_argument('--log_every', type=int, default=100,
                   help="Log every N generations (default 100).")
    p.add_argument('--wf_oracle_only', action='store_true',
                   help="Population mode only: zero out DMS/thermo/MI weights so that only the "
                        "stability gate and oracle-based haplotype factors drive fitness. "
                        "Recommended for resistance evolution WF runs (avoids the z-score "
                        "normalization artifact that makes ~58%% of random mutations appear "
                        "beneficial, which prevents resistance sweeps from being tracked).")
    p.add_argument('--wf_purify', type=float, default=0.02,
                   help="Per-mutation purifying selection cost used with --wf_oracle_only "
                        "(default 0.02). Prevents neutral sequence drift that would cause "
                        "every individual to have a unique sequence, slowing computation. "
                        "Each mutation reduces log-fitness by this amount; 0.02 gives ~4%% "
                        "cost per mutation, keeping resistance sweeps dominant.")

    # Adaptive-walk mode
    p.add_argument('--walks_only', action='store_true',
                   help="Run adaptive walks only — skip population simulation.")
    p.add_argument('--pop_only', action='store_true',
                   help="Run population simulation only — skip adaptive walks.")
    p.add_argument('--walks', action='store_true',
                   help="Deprecated alias for --walks_only (kept for backwards compatibility).")
    p.add_argument('--walk_reps', type=int, default=19,
                   help="Number of replicate walks per key position (max 19 unique substitutions).")
    p.add_argument('--walk_steps', type=int, default=50,
                   help="Number of mutation steps per walk (default 50).")
    p.add_argument('--walk_tries', type=int, default=100,
                   help="Number of random candidates per step in walks (default 100).")
    p.add_argument('--seed', type=int, default=None,
                   help="Random seed for reproducibility (default: unseeded).")

    # Plotting
    p.add_argument('--plot', action='store_true',
                   help="If set, generate PNG plots from CSV outputs.")

    return p.parse_args()


def main():
    import time as _time
    _t0 = _time.time()

    args = parse_args()
    random.seed(args.seed)

    # Parse drug cycle patterns
    try:
        on_g, off_g = (int(x) for x in args.drug_cycle.split(','))
    except Exception:
        warn(f"Could not parse --drug_cycle '{args.drug_cycle}', defaulting to 3,1.")
        on_g, off_g = 3, 1

    try:
        on_g_cyc, off_g_cyc = (int(x) for x in args.drug_cycle_cyc.split(','))
    except Exception:
        warn(f"Could not parse --drug_cycle_cyc '{args.drug_cycle_cyc}', defaulting to 3,1.")
        on_g_cyc, off_g_cyc = 3, 1

    # ── Startup banner ────────────────────────────────────────────────────────
    from datetime import datetime as _dt
    print("\n" + "="*60)
    print("  PfPATH — PfDHFR Adaptive Evolution Simulator")
    print("="*60)
    print(f"  Started      : {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  WT sequence  : {args.wt}")
    print(f"  Output dir   : {args.out or 'auto-timestamped'}")
    _walks_only = getattr(args, 'walks_only', False) or getattr(args, 'walks', False)
    _pop_only   = getattr(args, 'pop_only', False)
    _mode_str   = ('walks only' if _walks_only else 'pop/WF only' if _pop_only else 'WF population + adaptive walks')
    print(f"  Mode         : {_mode_str}")
    print(f"  β (selection): {args.beta}")
    print(f"  Population N : {args.pop:,}")
    print(f"  Generations  : {args.gens:,}")
    print(f"  Mutation rate: {args.mu:.0e}")
    print(f"  Drug mode    : {args.drug_mode}  (cycle: {args.drug_cycle})")
    print(f"  Oracle file  : {args.oracle or 'none'}  (mode: {args.oracle_mode})")
    print(f"  Key positions: {args.key_positions}")
    print("="*60)

    # ── Data loading (with feedback) ─────────────────────────────────────────
    print("\n[1/2] Loading input data...")
    print(f"  WT fasta     ...", end=" ", flush=True)
    wt = load_wt_fasta(args.wt)
    print(f"done  ({len(wt)} residues)")

    print(f"  DMS scores   ...", end=" ", flush=True)
    dms = load_dms(args.dms)
    print(f"done  ({len(dms):,} entries)" if dms else "skipped")

    print(f"  Thermo (SPIRED)...", end=" ", flush=True)
    thermo_norm, thermo_raw = load_thermo(args.thermo)
    print(f"done  ({len(thermo_norm):,} entries)" if thermo_norm else "skipped")

    print(f"  ddG single   ...", end=" ", flush=True)
    ddg_s = load_ddg_single(args.ddg_s)
    print(f"done  ({len(ddg_s):,} entries)" if ddg_s else "skipped")

    print(f"  ddG combos   ...", end=" ", flush=True)
    ddg_c = load_ddg_combos(args.ddg_combo)
    print(f"done  ({len(ddg_c):,} entries)" if ddg_c else "skipped")

    print(f"  MI coupling  ...", end=" ", flush=True)
    mi = load_mi(args.mi)
    print(f"done  ({len(mi):,} pairs)" if mi else "skipped")

    print(f"  DCCM matrix  ...", end=" ", flush=True)
    dccm = load_dccm(args.dccm)
    print(f"done  ({len(dccm):,} pairs)" if dccm else "skipped")

    print(f"  BLOSUM       ...", end=" ", flush=True)
    blosum = load_blosum(getattr(args, 'blosum', None))
    print(f"done  ({len(blosum):,} entries)" if blosum else "skipped")

    pocket_weights = load_pocket(getattr(args, 'pocket', None))

    # ── Oracle calibration ────────────────────────────────────────────────────
    print(f"  Oracle/wetlab...", end=" ", flush=True)
    oracle_params = load_oracle(args.oracle, args.oracle_mode)
    print("done" if oracle_params else "skipped")

    if oracle_params is not None:
        single_adv_on       = oracle_params["single_adv_on"]
        double_adv_on       = oracle_params["double_adv_on"]
        triple_adv_on       = oracle_params["triple_adv_on"]
        quad_adv_on         = oracle_params["quad_adv_on"]
        single_cost_off     = oracle_params["single_cost_off"]
        double_cost_off     = oracle_params["double_cost_off"]
        triple_cost_off     = oracle_params["triple_cost_off"]
        quad_cost_off       = oracle_params["quad_cost_off"]
        cyc_single_adv_on   = oracle_params["cyc_single_adv_on"]
        cyc_double_adv_on   = oracle_params["cyc_double_adv_on"]
        cyc_single_cost_off = oracle_params["cyc_single_cost_off"]
        cyc_double_cost_off = oracle_params["cyc_double_cost_off"]
    else:
        single_adv_on       = args.single_adv_on
        double_adv_on       = args.double_adv_on
        triple_adv_on       = args.triple_adv_on
        quad_adv_on         = args.quad_adv_on
        single_cost_off     = args.single_cost_off
        double_cost_off     = args.double_cost_off
        triple_cost_off     = args.triple_cost_off
        quad_cost_off       = args.quad_cost_off
        cyc_single_adv_on   = args.cyc_single_adv_on
        cyc_double_adv_on   = args.cyc_double_adv_on
        cyc_single_cost_off = args.cyc_single_cost_off
        cyc_double_cost_off = args.cyc_double_cost_off

    # Resolve which modes to run first — needed for oracle-only guard below.
    walks_only = args.walks_only or args.walks
    pop_only   = args.pop_only
    run_pop    = not walks_only
    run_walks  = not pop_only

    # Walks always use the full fitness landscape.
    walk_weights = {
        'w_dms':      args.w_dms,
        'w_thermo':   args.w_thermo,
        'w_ddg':      args.w_ddg,
        'w_ddg_pair': args.w_ddg_pair,
        'w_mi':       args.w_mi,
        'w_dccm':     args.w_dccm,
        'w_blosum':   args.w_blosum,
        'w_count':    args.w_count,
        'w_unknown':  args.w_unknown,
        'w_lone':     args.w_lone,
        'w_purify':   0.0,
        'max_mut_load': args.max_mut_load,
        'beta':       args.beta,
    }

    # Population uses oracle-only zeroed weights when --wf_oracle_only is set.
    # This bypasses the DMS z-score normalisation artefact (which makes ~58% of
    # random mutations appear beneficial and prevents resistance sweeps from being
    # tracked). Walks are unaffected — they always get the full landscape above.
    if getattr(args, 'wf_oracle_only', False) and run_pop:
        pop_weights = {
            'w_dms': 0.0, 'w_thermo': 0.0, 'w_ddg': 0.0, 'w_ddg_pair': 0.0,
            'w_mi':  0.0, 'w_dccm':   0.0, 'w_blosum': 0.0,
            'w_count': args.w_count,
            'w_unknown': 0.0, 'w_lone': 0.0,
            'w_purify': args.wf_purify,
            'max_mut_load': args.max_mut_load,
            'beta': args.beta,
        }
        print(f"[WF oracle-only] Population: landscape zeroed, "
              f"purifying cost={args.wf_purify:.3f}. Walks: full landscape.")
    else:
        pop_weights = walk_weights

    _model_kwargs = dict(
        wt_seq=wt, dms=dms, thermo_norm=thermo_norm, thermo_raw=thermo_raw,
        ddg_single=ddg_s, ddg_highorder=ddg_c, mi=mi, dccm=dccm,
        ddg_cutoff_kcal=args.ddg_cutoff, ddg_soft_start=args.ddg_soft_start,
        mi_lone_threshold=args.mi_lone_threshold,
        blosum=blosum, pocket_weights=pocket_weights,
    )
    walk_model = FitnessModel(weights=walk_weights, **_model_kwargs)
    pop_model  = FitnessModel(weights=pop_weights,  **_model_kwargs) \
                 if pop_weights is not walk_weights else walk_model

    # Parse key positions
    try:
        key_positions = [int(x) for x in args.key_positions.split(',') if x.strip()]
    except ValueError:
        raise ValueError(f"Could not parse --key_positions '{args.key_positions}'. "
                         "Use a comma-separated list, e.g. '41,42,50,99,155'.")

    # Auto-name output directory if not specified
    if args.out is None:
        from datetime import datetime
        stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("runs", f"pfpath_run_{stamp}")
    else:
        out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[2/2] Running simulations  →  {out_dir}")
    print("-"*60)

    # run_pop / run_walks already set above (needed for oracle-only guard).

    if run_walks:
        run_adaptive_walks(
            model=walk_model,
            wt=wt,
            key_positions=key_positions,
            out_dir=out_dir,
            reps=args.walk_reps,
            steps=args.walk_steps,
            tries_per_step=args.walk_tries
        )

    if run_pop:
        run_population(
            model=pop_model,
            wt=wt,
            pop=args.pop,
            gens=args.gens,
            mu=args.mu,
            out_dir=out_dir,
            key_positions=key_positions,
            drug_mode=args.drug_mode,
            drug_cycle=(on_g, off_g),
            single_cost_off=single_cost_off,
            double_cost_off=double_cost_off,
            triple_cost_off=triple_cost_off,
            quad_cost_off=quad_cost_off,
            single_adv_on=single_adv_on,
            double_adv_on=double_adv_on,
            triple_adv_on=triple_adv_on,
            quad_adv_on=quad_adv_on,
            drug_mode_cyc=args.drug_mode_cyc,
            drug_cycle_cyc=(on_g_cyc, off_g_cyc),
            cyc_single_cost_off=cyc_single_cost_off,
            cyc_double_cost_off=cyc_double_cost_off,
            cyc_single_adv_on=cyc_single_adv_on,
            cyc_double_adv_on=cyc_double_adv_on,
            log_every=args.log_every,
        )

    if args.plot:
        print("\nGenerating plots...")
        plot_population_outputs(out_dir, args.pop, key_positions)
        plot_walks_outputs(out_dir)

    # ── Completion summary ────────────────────────────────────────────────────
    _elapsed = _time.time() - _t0
    _h, _rem = divmod(int(_elapsed), 3600)
    _m, _s   = divmod(_rem, 60)
    print("\n" + "="*60)
    print("  PfPATH run complete")
    print("="*60)
    print(f"  Finished     : {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total time   : {_h:02d}h {_m:02d}m {_s:02d}s")
    print(f"  Output dir   : {out_dir}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
# End of pfpath_simulator.py


'''
python pfpath_simulator.py \
  --wt wt.fasta \
  --dms epistasis_single_mutant_matrix.csv \
  --thermo spired_thermo.csv \
  --ddg_s PfDHFR_singles-ddG.csv \
  --ddg_combo PfDHFR_doubles-ddG.csv PfDHFR_triples-ddG.csv PfDHFR_quads-ddG.csv \
  --mi co-evolution.csv \
  --dccm pfdhfr_dccm_matrix.csv \
  --walks \
  --key_positions 7,41,42,50,99,155 \
  --walk_reps 19 \
  --walk_steps 50 \
  --out pfpath_walks \
  --plot
  
  
python pfpath_simulator.py \
  --wt wt.fasta \
  --dms epistasis_single_mutant_matrix.csv \
  --thermo spired_thermo.csv \
  --ddg_s PfDHFR_singles-ddG.csv \
  --ddg_combo PfDHFR_doubles-ddG.csv PfDHFR_triples-ddG.csv PfDHFR_quads-ddG.csv \
  --mi co-evolution.csv \
  --dccm pfdhfr_dccm_matrix.csv \
  --oracle oracle_wetlab.csv \
  --oracle_mode combo \
  --pop 10000 \
  --gens 20000 \
  --mu 1e-5 \
  --drug_mode cycle \
  --drug_cycle 3,1 \
  --key_positions 7,41,42,50,99,155 \
  --out pfpath_run_combo \ #or pfpath_run_ki \ or pfpath_run_eff
  --plot

'''