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
- Adaptive walks around key positions    [--walks]
- Population (Wright–Fisher) simulation  [default]
"""

import argparse
import csv
import warnings
import random
import re
from collections import defaultdict, Counter
from typing import Dict, Tuple, List, Optional
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# colour palette for adaptive walks
TAB20 = list(plt.get_cmap("tab20").colors)


# ========================= Utility helpers =========================

def warn(msg: str) -> None:
    warnings.warn(msg)


def normalise_array(values: List[float]) -> np.ndarray:
    """Z-score normalise then squash with tanh(z/2) into [-1, 1]."""
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
    """Load DMS single-mutant fitness CSV (pos, subs, score). Returns normalised dict[(pos, aa)]."""
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
    """Load SPIRED ddG CSV (mutant, ddG). Returns (normalised, raw) dicts; sign-flipped so stabilising is positive."""
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
    """Parse Rosetta label e.g. N108I; offset -9 maps canonical → shifted PfDHFR numbering."""
    label = label.strip()
    pos = int(label[1:-1]) + offset
    aa = label[-1]
    return pos, aa


def load_ddg_single(path: Optional[str]) -> Dict[Tuple[int, str], float]:
    """Load Rosetta single-mutant ddG CSV (case_name, total_score). Returns normalised, sign-flipped dict[(pos, aa)]."""
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
    """Load Rosetta ddG CSVs for double/triple/quad combos. Returns normalised, sign-flipped dict keyed by frozenset."""
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
    """Load pairwise MI coevolution CSV (Res1, Res2, MI). Returns symmetric normalised dict[(i, j)]."""
    if not path:
        warn("MI file not provided; MI term will be omitted.")
        return {}

    data = {}
    vals = []
    with open(path) as f:
        reader = csv.DictReader(f)
        if 'Res1' not in reader.fieldnames or 'Res2' not in reader.fieldnames or 'MI' not in reader.fieldnames:
            warn("MI file missing 'Res1', 'Res2', or 'MI'; skipping MI.")
            return {}
        for row in reader:
            try:
                i = int(re.sub(r'\D', '', row['Res1']))
                j = int(re.sub(r'\D', '', row['Res2']))
                v = float(row['MI'])
            except Exception:
                continue
            data[(i, j)] = v
            data[(j, i)] = v
            vals.append(v)

    if not vals:
        warn("No valid MI entries found; skipping MI.")
        return {}

    norm_vals = normalise_array(vals)
    out = {}
    unique_pairs = []
    seen = set()
    for (i, j), _ in data.items():
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(key)

    for key, nv in zip(unique_pairs, norm_vals):
        i, j = key
        out[(i, j)] = nv
        out[(j, i)] = nv
    return out


def load_dccm(path: Optional[str]) -> Dict[Tuple[int, int], float]:
    """Load DCCM matrix CSV (header + row index = residue). Returns normalised off-diagonal dict[(i, j)]."""
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
    """Composite fitness model: F(seq) = exp(β(S1+S2+Λ)) × Φ_ΔΔG, integrating DMS, thermo, Rosetta ddG, MI, DCCM."""

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
        ddg_soft_start: Optional[float] = 2.0
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

        self.ddg_cutoff_kcal = ddg_cutoff_kcal
        self.ddg_soft_start = ddg_soft_start

    def _mutations(self, seq: str) -> List[Tuple[int, str]]:
        """Return (pos, aa) mutations relative to WT, 1-based."""
        muts = []
        for i, (a, b) in enumerate(zip(self.wt, seq)):
            if a != b:
                muts.append((i + 1, b))
        return muts

    def _stability_penalty_factor(self, muts: List[Tuple[int, str]]) -> float:
        """Soft-cutoff stability factor from SPIRED ddG; approaches zero above ddg_cutoff_kcal."""
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

        # Stability penalty
        stability_factor = self._stability_penalty_factor(muts)
        if stability_factor <= 1e-12:
            return 1e-12

        # Single-mutation contributions
        S1 = 0.0
        for (p, aa) in muts:
            S1 += (
                self.w['w_dms'] * self.dms.get((p, aa), 0.0) +
                self.w['w_thermo'] * self.thermo_norm.get((p, aa), 0.0) +
                self.w['w_ddg'] * self.ddg_s.get((p, aa), 0.0)
            )

        # Pairwise contributions
        S2 = 0.0
        for i, m1 in enumerate(muts):
            p1, aa1 = m1
            for m2 in muts[i + 1:]:
                p2, aa2 = m2
                key_pair = frozenset([m1, m2])
                S2 += self.w['w_ddg_pair'] * self.ddg_c.get(key_pair, 0.0)
                S2 += self.w['w_mi'] * self.mi.get((p1, p2), 0.0)
                S2 += self.w['w_dccm'] * self.dccm.get((p1, p2), 0.0)

        # Penalty for too many simultaneous mutations
        penalty = -self.w['w_count'] * max(0, len(muts) - 6)

        total_score = S1 + S2 + penalty
        if not np.isfinite(total_score):
            return 1e-12

        beta = self.w.get('beta', 2.0)
        x = beta * total_score
        x = float(np.clip(x, -50.0, 50.0))  # avoid overflow
        intrinsic = float(np.exp(x))
        intrinsic = max(intrinsic, 1e-12)

        fitness = intrinsic * stability_factor
        if not np.isfinite(fitness) or fitness <= 0.0:
            return 1e-12

        return fitness


# ========================= Drug / haplotype logic =========================

def classify_pf_dhfr_haplotype(seq: str) -> str:
    """Classify sequence into quad/triple/partial/wt_like using shifted numbering (canonical − 9)."""
    # Need length at least 155
    if len(seq) < 155:
        return 'other'

    mut42I  = (seq[42  - 1] == 'I')
    mut50R  = (seq[50  - 1] == 'R')
    mut99N  = (seq[99  - 1] == 'N')
    mut155L = (seq[155 - 1] == 'L')

    flags = [mut42I, mut50R, mut99N, mut155L]
    count = sum(flags)

    if all(flags):
        return 'quad'
    if mut42I and mut50R and mut99N and not mut155L:
        return 'triple'
    if count > 0:
        return 'partial'
    return 'wt_like'


def drug_state_for_generation(mode: str, gen: int, cycle_pattern: Tuple[int, int]) -> str:
    """Return 'on' or 'off' drug state for generation g given mode (none/on/off/cycle) and cycle pattern."""
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
    triple_cost_off: float,
    quad_cost_off: float,
    triple_adv_on: float,
    quad_adv_on: float
) -> float:
    """Apply per-haplotype selection coefficient given drug state."""
    if drug_state == 'off':
        if hap == 'triple':
            return max(1.0 - triple_cost_off, 0.0)
        if hap == 'quad':
            return max(1.0 - quad_cost_off, 0.0)
        return 1.0

    if drug_state == 'on':
        if hap == 'triple':
            return 1.0 + triple_adv_on
        if hap == 'quad':
            return 1.0 + quad_adv_on
        return 1.0

    return 1.0


# ========================= Mutation & simulation =========================

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def mutate(seq: str, mu: float, alphabet=AMINO_ACIDS) -> str:
    """Per-site uniform mutation with probability mu."""
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
    triple_cost_off: float,
    quad_cost_off: float,
    triple_adv_on: float,
    quad_adv_on: float,
    log_every: int = 100
) -> None:
    """Wright–Fisher simulation with selection. Writes freqs.csv and haplotypes.csv to out_dir."""
    population = [wt] * pop

    freq_path = os.path.join(out_dir, 'freqs.csv')
    haplo_path = os.path.join(out_dir, 'haplotypes.csv')

    with open(freq_path, 'w') as f_freq, open(haplo_path, 'w') as f_hap:
        f_freq.write("generation,mut_position,count\n")
        f_hap.write("generation,wt_like,triple,quad,partial,other\n")

        for g in range(gens):
            # Mutation step
            mutated_pop = [mutate(seq, mu) for seq in population]

            # Drug state
            drug_state = drug_state_for_generation(drug_mode, g, drug_cycle)

            # Unique genotypes for fitness caching
            unique_seqs = list(set(mutated_pop))
            final_fitness = {}

            for seq in unique_seqs:
                base = model.compute_fitness(seq)
                if not np.isfinite(base) or base <= 0.0:
                    base = 1e-12

                hap = classify_pf_dhfr_haplotype(seq)
                factor = haplotype_fitness_factor(
                    hap, drug_state,
                    triple_cost_off, quad_cost_off,
                    triple_adv_on, quad_adv_on
                )

                val = base * factor
                if not np.isfinite(val) or val <= 0.0:
                    val = 1e-12

                final_fitness[seq] = val

            # Reproduction (Wright–Fisher)
            weights = [final_fitness[s] for s in mutated_pop]
            total_w = sum(weights)
            if total_w <= 0.0:
                probs = None
            else:
                probs = [w / total_w for w in weights]

            population = random.choices(mutated_pop, probs, k=pop)

            # Logging
            if (g % log_every) == 0:
                # Position-level frequencies (always log all key positions)
                pos_counts = Counter()
                for seq in population:
                    for p in key_positions:
                        if p <= len(seq) and p <= len(wt) and seq[p - 1] != wt[p - 1]:
                            pos_counts[p] += 1
                for p in key_positions:
                    c = pos_counts.get(p, 0)
                    f_freq.write(f"{g},{p},{c}\n")

                # Haplotype frequencies
                hap_counts = Counter()
                for seq in population:
                    hap_counts[classify_pf_dhfr_haplotype(seq)] += 1

                wt_like = hap_counts.get('wt_like', 0)
                triple = hap_counts.get('triple', 0)
                quad = hap_counts.get('quad', 0)
                partial = hap_counts.get('partial', 0)
                other = hap_counts.get('other', 0)
                f_hap.write(f"{g},{wt_like},{triple},{quad},{partial},{other}\n")


# ========================= Adaptive walks =========================

def describe_mutations(wt: str, seq: str) -> str:
    """Return compact mutation string relative to WT, e.g. 'N51I+C59R+S108N'."""
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
    """Stochastic adaptive walks from WT, one bundle per key position. Writes walks.csv to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "walks.csv")

    with open(out_path, "w") as f:
        f.write("key_position,rep,step,fitness,sequence,mutations\n")

        for kp in key_positions:
            if kp < 1 or kp > len(wt):
                continue

            wt_aa = wt[kp - 1]
            alt_aas = [a for a in AMINO_ACIDS if a != wt_aa]  # 19 AAs
            # at most 19 unique starting substitutions
            n_reps = min(reps, len(alt_aas))

            for idx in range(n_reps):
                rep_id = idx + 1
                start_aa = alt_aas[idx]

                # Start from WT
                current_seq = wt
                current_fit = model.compute_fitness(current_seq)
                muts_str = describe_mutations(wt, current_seq)
                f.write(f"{kp},{rep_id},0,{current_fit:.6f},{current_seq},{muts_str}\n")

                # Step 1: force mutation at kp to a specific alternative amino acid
                seq_list = list(current_seq)
                seq_list[kp - 1] = start_aa
                current_seq = "".join(seq_list)
                current_fit = model.compute_fitness(current_seq)
                muts_str = describe_mutations(wt, current_seq)
                f.write(f"{kp},{rep_id},1,{current_fit:.6f},{current_seq},{muts_str}\n")

                # Steps 2..steps: stochastic hill-climb
                for step in range(2, steps + 1):
                    best_seq = current_seq
                    best_fit = current_fit

                    # Try random single mutants and keep best
                    for _ in range(tries_per_step):
                        i = random.randrange(len(wt))
                        aa_here = current_seq[i]
                        cand_choices = [a for a in AMINO_ACIDS if a != aa_here]
                        cand_aa = random.choice(cand_choices)

                        cand_list = list(current_seq)
                        cand_list[i] = cand_aa
                        cand_seq = "".join(cand_list)

                        f_cand = model.compute_fitness(cand_seq)
                        if f_cand > best_fit:
                            best_fit = f_cand
                            best_seq = cand_seq

                    current_seq = best_seq
                    current_fit = best_fit

                    muts_str = describe_mutations(wt, current_seq)
                    f.write(f"{kp},{rep_id},{step},{current_fit:.6f},{current_seq},{muts_str}\n")

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
    """Classify mutation list into quad/triple/other using shifted numbering (canonical − 9)."""
    if not muts:
        return "wt_like"

    # map position -> mutant AA
    pos2aa = {pos: mut for (pos, wt, mut) in muts}

    has_42I  = (pos2aa.get(42)  == "I")
    has_50R  = (pos2aa.get(50)  == "R")
    has_99N  = (pos2aa.get(99)  == "N")
    has_155L = (pos2aa.get(155) == "L")

    if has_42I and has_50R and has_99N and has_155L:
        return "quad"
    if has_42I and has_50R and has_99N:
        return "triple"
    return "other"

def load_oracle_pyr(path: Optional[str], mode: str = "combo"):
    """Load wet-lab oracle CSV and derive triple/quad adv_on/cost_off (modes: ki, eff, combo)."""
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

    # Scaling factor for Ki advantage
    alpha = 0.2

    # ---- compute raw Ki-based advantage ----
    def adv_from_Ki(phi):
        return max(0.0, alpha * np.log10(max(phi / phi_wt, 1.0)))

    adv_triple = adv_from_Ki(phi_triple)
    adv_quad   = adv_from_Ki(phi_quad)

    # ---- compute raw efficiency costs ----
    cost_triple = max(0.0, min(1.0, 1.0 - eff_triple / eff_wt))
    cost_quad   = max(0.0, min(1.0, 1.0 - eff_quad   / eff_wt))

    # ===================================================================
    # MODE SWITCHING
    # ===================================================================
    if mode == "ki":
        triple_adv_on   = adv_triple
        quad_adv_on     = adv_quad
        triple_cost_off = 0.0
        quad_cost_off   = 0.0

    elif mode == "eff":
        triple_adv_on   = 0.0
        quad_adv_on     = 0.0
        triple_cost_off = cost_triple
        quad_cost_off   = cost_quad

    else:  # combo ← recommended
        w_ki  = 0.7
        w_eff = 0.3
        triple_adv_on   = w_ki * adv_triple
        quad_adv_on     = w_ki * adv_quad
        triple_cost_off = w_eff * cost_triple
        quad_cost_off   = w_eff * cost_quad

    print(
        f"[Oracle mode={mode}] "
        f"triple_adv_on={triple_adv_on:.4f}, quad_adv_on={quad_adv_on:.4f}, "
        f"triple_cost_off={triple_cost_off:.4f}, quad_cost_off={quad_cost_off:.4f}"
    )

    return dict(
        triple_adv_on   = triple_adv_on,
        quad_adv_on     = quad_adv_on,
        triple_cost_off = triple_cost_off,
        quad_cost_off   = quad_cost_off,
    )

# ========================= Simple plotting helpers =========================

def plot_population_outputs(out_dir: str, pop: int, key_positions: List[int]) -> None:
    """Plot allele-frequency and haplotype-frequency trajectories from freqs.csv and haplotypes.csv."""
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
        for col in ['wt_like', 'triple', 'quad', 'partial', 'other']:
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
    """Extract residue index from token like 'A7S'."""
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

    for kp in sorted(df['key_position'].unique()):
        sub = df[df['key_position'] == kp]

        plt.figure(figsize=(7, 7))

        reps = sorted(sub['rep'].unique())
        for idx, rep in enumerate(reps):
            srep = sub[sub['rep'] == rep].sort_values('step')

            steps = srep['step'].values
            # log10 fitness for y-axis
            fit_raw = srep['fitness'].values
            fit_log = np.log10(np.clip(fit_raw, 1e-12, None))

            # Start mutation at step 1
            start_row = srep[srep['step'] == 1].iloc[0]
            start_mut_str = start_row['mutations']
            start_token = start_mut_str.split('+')[0] if start_mut_str else "WT"

            # Final mutations string at last step
            final_row = srep.iloc[-1]
            final_mut_str = final_row['mutations']
            tokens = [t for t in final_mut_str.split('+') if t]

            # Choose last token that is NOT at the key position; if none, just last token
            kp_int = int(kp)
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
    p.add_argument('--key_positions', type=str, default='41,42,50,99,155',
                   help="Comma-separated list of key positions to track, e.g. '41,42,50,99,155'.")

    # Population-genetic parameters
    p.add_argument('--pop', type=int, default=10000,
                   help="Population size (default 10000).")
    p.add_argument('--gens', type=int, default=20000,
                   help="Number of generations (default 20000).")
    p.add_argument('--mu', type=float, default=1e-5,
                   help="Per-site mutation rate (default 1e-5).")

    p.add_argument('--out', default='pfpath_run',
                   help="Output directory (default 'pfpath_run').")

    # Weights for different components
    p.add_argument('--w_dms', type=float, default=1.0)
    p.add_argument('--w_thermo', type=float, default=1.0)
    p.add_argument('--w_ddg', type=float, default=1.0)
    p.add_argument('--w_ddg_pair', type=float, default=1.0)
    p.add_argument('--w_mi', type=float, default=0.5)
    p.add_argument('--w_dccm', type=float, default=0.5)
    p.add_argument('--w_count', type=float, default=0.05)
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
                   help="Drug selection mode: none, on, off, cycle (default none).")
    p.add_argument('--drug_cycle', type=str, default='3,1',
                   help="Cycle pattern 'on,off' generations for drug_mode=cycle (default '3,1').")
    p.add_argument('--triple_cost_off', type=float, default=0.02,
                   help="Fitness cost of triple mutant off-drug (default 0.02).")
    p.add_argument('--quad_cost_off', type=float, default=0.10,
                   help="Fitness cost of quadruple mutant off-drug (default 0.10).")
    p.add_argument('--triple_adv_on', type=float, default=0.08,
                   help="Fitness advantage of triple mutant on-drug (default 0.08).")
    p.add_argument('--quad_adv_on', type=float, default=0.12,
                   help="Fitness advantage of quadruple mutant on-drug (default 0.12).")

    p.add_argument('--log_every', type=int, default=100,
                   help="Log every N generations (default 100).")

    # Adaptive-walk mode
    p.add_argument('--walks', action='store_true',
                   help="If set, run adaptive walks instead of population simulation.")
    p.add_argument('--walk_reps', type=int, default=19,
                   help="Number of replicate walks per key position (max 19 unique substitutions).")
    p.add_argument('--walk_steps', type=int, default=50,
                   help="Number of mutation steps per walk (default 50).")
    p.add_argument('--walk_tries', type=int, default=20,
                   help="Number of random candidates per step in walks (default 20).")

    # Plotting
    p.add_argument('--plot', action='store_true',
                   help="If set, generate PNG plots from CSV outputs.")

    return p.parse_args()


def main():
    args = parse_args()
    random.seed()

    # Parse drug cycle pattern
    try:
        on_g, off_g = (int(x) for x in args.drug_cycle.split(','))
    except Exception:
        warn(f"Could not parse --drug_cycle '{args.drug_cycle}', defaulting to 3,1.")
        on_g, off_g = 3, 1

    wt = load_wt_fasta(args.wt)
    dms = load_dms(args.dms)
    thermo_norm, thermo_raw = load_thermo(args.thermo)
    ddg_s = load_ddg_single(args.ddg_s)
    ddg_c = load_ddg_combos(args.ddg_combo)
    mi = load_mi(args.mi)
    dccm = load_dccm(args.dccm)

    oracle_params = load_oracle_pyr(args.oracle, args.oracle_mode)

    if oracle_params is not None:
        triple_adv_on   = oracle_params["triple_adv_on"]
        quad_adv_on     = oracle_params["quad_adv_on"]
        triple_cost_off = oracle_params["triple_cost_off"]
        quad_cost_off   = oracle_params["quad_cost_off"]
    else:
        triple_adv_on   = args.triple_adv_on
        quad_adv_on     = args.quad_adv_on
        triple_cost_off = args.triple_cost_off
        quad_cost_off   = args.quad_cost_off

    weights = {
        'w_dms': args.w_dms,
        'w_thermo': args.w_thermo,
        'w_ddg': args.w_ddg,
        'w_ddg_pair': args.w_ddg_pair,
        'w_mi': args.w_mi,
        'w_dccm': args.w_dccm,
        'w_count': args.w_count,
        'beta': args.beta,
    }

    model = FitnessModel(
        wt_seq=wt,
        dms=dms,
        thermo_norm=thermo_norm,
        thermo_raw=thermo_raw,
        ddg_single=ddg_s,
        ddg_highorder=ddg_c,
        mi=mi,
        dccm=dccm,
        weights=weights,
        ddg_cutoff_kcal=args.ddg_cutoff,
        ddg_soft_start=args.ddg_soft_start
    )

    # Parse key positions
    try:
        key_positions = [int(x) for x in args.key_positions.split(',') if x.strip()]
    except ValueError:
        raise ValueError(f"Could not parse --key_positions '{args.key_positions}'. "
                         "Use a comma-separated list, e.g. '41,42,50,99,155'.")

    # Treat --out as directory
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    if args.walks:
        # Adaptive-walk mode
        run_adaptive_walks(
            model=model,
            wt=wt,
            key_positions=key_positions,
            out_dir=out_dir,
            reps=args.walk_reps,
            steps=args.walk_steps,
            tries_per_step=args.walk_tries
        )
    else:
        # Population mode
        run_population(
            model=model,
            wt=wt,
            pop=args.pop,
            gens=args.gens,
            mu=args.mu,
            out_dir=out_dir,
            key_positions=key_positions,
            drug_mode=args.drug_mode,
            drug_cycle=(on_g, off_g),
            triple_cost_off=triple_cost_off,
            quad_cost_off=quad_cost_off,
            triple_adv_on=triple_adv_on,
            quad_adv_on=quad_adv_on,
            log_every=args.log_every
        )

    if args.plot:
        # Plots will only be created for outputs that exist
        plot_population_outputs(out_dir, args.pop, key_positions)
        plot_walks_outputs(out_dir)


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