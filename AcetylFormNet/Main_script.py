"""
================================================================================
FormNet / ModFormNet — acetylation Experiment 2
ONE SCRIPT: MaxQuant -> formnet_sites -> exact combinatorial recovery -> figures
================================================================================

Colab, in one cell:

    !apt-get install -qq fonts-liberation > /dev/null
    import matplotlib.font_manager as fm; fm.fontManager.__init__()
    %run formnet_acetyl_all_in_one.py

Requires only numpy / pandas / scipy / matplotlib, all preinstalled in Colab.
NO cnsplots: it pulls scanpy, anndata, gseapy and lifelines, and resolving that
tree reinstalls numpy underneath the running kernel. The panel layout it
provided is reimplemented here in ~110 lines (section 1).

--------------------------------------------------------------------------------
INPUT — two entry points, auto-detected
--------------------------------------------------------------------------------
A. Raw, as in the notebook. Put in /content (or CWD):
       MaxQuant_output_files.zip     containing .../Experiment-2_txt/
       <one human>.fasta
   The script rebuilds formnet_sites from scratch and caches it to CSV.

B. Cached. If formnet_sites.csv exists it is loaded and the pipeline is skipped.
   Columns: protein_id, site_id, position, state_class, marginal
   state_class in {unmodified_0, partial, modified_unquantified, unmeasured_X}

--------------------------------------------------------------------------------
NOTATION  (fixed once, used in every printout and figure)
--------------------------------------------------------------------------------
    R      FASTA lysine coordinates in the protein group
    D      detected coordinates (any native evidence, modified or unmodified)
    B      boundary-fixed coordinates, marginal = 0
    M      interior coordinates, 0 < marginal < 1   (quantitative marginals)
    Qm     modified but not quantitatively resolved
    X      no evidence at all
    C      directly constrained coordinates = B + M
    U      unconstrained coordinates = Qm + X = R - C

    A(m)   the admissible support: which combinations the data allow
    P(m)   the feasible polytope: how much weight those combinations may carry

Two facts used throughout, both exact:
    |A(m)| over the measured projection = 2^M
    |A(m)| over all R lysines           = 2^(R-B) = 2^M x 2^U
    E[k] over the constrained coordinates = sum_j m_j     (FIXED, not bounded)

--------------------------------------------------------------------------------
OUTPUT
--------------------------------------------------------------------------------
    formnet_sites.csv
    formnet_protein_summary.csv
    formnet_configuration_bounds.csv
    formnet_grade_envelopes.csv
    formnet_grade_hierarchy.csv
    formnet_figure1_proteome.{pdf,png}
    formnet_figure2_<ACCESSION>.{pdf,png}   one per exemplar
    formnet_figure3_hierarchy.{pdf,png}
    formnet_figure4_<ACCESSION>.{pdf,png}   one per regime exemplar
    console report, sections 1-5
================================================================================
"""

from __future__ import annotations

import itertools
import os
import re
import warnings
import zipfile
from math import comb, log2
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import linprog

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================

ROOT = Path("/content") if Path("/content").exists() else Path(".")
ZIP_NAME = "MaxQuant_output_files.zip"
EXPERIMENT_DIR = "Experiment-2_txt"
SITES_CACHE = ROOT / "formnet_sites.csv"

# --- pipeline constants, exactly as in the notebook ---------------------------
C_CHEM = 0.1038                      # chemical-label channel fraction
D_MAP = {"1_10": 6.37, "1_100": 63.7, "1_1000": 637.0}
LOC_PROB_MIN = 0.75
MIN_DILUTIONS = 2
MAX_DILUTION_FOLD_RANGE = 2.0
STOICH_MAX = 0.10                    # quantitative ceiling retained

# --- analysis ----------------------------------------------------------------
MAX_M_LP = 14          # exact LP grade envelopes up to this many interior coords
MAX_M_ENUM = 18        # configuration enumeration ceiling
TOP_CONFIGS = 12       # rows in the configuration ladder panel

EXEMPLARS = ["Q09666", "Q15149", "P12270", "P35579"]
PROTEIN_NAMES = {"Q09666": "AHNAK", "Q15149": "PLEC",
                 "P12270": "TPR", "P35579": "MYH9"}

MAKE_FIGURES = True

# --- palette: one meaning per colour, held across every figure ----------------
C_ZERO    = "#3B6E8F"   # boundary-fixed:  support constraint
C_PARTIAL = "#C8562B"   # interior:        weight constraint
C_UNQ     = "#8A7A3D"   # modified, unquantified
C_UNMEAS  = "#E3E0D9"   # unmeasured
C_UNIQUE  = "#4E6E58"   # unique weighted solution
C_INK     = "#1A1A1A"
C_RULE    = "#8C8C8C"

STATE_COLORS = {"unmodified_0": C_ZERO, "partial": C_PARTIAL,
                "modified_unquantified": C_UNQ, "unmeasured_X": C_UNMEAS}
STATE_LABELS = {"unmodified_0": "Boundary-fixed (0)",
                "partial": "Quantitatively constrained",
                "modified_unquantified": "Modified, unquantified",
                "unmeasured_X": "Unmeasured"}
STATE_ORDER = ["unmodified_0", "partial", "modified_unquantified", "unmeasured_X"]


def rule(title="", char="=", width=80):
    if title:
        print(char * width)
        print(title)
        print(char * width)
    else:
        print(char * width)


def thou(x):
    return f"{int(round(x)):,}"


# ==============================================================================
# 1. PANEL LAYOUT  (replaces cnsplots; units are POINTS)
# ==============================================================================

PT_PER_INCH = 72.0


class MultiPanel:
    """540 pt = 7.5 in = 190 mm, a Nature double-column figure, so every
    fontsize in this file is a literal point size."""

    def __init__(self, max_width=540, label_size=8.5):
        self.max_width = float(max_width)
        self.label_size = label_size
        self.fig = plt.figure(figsize=(max_width / PT_PER_INCH, 1.0),
                              facecolor="white")
        self._axes, self._labels = [], []
        self._x = self._row_top = self._row_h = 0.0

    def panel(self, label=None, width=150, height=120, pad_left=30, pad_top=16,
              margin_right=0, margin_bottom=0, margin_left=0, margin_top=0):
        cw = margin_left + pad_left + width + margin_right
        ch = margin_top + pad_top + height + margin_bottom
        if self._x > 0 and self._x + cw > self.max_width + 1e-6:
            self._row_top += self._row_h
            self._x = self._row_h = 0.0
        x = self._x + margin_left + pad_left
        y = self._row_top + margin_top + pad_top
        ax = self.fig.add_axes([0, 0, 1, 1])
        self._axes.append((ax, x, y, float(width), float(height)))
        if label:
            self._labels.append((label, self._x + margin_left,
                                 self._row_top + margin_top))
        self._x += cw
        self._row_h = max(self._row_h, ch)
        plt.sca(ax)
        return ax

    def save(self, stem, dpi=600):
        W, H = self.max_width, max(self._row_top + self._row_h, 1.0)
        self.fig.set_size_inches(W / PT_PER_INCH, H / PT_PER_INCH)
        for ax, x, y, w, h in self._axes:
            ax.set_position([x / W, 1.0 - (y + h) / H, w / W, h / H])
        for txt, x, y in self._labels:
            self.fig.text(x / W, 1.0 - (y + self.label_size * 0.55) / H, txt,
                          fontsize=self.label_size, fontweight="bold",
                          color=C_INK, ha="left", va="top")
        self.fig.savefig(f"{stem}.pdf", dpi=dpi, facecolor="white")
        self.fig.savefig(f"{stem}.png", dpi=dpi, facecolor="white")
        plt.close(self.fig)
        print(f"    wrote {stem}.pdf / .png")


def setup_matplotlib():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def _bare(ax, left=True, bottom=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(length=2.5, width=0.6, labelsize=6)
    for s in ("left", "bottom"):
        if ax.spines[s].get_visible():
            ax.spines[s].set_linewidth(0.6)


# ==============================================================================
# 2. EXACT BOUND MATHEMATICS
# ==============================================================================

def frechet_configuration_bounds(m, site_labels, max_enumerate=MAX_M_ENUM):
    """
    Sharp bounds on the weight of every admissible combination over the
    interior coordinates.

    For a combination with ON-set S, let c_j = m_j for j in S and 1 - m_j
    otherwise. The Frechet-Hoeffding bounds give

        upper = min_j c_j
        lower = max(0, sum_j c_j - (M - 1))

    These are exact, and reduce to the familiar special cases:
        all-unmodified   [max(0, 1 - sum m),  1 - max m]
        single site j     [max(0, 2 m_j - sum m),  m_j]
        |S| >= 2          [0 (for small m),  min_{j in S} m_j]

    status:
        necessary  lower > 0   the data REQUIRE this combination to carry weight
        permitted  lower = 0   admissible, capped at upper
        excluded   upper = 0   removed from the support
    """
    m = np.asarray(m, float)
    M = len(m)
    if M == 0:
        return pd.DataFrame([{"configuration": "", "label": "no interior coordinate",
                              "k": 0, "lower": 1.0, "upper": 1.0,
                              "status": "necessary"}])
    if M > max_enumerate:
        return pd.DataFrame(columns=["configuration", "label", "k",
                                     "lower", "upper", "status"])

    rows = []
    for state in itertools.product([0, 1], repeat=M):
        c = np.where(np.array(state) == 1, m, 1.0 - m)
        upper = float(c.min())
        lower = max(0.0, float(c.sum() - (M - 1)))
        on = [site_labels[j] for j in range(M) if state[j]]
        rows.append({
            "configuration": "".join(map(str, state)),
            "label": " + ".join(on) if on else "no additional modification",
            "k": int(sum(state)),
            "lower": lower, "upper": upper,
            "status": "excluded" if upper <= 0 else
                      ("necessary" if lower > 1e-12 else "permitted"),
        })
    return (pd.DataFrame(rows)
            .sort_values(["upper", "k"], ascending=[False, True])
            .reset_index(drop=True))


def _polytope(m):
    M = len(m)
    S = np.array(list(itertools.product([0, 1], repeat=M)), float)
    A_eq = np.vstack([np.ones(len(S)), S.T])
    b_eq = np.concatenate([[1.0], np.asarray(m, float)])
    return S, A_eq, b_eq


def _lp(c, A_eq, b_eq, sense="min"):
    r = linprog(c if sense == "min" else -c, A_eq=A_eq, b_eq=b_eq,
                bounds=(0.0, 1.0), method="highs")
    if not r.success:
        return np.nan, None
    val = float(r.fun if sense == "min" else -r.fun)
    return val, r.x


def grade_envelope(m, max_lp=MAX_M_LP):
    """
    Exact interval [q_k_min, q_k_max] of weight the data permit on each grade
    k = ||x||_1, obtained by linear programming over P(m).

    Also returns the vertex attaining each extreme, which is used for the
    attained entropy range.

    Above max_lp interior coordinates the LP is skipped and the Markov
    relaxation q_k <= sum(m)/k is reported instead, flagged exact = False.
    """
    m = np.asarray(m, float)
    M = len(m)
    mu = float(m.sum())
    if M == 0:
        return pd.DataFrame([{"k": 0, "n_configurations": 1, "lower": 1.0,
                              "upper": 1.0, "exact": True}]), []

    if M > max_lp:
        rows = [{"k": 0, "n_configurations": 1,
                 "lower": max(0.0, 1.0 - mu), "upper": float(1.0 - m.max()),
                 "exact": True}]
        for k in range(1, M + 1):
            rows.append({"k": k, "n_configurations": comb(M, k), "lower": 0.0,
                         "upper": min(1.0, mu / k), "exact": False})
        return pd.DataFrame(rows), []

    S, A_eq, b_eq = _polytope(m)
    kvec = S.sum(axis=1)
    rows, verts = [], []
    for k in range(M + 1):
        c = (kvec == k).astype(float)
        lo, plo = _lp(c, A_eq, b_eq, "min")
        hi, phi = _lp(c, A_eq, b_eq, "max")
        rows.append({"k": k, "n_configurations": comb(M, k),
                     "lower": lo, "upper": hi, "exact": True})
        for p in (plo, phi):
            if p is not None:
                verts.append(p)
    return pd.DataFrame(rows), verts


def moment_bounds(m, max_lp=MAX_M_LP):
    """
    E[k] is FIXED at sum_j m_j — a marginal set determines the mean grade
    exactly, with no interval. Var[k] = E[k^2] - E[k]^2 is bounded, and
    E[k^2] is linear in p, so its extremes are exact LP values.
    """
    m = np.asarray(m, float)
    M = len(m)
    mu = float(m.sum())
    out = {"M": M, "expected_k": mu, "var_lower": np.nan, "var_upper": np.nan,
           "k_occupied_min": 0, "k_occupied_max": M, "exact_var": False}
    if M == 0 or M > max_lp:
        return out
    S, A_eq, b_eq = _polytope(m)
    k2 = (S.sum(axis=1)) ** 2
    lo, _ = _lp(k2, A_eq, b_eq, "min")
    hi, _ = _lp(k2, A_eq, b_eq, "max")
    out["var_lower"] = max(0.0, lo - mu ** 2)
    out["var_upper"] = hi - mu ** 2
    out["exact_var"] = True
    return out


def residual_weight_freedom(m):
    """
    W = U[p_0] - L[p_0] = sum_j m_j - max_j m_j     (valid when sum m <= 1)

    Width of the bound on the all-unmodified combination. W = 0 if and only if
    M <= 1, so W is the continuous form of the uniqueness census. Unlike
    sum(m) it does not scale with the overall extent of modification.
    """
    m = np.asarray(m, float)
    return 0.0 if m.size == 0 else float(m.sum() - m.max())


def projection_class(M):
    """
    P(m) has 2^M variables and M+1 equality constraints. The product
    distribution prod_j m_j^x_j (1-m_j)^(1-x_j) is strictly positive and meets
    every marginal, so P(m) always contains a relatively interior point and has
    full dimension 2^M - (M+1). Therefore:

        M = 0    one admissible combination, weight 1     trivially determined
        M = 1    two combinations, weights (1-m, m)        uniquely determined
        M >= 2   dim = 2^M - (M+1) >= 1                    bounded family

    This assay cannot produce a boundary-1 marginal, so no M >= 2 projection
    degenerates to a point.
    """
    return ("trivially_determined" if M == 0 else
            "uniquely_determined" if M == 1 else "bounded_family")


def grade_entropy_attained(m, verts, max_lp=MAX_M_LP):
    """
    Grade entropy H_k(q) = -sum_k q_k log q_k.

    H is concave and q -> p is linear, so exact extrema need either vertex
    enumeration of the projected polytope or a convex solve. What is reported
    here is the ATTAINED range over a set of genuinely feasible points: the
    max-entropy product distribution plus every LP vertex found while computing
    the grade envelope. Any attained value is achievable, so

        [H_attained_min, H_attained_max]  is an INNER bound on the true range.

    It is labelled as such in the report and never presented as the exact range.
    """
    m = np.asarray(m, float)
    M = len(m)
    if M == 0 or M > max_lp:
        return {"H_attained_min": np.nan, "H_attained_max": np.nan,
                "H_product": np.nan, "n_points": 0}

    S, _, _ = _polytope(m)
    kvec = S.sum(axis=1).astype(int)

    def H_of_p(p):
        q = np.bincount(kvec, weights=p, minlength=M + 1)
        q = q[q > 1e-15]
        return float(-(q * np.log(q)).sum())

    p_prod = np.prod(np.where(S == 1, m, 1.0 - m), axis=1)
    p_prod = p_prod / p_prod.sum()
    vals = [H_of_p(p_prod)] + [H_of_p(v) for v in verts]
    return {"H_attained_min": float(np.min(vals)),
            "H_attained_max": float(np.max(vals)),
            "H_product": float(H_of_p(p_prod)),
            "n_points": len(vals)}


# ==============================================================================
# 3. PIPELINE:  MaxQuant + FASTA  ->  formnet_sites
#    Faithful to notebook cells 0-13.
# ==============================================================================

def _fasta_accession(header):
    h = header[1:].strip()
    if "|" in h:
        parts = h.split("|")
        if len(parts) >= 2:
            return parts[1]
    return h.split()[0]


def read_fasta(path):
    seqs, acc, buf = {}, None, []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if acc is not None:
                    seqs[acc] = "".join(buf)
                acc, buf = _fasta_accession(line), []
            else:
                buf.append(line)
    if acc is not None:
        seqs[acc] = "".join(buf)
    return seqs


def lysine_positions(seq):
    return {i + 1 for i, aa in enumerate(seq) if aa == "K"}


def locate_maxquant():
    zp = ROOT / ZIP_NAME
    if zp.exists():
        target = ROOT / "mq"
        if not target.exists():
            with zipfile.ZipFile(zp) as z:
                z.extractall(target)
        hits = list(target.rglob(EXPERIMENT_DIR))
        if hits:
            return hits[0]
    hits = list(ROOT.rglob(EXPERIMENT_DIR))
    return hits[0] if hits else None


def find_fasta():
    cands = sorted(list(ROOT.glob("*.fasta")) + list(ROOT.glob("*.fa")))
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise FileNotFoundError(f"No .fasta in {ROOT}")
    raise ValueError(f"Leave exactly one FASTA in {ROOT}; found {cands}")


def build_formnet_sites(mq_dir, fasta_path, verbose=True):
    def say(*a):
        if verbose:
            print("   ", *a)

    rule("BUILDING formnet_sites FROM MAXQUANT", "-")
    say("MaxQuant dir:", mq_dir)
    say("FASTA:", fasta_path)

    evidence = pd.read_csv(mq_dir / "evidence.txt", sep="\t", low_memory=False,
                           usecols=["Sequence", "Modified sequence", "Acetyl (K)",
                                    "Acetyl (K) site IDs", "Experiment",
                                    "Ratio H/L", "Intensity H", "Reverse",
                                    "Potential contaminant", "Peptide ID"])
    peptides = pd.read_csv(mq_dir / "peptides.txt", sep="\t", low_memory=False,
                           usecols=["id", "Sequence", "Leading razor protein",
                                    "Start position", "Protein group IDs"])
    acetyl_sites = pd.read_csv(mq_dir / "Acetyl (K)Sites.txt", sep="\t",
                               low_memory=False,
                               usecols=["id", "Protein", "Position",
                                        "Localization prob", "Intensity H",
                                        "Intensity H 1_10", "Intensity H 1_100",
                                        "Intensity H 1_1000", "Intensity H 1_12_5",
                                        "Reverse", "Potential contaminant"])
    say("evidence", evidence.shape, "| peptides", peptides.shape,
        "| sites", acetyl_sites.shape)

    fasta = read_fasta(fasta_path)
    say("FASTA proteins:", thou(len(fasta)))

    # ---- unmodified coordinates: native peptides with zero acetylation -------
    ev = evidence.merge(peptides.rename(columns={"id": "Peptide ID",
                                                "Sequence": "Peptide sequence"}),
                        on="Peptide ID", how="left")
    ev = ev[ev["Reverse"].isna() & ev["Potential contaminant"].isna()].copy()
    ev["Intensity H"] = pd.to_numeric(ev["Intensity H"], errors="coerce").fillna(0)
    ev["Start position"] = pd.to_numeric(ev["Start position"], errors="coerce")

    native_unmod = ev[(ev["Acetyl (K)"] == 0) & (ev["Intensity H"] > 0)
                      & ev["Start position"].notna()
                      & ev["Leading razor protein"].notna()]
    unmodified_coordinates = set()
    for acc, start, seq in native_unmod[["Leading razor protein", "Start position",
                                        "Sequence"]].drop_duplicates().itertuples(
                                            index=False):
        for off, aa in enumerate(str(seq)):
            if aa == "K":
                unmodified_coordinates.add((str(acc), int(start) + off))
    say("unmodified K coordinates:", thou(len(unmodified_coordinates)))

    # ---- acetylated coordinates ---------------------------------------------
    site = acetyl_sites.copy()
    icols = ["Intensity H", "Intensity H 1_10", "Intensity H 1_100",
             "Intensity H 1_1000", "Intensity H 1_12_5"]
    for c in icols:
        site[c] = pd.to_numeric(site[c], errors="coerce").fillna(0)
    site["Localization prob"] = pd.to_numeric(site["Localization prob"],
                                              errors="coerce")
    site["Position"] = pd.to_numeric(site["Position"], errors="coerce")
    native_ac = site[site["Reverse"].isna() & site["Potential contaminant"].isna()
                     & site["Position"].notna() & site["Protein"].notna()
                     & (site["Localization prob"] >= LOC_PROB_MIN)
                     & (site[icols].max(axis=1) > 0)]
    acetylated_coordinates = set(zip(native_ac["Protein"].astype(str),
                                     native_ac["Position"].astype(int)))
    say("acetylated K coordinates:", thou(len(acetylated_coordinates)))

    zero_coordinates = unmodified_coordinates - acetylated_coordinates
    detected_coordinates = unmodified_coordinates | acetylated_coordinates
    say("detected:", thou(len(detected_coordinates)),
        "| boundary zeros:", thou(len(zero_coordinates)))

    # ---- stoichiometry from the dilution series -----------------------------
    quant = evidence[evidence["Reverse"].isna()
                     & evidence["Potential contaminant"].isna()
                     & (evidence["Acetyl (K)"] == 1)
                     & evidence["Experiment"].isin(D_MAP)].copy()
    quant["Ratio H/L"] = pd.to_numeric(quant["Ratio H/L"], errors="coerce")
    quant = quant[quant["Ratio H/L"].notna() & (quant["Ratio H/L"] > 0)]
    quant = quant[quant["Acetyl (K) site IDs"].astype(str).str.match(r"^\d+$")].copy()
    quant["site_id"] = quant["Acetyl (K) site IDs"].astype(int)

    site_map = acetyl_sites[["id", "Protein", "Position", "Localization prob",
                             "Reverse", "Potential contaminant"]].copy()
    site_map["Position"] = pd.to_numeric(site_map["Position"], errors="coerce")
    site_map["Localization prob"] = pd.to_numeric(site_map["Localization prob"],
                                                  errors="coerce")
    site_map = site_map[site_map["Reverse"].isna()
                        & site_map["Potential contaminant"].isna()
                        & site_map["Position"].notna()
                        & (site_map["Localization prob"] >= LOC_PROB_MIN)]
    quant = quant.merge(site_map[["id", "Protein", "Position"]],
                        left_on="site_id", right_on="id", how="inner")

    q = quant.groupby(["Modified sequence", "site_id", "Protein", "Position",
                       "Experiment"], as_index=False)["Ratio H/L"].median()
    q["D"] = q["Experiment"].map(D_MAP)
    q["RD"] = q["D"] / q["Ratio H/L"]        # H = native, L = chemical

    rows, dropped_denominator = [], 0
    for keys, blk in q.groupby(["Modified sequence", "site_id", "Protein",
                                "Position"]):
        vals = blk["RD"].dropna().to_numpy(float)
        if len(vals) < MIN_DILUTIONS:
            continue
        if vals.max() / vals.min() > MAX_DILUTION_FOLD_RANGE:
            continue
        RD = float(np.median(vals))
        denom = RD - (1 - C_CHEM)
        if denom <= 0:
            dropped_denominator += 1
            continue
        S_ = C_CHEM / denom
        if not np.isfinite(S_) or not (0 < S_ <= STOICH_MAX):
            continue
        rows.append({"Protein": str(keys[2]), "Position": int(keys[3]),
                     "stoichiometry": S_})
    pep_stoich = pd.DataFrame(rows)
    say("quantified modified peptide forms:", thou(len(pep_stoich)))
    say(f"dropped, RD below the {1 - C_CHEM:.4f} threshold:",
        thou(dropped_denominator),
        "  <- the high-stoichiometry tail; see report section 1")

    site_stoich = (pep_stoich.groupby(["Protein", "Position"], as_index=False)
                   ["stoichiometry"].sum())
    site_stoich = site_stoich[(site_stoich["stoichiometry"] > 0)
                              & (site_stoich["stoichiometry"] <= STOICH_MAX)]
    say("quantified sites:", thou(len(site_stoich)))

    partial_lookup = {(str(r.Protein), int(r.Position)): float(r.stoichiometry)
                      for r in site_stoich.itertuples(index=False)}
    acetylated_unquantified = acetylated_coordinates - set(partial_lookup)

    # ---- classify every FASTA lysine ----------------------------------------
    by_acc = {}
    for acc, pos in detected_coordinates:
        by_acc.setdefault(acc, set())
    zero_by, part_by, unq_by = {}, {}, {}
    for acc, pos in zero_coordinates:
        zero_by.setdefault(acc, set()).add(pos)
    for acc, pos in partial_lookup:
        part_by.setdefault(acc, set()).add(pos)
    for acc, pos in acetylated_unquantified:
        unq_by.setdefault(acc, set()).add(pos)

    records = []
    for acc in sorted(a for a in by_acc if a in fasta):
        zK, pK, uK = zero_by.get(acc, set()), part_by.get(acc, set()), unq_by.get(acc, set())
        for pos in sorted(lysine_positions(fasta[acc])):
            if pos in pK:
                sc, mg = "partial", partial_lookup[(acc, pos)]
            elif pos in zK:
                sc, mg = "unmodified_0", 0.0
            elif pos in uK:
                sc, mg = "modified_unquantified", np.nan
            else:
                sc, mg = "unmeasured_X", np.nan
            records.append({"protein_id": acc, "site_id": f"K{pos}",
                            "position": pos, "amino_acid": "K",
                            "PTM": "acetylation", "state_class": sc,
                            "marginal": mg})

    fs = pd.DataFrame(records)
    fs.to_csv(SITES_CACHE, index=False)
    say("formnet_sites:", fs.shape, "-> cached at", SITES_CACHE)
    return fs


def load_formnet_sites():
    if "formnet_sites" in globals() and isinstance(globals()["formnet_sites"],
                                                   pd.DataFrame):
        print("Using formnet_sites already in the session.")
        return globals()["formnet_sites"]
    if SITES_CACHE.exists():
        print(f"Loading cached {SITES_CACHE}")
        return pd.read_csv(SITES_CACHE)
    mq = locate_maxquant()
    if mq is None:
        raise FileNotFoundError(
            f"Need either {SITES_CACHE}, or {ZIP_NAME} / a {EXPERIMENT_DIR} "
            f"directory plus one .fasta, in {ROOT}")
    return build_formnet_sites(mq, find_fasta())


# ==============================================================================
# 4. PER-PROTEIN SUMMARY
# ==============================================================================

def protein_summary(formnet_sites, verbose=True):
    rows = []
    n = formnet_sites["protein_id"].nunique()
    for i, (pid, blk) in enumerate(formnet_sites.groupby("protein_id")):
        if verbose and i and i % 1000 == 0:
            print(f"    {i}/{n}")
        cls = blk["state_class"]
        R = int(len(blk))
        B = int((cls == "unmodified_0").sum())
        Qm = int((cls == "modified_unquantified").sum())
        X = int((cls == "unmeasured_X").sum())
        mm = blk.loc[cls == "partial", "marginal"].dropna().to_numpy(float)
        M = len(mm)
        C = B + M
        U = Qm + X
        mu = float(mm.sum()) if M else 0.0

        rows.append({
            "protein_id": pid,
            "R": R, "D": R - X, "B": B, "M": M, "Qm": Qm, "X": X, "C": C, "U": U,
            "coverage_detected": (R - X) / R,
            "coverage_constrained": C / R,
            "complete_detection": X == 0,
            "complete_constraint": U == 0,
            # support, exact
            "bits_full": R,
            "bits_eliminated": B,
            "bits_admissible_full": R - B,          # log2 |A(m)| over all R
            "bits_admissible_projection": M,        # log2 |A(m)| over the projection
            "projection_polytope_dim": (2 ** M - M - 1) if M <= 20 else np.inf,
            "solution_class": projection_class(M),
            # weight
            "expected_k": mu if M else np.nan,
            "p0_lower": max(0.0, 1.0 - mu) if M else 1.0,
            "p0_upper": float(np.min(1.0 - mm)) if M else 1.0,
            "p_any_lower": float(mm.max()) if M else 0.0,
            "p_any_upper": min(1.0, mu) if M else 0.0,
            "W": residual_weight_freedom(mm),
            "m_max": float(mm.max()) if M else np.nan,
            "m_min": float(mm.min()) if M else np.nan,
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 5. THE REPORT
# ==============================================================================

def section1_coverage(summ, formnet_sites):
    rule("1.  COVERAGE — the observable ModForm space")
    counts = formnet_sites["state_class"].value_counts()
    total = int(counts.sum())

    print(f"Protein groups                        {thou(summ.R.count())}")
    print(f"FASTA lysine coordinates      R       {thou(summ.R.sum())}")
    print()
    for st in STATE_ORDER:
        v = int(counts.get(st, 0))
        print(f"  {STATE_LABELS[st]:<30}{thou(v):>10}   {100*v/total:6.2f}%")
    print()
    print(f"Detected                      D       {thou(summ.D.sum()):>10}   "
          f"{100*summ.D.sum()/summ.R.sum():6.2f}%")
    print(f"Directly constrained          C = B+M {thou(summ.C.sum()):>10}   "
          f"{100*summ.C.sum()/summ.R.sum():6.2f}%")
    print(f"Unconstrained                 U = R-C {thou(summ.U.sum()):>10}   "
          f"{100*summ.U.sum()/summ.R.sum():6.2f}%")
    print()
    print("Projections:")
    print(f"  complete   U = 0                    {thou((summ.U == 0).sum()):>10}   "
          f"{100*(summ.U == 0).mean():6.2f}%   the projection IS the full object")
    print(f"  incomplete U > 0                    {thou((summ.U > 0).sum()):>10}   "
          f"{100*(summ.U > 0).mean():6.2f}%   each combination is a fibre of 2^U")
    print()
    print("Distribution of constrained fraction C/R:")
    print(summ["coverage_constrained"].describe(
        percentiles=[.25, .5, .75, .95]).to_string())
    print()
    print("Distribution of the interior marginals m_j:")
    mvals = formnet_sites.loc[formnet_sites.state_class == "partial", "marginal"]
    print(mvals.describe(percentiles=[.25, .5, .75, .95, .99]).to_string())
    print()
    print("NOTE  The pipeline retains 0 < m <= "
          f"{STOICH_MAX:g}. Sites whose RD falls below {1 - C_CHEM:.4f} are")
    print("      discarded by the denominator guard, so the surviving marginals are")
    print("      biased against the high-stoichiometry tail. This is a property of")
    print("      the input, not of the theorem, and it bounds what the weight side")
    print("      of the analysis can show on this dataset.")
    print()


def section2_boundary(summ):
    rule("2.  BOUNDARY MARGINALS — exact exclusion, exact identity constraint")
    R = int(summ.R.sum())
    B = int(summ.B.sum())
    M = int(summ.M.sum())
    C = int(summ.C.sum())

    print("Bits are additive across a Cartesian product; state COUNTS are not.")
    print("A sum of 2^R over proteins is dominated by the single largest protein")
    print("and is meaningless, so the ledger below is reported in bits.")
    print()
    print("Full lysine-acetylation space")
    print(f"  log2 |Omega|                        {thou(R):>12} bits")
    print(f"  eliminated by boundary marginals    {thou(B):>12} bits")
    print(f"  log2 |A(m)| over all R lysines      {thou(R - B):>12} bits")
    print(f"  fraction of freedom resolved        {100*B/R:>11.3f} %")
    print()
    print("Directly measured projection")
    print(f"  log2 2^C  constrained coordinates   {thou(C):>12} bits")
    print(f"  eliminated                          {thou(B):>12} bits")
    print(f"  log2 |A(m)| = 2^M surviving         {thou(M):>12} bits")
    print(f"  fraction of the projection excluded {100*(1 - 2.0**-B):>11.6f} %"
          if B < 60 else "")
    print()
    print("Identity check   2^(M) x 2^(U) = 2^(R-B):")
    lhs = int(summ.M.sum() + summ.U.sum())
    print(f"  sum(M) + sum(U) = {thou(lhs)}   sum(R) - sum(B) = {thou(R - B)}"
          f"   {'OK' if lhs == R - B else 'MISMATCH'}")
    print()
    print("Per protein, bits eliminated B:")
    print(summ["B"].describe(percentiles=[.5, .9, .99]).to_string())
    top = summ.nlargest(10, "B")[["protein_id", "R", "B", "M", "U",
                                  "bits_admissible_full"]]
    print()
    print("Most compressed protein groups:")
    print(top.to_string(index=False))
    print()


def section3_identity_vs_weight(summ):
    rule("3.  IDENTITY RESOLUTION vs WEIGHT RESOLUTION")
    print("Two independent axes. Conflating them is the main interpretive risk.")
    print()
    print("  identity  U = 0 complete  |  U > 0 incomplete (fibre of 2^U)")
    print("  weight    M = 0 single combination | M = 1 unique | M >= 2 bounded")
    print()
    d = summ.copy()
    d["identity"] = np.where(d.U == 0, "complete (U=0)", "incomplete (U>0)")
    d["weight"] = d.M.map(lambda M: {"trivially_determined": "M=0 single",
                                     "uniquely_determined": "M=1 unique",
                                     "bounded_family": "M>=2 bounded"}
                          [projection_class(M)])
    ct = pd.crosstab(d["identity"], d["weight"])
    for c in ["M=0 single", "M=1 unique", "M>=2 bounded"]:
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[["M=0 single", "M=1 unique", "M>=2 bounded"]]
    print(ct.to_string())
    print()
    exact_identity = int(((d.U == 0) & (d.M == 0)).sum())
    exact_complete = int(((d.U == 0) & (d.M == 1)).sum())
    print(f"Exact identity            |A(m)| = 1, one complete modform   "
          f"{thou(exact_identity)}")
    print(f"Exact complete weighting  U = 0, M = 1, P(m) a point          "
          f"{thou(exact_complete)}")
    print(f"Exact projected weighting M <= 1 but U > 0                    "
          f"{thou(int(((d.U > 0) & (d.M <= 1)).sum()))}")
    print(f"Bounded weighting         M >= 2                              "
          f"{thou(int((d.M >= 2).sum()))}")
    print()
    bf = d[d.M >= 2]
    if len(bf):
        print(f"Over the bounded family, summed projected polytope dimension "
              f"sum(2^M - M - 1) = {thou(bf.projection_polytope_dim.sum())}")
        print("This is the honest measure of what remains undetermined, and unlike")
        print("state counts it is additive across proteins.")
    print()
    print("Uniqueness holds iff M <= 1. P(m) has 2^M variables and M+1 equality")
    print("constraints; the product distribution is strictly positive and meets")
    print("every marginal, so P(m) always has full dimension 2^M - (M+1), which is")
    print(">= 1 for M >= 2. This assay yields no boundary-1 marginal, so no M >= 2")
    print("projection can degenerate to a point.")
    print()
    print("Residual weight freedom W = sum(m) - max(m), width of the bound on the")
    print("all-unmodified combination. W = 0 iff M <= 1.")
    q = d[d.M > 0]
    print(f"  W = 0 (determined)   {thou(int((q.W <= 0).sum()))}")
    print(f"  W > 0 (bounded)      {thou(int((q.W > 0).sum()))}")
    if (q.W > 0).any():
        print(q.loc[q.W > 0, "W"].describe(
            percentiles=[.5, .95]).to_string())
    print()


def section4_grades(formnet_sites, summ, exemplars, max_lp=MAX_M_LP):
    rule("4.  GRADE-SPACE RECOVERY  (identity-degenerate but grade-constrained)")
    print("E[k] = sum_j m_j is FIXED by the marginal set, not bounded.")
    print("q_k = sum_{||x||=k} p_x is bounded exactly by LP over P(m).")
    print()
    all_env, all_cfg = [], []
    targets = [p for p in exemplars if p in set(summ.protein_id)]
    if not targets:
        targets = summ.nlargest(4, "M").protein_id.tolist()

    for pid in targets:
        blk = formnet_sites[formnet_sites.protein_id == pid]
        part = blk[blk.state_class == "partial"].sort_values("position")
        m = part["marginal"].to_numpy(float)
        labels = [f"K{int(p)}" for p in part["position"]]
        row = summ.loc[summ.protein_id == pid].iloc[0]
        name = PROTEIN_NAMES.get(pid, pid)

        rule(f"{name}  ({pid})", "-")
        print(f"  R={row.R}  D={row.D}  B={row.B}  M={row.M}  Qm={row.Qm}  "
              f"X={row.X}  C={row.C}  U={row.U}")
        print(f"  log2|A(m)| over R = {row.bits_admissible_full} bits   "
              f"({row.B} bits eliminated)")
        print(f"  projection support 2^M = {2**int(row.M)} combinations, "
              f"polytope dim {int(row.projection_polytope_dim)}")
        if row.M == 0:
            print("  no interior coordinate; the projection carries weight 1 "
                  "on one combination")
            continue

        env, verts = grade_envelope(m, max_lp)
        mom = moment_bounds(m, max_lp)
        ent = grade_entropy_attained(m, verts, max_lp)
        env.insert(0, "protein_id", pid)
        all_env.append(env)

        print(f"  E[k] = sum m_j = {mom['expected_k']:.6g}   (exact)")
        if mom["exact_var"]:
            print(f"  Var[k] in [{mom['var_lower']:.6g}, {mom['var_upper']:.6g}]"
                  f"   (exact; E[k^2] is linear in p)")
        print(f"  grade range occupied: k = 0 .. {int(row.M)}")
        print(f"  W = {row.W:.6g}")
        print()
        print("  grade envelope" + ("" if env["exact"].all()
                                    else "   (some rows are Markov relaxations)"))
        show = env.copy()
        show["n_configurations"] = show["n_configurations"].map(thou)
        print(show.drop(columns=["protein_id"]).to_string(index=False,
              float_format=lambda v: f"{v:.6g}"))
        print()
        if np.isfinite(ent["H_attained_max"]):
            print(f"  grade entropy, ATTAINED range over {ent['n_points']} "
                  f"feasible points:")
            print(f"    [{ent['H_attained_min']:.6g}, {ent['H_attained_max']:.6g}]"
                  f"   H at the max-entropy point = {ent['H_product']:.6g}")
            print("    This is an INNER bound on the true range: every value shown")
            print("    is attained, but the exact extrema are not claimed.")
            print()

        cfg = frechet_configuration_bounds(m, labels)
        cfg.insert(0, "protein_id", pid)
        all_cfg.append(cfg)
        nec = cfg[cfg.status == "necessary"]
        print(f"  combinations: {len(cfg)} admissible, "
              f"{len(nec)} required (lower bound > 0)")
        print("  strongest ceilings:")
        print(cfg.drop(columns=["protein_id"]).head(TOP_CONFIGS)
              .to_string(index=False, float_format=lambda v: f"{v:.6g}"))
        print()

    env_df = pd.concat(all_env, ignore_index=True) if all_env else pd.DataFrame()
    cfg_df = pd.concat(all_cfg, ignore_index=True) if all_cfg else pd.DataFrame()
    return env_df, cfg_df, targets


# ==============================================================================
# 5B. THE GRADE HIERARCHY
#
#   Level 1  support geometry      k in [B1, B1 + P]          possibility
#   Level 2  exact first moment    E[K] = B1 + sum_j m_j      every feasible p
#   Level 3  grade envelope        q_k^min <= q_k <= q_k^max  forced vs permitted
#   Level 4  modified fraction     max m <= rho <= min(1, S)  B1 = 0 only
#   Level 5  conditional mean      max(1,S) <= E[K|K>=1] <= S/max m
#
# Levels 4 and 5 have content only when B1 = 0. If any coordinate is fixed
# modified then K >= B1 >= 1 always, rho = 1 exactly, and E[K|K>=1] collapses
# onto the exact first moment of level 2.
#
# Closed forms verified against the LP:
#     q_0^min = max(0, 1 - S)          q_0^max = 1 - max m
#     q_1^min = max(0, 2 max m - S)
#     rho     in [max m, min(1, S)]
#     E[K|K>=1] in [max(1, S), S / max m]
# ==============================================================================

MIN_M_REGIME = 3      # smallest M for which the grade regimes are meaningful


def grade_hierarchy(m, B1=0, max_lp=MAX_M_LP):
    """All five levels for one protein. m = the interior marginals only."""
    m = np.asarray(m, float)
    P = len(m)
    S = float(m.sum()) + B1
    Sp = float(m.sum())                      # partial-only sum
    mx = float(m.max()) if P else 0.0

    env, _ = grade_envelope(m, max_lp)
    env = env.copy()
    env["k"] = env["k"] + B1                 # grades carry the fixed-1 offset

    forced = env[env.lower > 1e-12]
    out = {
        "P": P, "B1": B1,
        "k_min": B1, "k_max": B1 + P,
        "n_identities": 2 ** P,
        "expected_k": S,
        "envelope": env,
        "forced_grades": forced.k.tolist(),
        "forced_weight": float(env.lower.sum()),
        "envelope_width": float((env.upper - env.lower).sum()),
        "q0_min": max(0.0, 1.0 - Sp), "q0_max": 1.0 - mx if P else 1.0,
        "q1_min_closed": max(0.0, 2 * mx - Sp),
    }

    if B1 == 0 and P > 0:
        out["rho_lower"] = mx
        out["rho_upper"] = min(1.0, Sp)
        out["cond_mean_lower"] = max(1.0, Sp)
        out["cond_mean_upper"] = Sp / mx
        out["cond_spread"] = ((Sp / mx) - 1.0) / (P - 1) if P > 1 else 0.0
        out["conditional_informative"] = True
    else:
        out["rho_lower"] = out["rho_upper"] = 1.0
        out["cond_mean_lower"] = out["cond_mean_upper"] = S
        out["cond_spread"] = np.nan
        out["conditional_informative"] = False
    return out


def hierarchy_table(formnet_sites, summ, max_lp=MAX_M_LP, verbose=True):
    """Per-protein level 2, 4 and 5 scalars, plus the envelope summaries."""
    rows = []
    for pid, blk in formnet_sites.groupby("protein_id"):
        mm = blk.loc[blk.state_class == "partial", "marginal"].dropna().to_numpy(float)
        if len(mm) == 0:
            continue
        S, mx, P = float(mm.sum()), float(mm.max()), len(mm)
        rows.append({
            "protein_id": pid, "P": P, "S": S, "m_max": mx,
            "k_min": 0, "k_max": P, "n_identities": 2 ** P,
            "expected_k": S,
            "rho_lower": mx, "rho_upper": min(1.0, S),
            "rho_width": min(1.0, S) - mx,
            "cond_mean_lower": max(1.0, S), "cond_mean_upper": S / mx,
            "cond_spread": ((S / mx) - 1.0) / (P - 1) if P > 1 else 0.0,
            "q0_min": max(0.0, 1.0 - S), "q0_max": 1.0 - mx,
            "q1_min": max(0.0, 2 * mx - S),
            "grade1_forced": (2 * mx - S) > 1e-12,
        })
    return pd.DataFrame(rows)


def select_exemplars(hier, summ, min_m=MIN_M_REGIME):
    """
    Pick one protein per regime. Each rule is a documented extremum, so the
    choice is reproducible rather than curated.

        support_compression   most boundary coordinates, B
        identity_degenerate   most interior coordinates, P  (2^P identities)
        grade_forced          P >= min_m, smallest cond_spread
                              -> one site dominates; modified copies carry ~1 mark
        grade_permissive      P >= min_m, largest cond_spread
                              -> marginals comparable; up to P marks permitted
        uniquely_determined   P = 1 with the most boundary compression
    """
    d = hier.merge(summ[["protein_id", "B", "R", "U"]], on="protein_id", how="left")
    big = d[d.P >= min_m]
    pick = {}

    def take(frame, col, how):
        if not len(frame):
            return None
        return (frame.nsmallest(1, col) if how == "min"
                else frame.nlargest(1, col)).protein_id.iloc[0]

    pick["support_compression"] = take(d, "B", "max")
    pick["identity_degenerate"] = take(d, "P", "max")
    pick["grade_forced"] = take(big, "cond_spread", "min")
    pick["grade_permissive"] = take(big, "cond_spread", "max")
    pick["uniquely_determined"] = take(d[d.P == 1], "B", "max")

    seen, out = set(), {}
    for k, v in pick.items():
        if v is not None and v not in seen:
            out[k] = v
            seen.add(v)
    return out


REGIME_LABEL = {
    "support_compression": "Maximal support compression",
    "identity_degenerate": "Maximal identity degeneracy",
    "grade_forced": "Degenerate identity, forced grade",
    "grade_permissive": "Degenerate identity, permissive grade",
    "uniquely_determined": "Unique weighted solution",
}


def section5_hierarchy(formnet_sites, summ, hier, exemplars, max_lp=MAX_M_LP):
    rule("5.  THE GRADE HIERARCHY")
    print("Five levels of evidential strength from the same marginal set:")
    print("  1  a grade is POSSIBLE                k in [B1, B1+P]")
    print("  2  the ensemble has an EXACT mean     E[K] = B1 + sum_j m_j")
    print("  3  a grade MUST/MAY carry weight      q_k in [q_k^min, q_k^max]")
    print("  4  the modified fraction is BOUNDED   rho in [max m, min(1,S)]")
    print("  5  its mean grade is BOUNDED          E[K|K>=1] in [max(1,S), S/max m]")
    print()
    print("Levels 4-5 are informative only when B1 = 0. This assay yields no")
    print("boundary-1 marginal, so B1 = 0 throughout and they always apply.")
    print()

    n_forced = int(hier.grade1_forced.sum())
    print(f"Proteins where grade 1 is FORCED (q_1^min = 2 max m - S > 0):  "
          f"{thou(n_forced)} of {thou(len(hier))}")
    print("  For these the data do not merely permit singly-modified copies;")
    print("  every compatible weighting must place weight at grade 1.")
    print()
    print("Conditional mean grade among modified copies, E[K|K>=1] <= S/max m:")
    print(hier["cond_mean_upper"].describe(
        percentiles=[.1, .5, .9]).to_string())
    print()
    print("Normalised spread (S/max m - 1)/(P-1) in [0,1]:")
    print("  0 = one site dominates, modified copies carry essentially one mark")
    print("  1 = marginals equal, up to P marks permitted on average")
    print(hier.loc[hier.P > 1, "cond_spread"].describe(
        percentiles=[.1, .5, .9]).to_string())
    print()

    rule("REGIME EXEMPLARS", "-")
    for regime, pid in exemplars.items():
        h = hier.loc[hier.protein_id == pid].iloc[0]
        s = summ.loc[summ.protein_id == pid].iloc[0]
        nm = PROTEIN_NAMES.get(pid, pid)
        print(f"\n{REGIME_LABEL[regime]}   {nm} ({pid})")
        print(f"   R={int(s.R)}  B={int(s.B)}  P={int(h.P)}  U={int(s.U)}")
        print(f"   L1  grades possible          k in [0, {int(h.P)}]   "
              f"{thou(h.n_identities)} admissible identities in the projection")
        print(f"   L2  exact first moment       E[K] = {h.S:.6g}")
        mm = formnet_sites.loc[(formnet_sites.protein_id == pid)
                               & (formnet_sites.state_class == "partial"),
                               "marginal"].dropna().to_numpy(float)
        hh = grade_hierarchy(mm, B1=0, max_lp=max_lp)
        env = hh["envelope"]
        print(f"   L3  grade envelope           forced grades "
              f"{hh['forced_grades']}   total forced weight "
              f"{hh['forced_weight']:.6g}")
        print(env.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
        print(f"   L4  modified fraction        rho in "
              f"[{h.rho_lower:.6g}, {h.rho_upper:.6g}]")
        print(f"   L5  conditional mean grade   E[K|K>=1] in "
              f"[{h.cond_mean_lower:.6g}, {h.cond_mean_upper:.6g}]"
              f"   of a possible [1, {int(h.P)}]")
        print(f"       normalised spread        {h.cond_spread:.4f}")
    print()
    return exemplars


# ==============================================================================
# 9. FIGURE 3 — the hierarchy across the proteome
# ==============================================================================

def figure3_hierarchy_map(hier, summ, exemplars, out="formnet_figure3_hierarchy",
                          max_width=540):
    d = hier.merge(summ[["protein_id", "B"]], on="protein_id", how="left")
    ex_ids = list(exemplars.values())
    mp = MultiPanel(max_width=max_width)

    # ---- A. level 2: the exact first moment --------------------------------
    mp.panel("A", width=140, height=112, pad_left=44, pad_top=16,
             margin_right=22, margin_bottom=30)
    ax = plt.gca()
    v = d.S[d.S > 0]
    ax.hist(v, bins=np.logspace(np.log10(v.min()), np.log10(v.max()), 34),
            color=C_PARTIAL, edgecolor="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$E[K] = \\sum_j m_j$", fontsize=6.5)
    ax.set_ylabel("Proteins", fontsize=6.5)
    _bare(ax)
    ax.set_title("Level 2   exact first moment", fontsize=7, fontweight="bold",
                 pad=4, loc="left")
    ax.text(0.97, 0.94, "exact for every\nfeasible weighting", fontsize=5.2,
            color=C_RULE, transform=ax.transAxes, ha="right", va="top",
            linespacing=1.3)

    # ---- B. the regime map --------------------------------------------------
    mp.panel("B", width=250, height=112, pad_left=46, pad_top=16,
             margin_right=0, margin_bottom=30)
    ax = plt.gca()
    q = d[d.P > 1]
    ax.scatter(q.P, q.cond_spread, s=3 + 1.6 * q.B, facecolor=C_PARTIAL,
               edgecolor="white", linewidth=0.25, alpha=0.6, zorder=2)
    forced = q[q.grade1_forced]
    ax.scatter(forced.P, forced.cond_spread, s=3 + 1.6 * forced.B, marker="o",
               facecolor="none", edgecolor=C_ZERO, linewidth=0.6, zorder=3,
               label="grade 1 forced")
    ax.set_xticks(range(2, int(q.P.max()) + 1, max(1, int(q.P.max()) // 6)))
    ax.set_xlim(1.4, q.P.max() + 0.8)
    ax.set_xlabel("Identity degeneracy   $P$   ($2^P$ identities)", fontsize=6.5)
    ax.set_ylabel("Normalised spread\n$(S/\\max m - 1)/(P-1)$", fontsize=6.5,
                  linespacing=1.3)
    ax.set_ylim(-0.04, 1.04)
    _bare(ax)
    for pid in ex_ids:
        r = q.loc[q.protein_id == pid]
        if not len(r):
            continue
        ax.annotate(PROTEIN_NAMES.get(pid, pid),
                    xy=(float(r.P.iloc[0]), float(r.cond_spread.iloc[0])),
                    xytext=(6, 6), textcoords="offset points", fontsize=5.6,
                    color=C_INK, arrowprops=dict(arrowstyle="-", lw=0.5,
                                                 color=C_INK, shrinkA=0, shrinkB=1.5))
    ax.text(0.99, 0.03, "one site dominates\nmodified copies carry ~1 mark",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0,
            color="#9C9891", style="italic", linespacing=1.3)
    ax.text(0.99, 0.97, "marginals comparable\nup to $P$ marks permitted",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.0,
            color="#9C9891", style="italic", linespacing=1.3)
    ax.legend(fontsize=5.2, frameon=False, loc="lower left", handletextpad=0.4,
              borderpad=0.2, scatterpoints=1)
    ax.set_title(f"Regime map   n = {thou(len(q))} proteins with $P \\geq$ 2",
                 fontsize=7, fontweight="bold", pad=4, loc="left")

    # ---- C. level 4: the modified fraction ---------------------------------
    mp.panel("C", width=180, height=112, pad_left=48, pad_top=16,
             margin_right=26, margin_bottom=32)
    ax = plt.gca()
    ax.scatter(d.rho_lower, d.rho_upper, s=4, facecolor=C_ZERO,
               edgecolor="none", alpha=0.5)
    lo = min(d.rho_lower.min(), d.rho_upper.min())
    hi = max(d.rho_lower.max(), d.rho_upper.max())
    ax.plot([lo, hi], [lo, hi], color=C_RULE, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$\\rho$ lower bound   $\\max_j m_j$", fontsize=6.5)
    ax.set_ylabel("$\\rho$ upper bound   $\\min(1, S)$", fontsize=6.5)
    _bare(ax)
    ax.text(0.04, 0.95, "on the diagonal:\n$\\rho$ determined ($P$ = 1)",
            transform=ax.transAxes, fontsize=5.0, color=C_RULE, va="top",
            linespacing=1.3)
    ax.set_title("Level 4   modified fraction $\\rho = P(K \\geq 1)$",
                 fontsize=7, fontweight="bold", pad=4, loc="left")

    # ---- D. level 5: conditional mean grade --------------------------------
    mp.panel("D", width=180, height=112, pad_left=46, pad_top=16,
             margin_right=0, margin_bottom=32)
    ax = plt.gca()
    q = d[d.P > 1]
    ax.scatter(q.P, q.cond_mean_upper, s=3 + 1.6 * q.B, facecolor=C_UNIQUE,
               edgecolor="white", linewidth=0.25, alpha=0.65, zorder=2)
    pp = np.arange(2, q.P.max() + 1)
    ax.plot(pp, pp, color=C_RULE, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axhline(1.0, color=C_RULE, lw=0.6, zorder=1)
    ax.set_xticks(range(2, int(q.P.max()) + 1, max(1, int(q.P.max()) // 6)))
    ax.set_xlim(1.4, q.P.max() + 0.8)
    ax.set_xlabel("Identity degeneracy   $P$", fontsize=6.5)
    ax.set_ylabel("$E[K \\mid K \\geq 1]$ upper bound   $S/\\max m$",
                  fontsize=6.5)
    _bare(ax)
    ax.text(0.55, 0.96, "ceiling $= P$", transform=ax.transAxes, ha="left",
            va="top", fontsize=5.0, color=C_RULE)
    ax.text(0.99, 0.03, "floor $= 1$   modified copies\ncarry exactly one mark",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0,
            color=C_RULE, linespacing=1.3)
    ax.set_title("Level 5   conditional mean grade", fontsize=7,
                 fontweight="bold", pad=4, loc="left")

    mp.save(out)


# ==============================================================================
# 10. FIGURE 4 — the hierarchy ladder for one protein
# ==============================================================================

def figure4_hierarchy_ladder(formnet_sites, summ, hier, focus, regime=None,
                             out=None, max_width=540, max_lp=MAX_M_LP):
    blk = formnet_sites[formnet_sites.protein_id == focus]
    mm = blk.loc[blk.state_class == "partial", "marginal"].dropna().to_numpy(float)
    if len(mm) == 0:
        print(f"    {focus}: P = 0, skipped")
        return
    h = grade_hierarchy(mm, B1=0, max_lp=max_lp)
    s = summ.loc[summ.protein_id == focus].iloc[0]
    hr = hier.loc[hier.protein_id == focus].iloc[0]
    env = h["envelope"]
    P = h["P"]
    S, mx = float(mm.sum()), float(mm.max())
    nm = PROTEIN_NAMES.get(focus, focus)
    out = out or f"formnet_figure4_{focus}"
    tag = f"   {REGIME_LABEL[regime]}" if regime else ""
    mp = MultiPanel(max_width=max_width)

    # ---- A. level 1: support geometry --------------------------------------
    mp.panel("A", width=196, height=112, pad_left=46, pad_top=26,
             margin_top=16, margin_right=26, margin_bottom=30)
    ax = plt.gca()
    ks = env.k.to_numpy()
    nc = env.n_configurations.to_numpy(float)
    ax.bar(ks, nc, width=0.66, color=C_UNMEAS, edgecolor=C_RULE, linewidth=0.4)
    if nc.max() > 50:
        ax.set_yscale("log")
    else:
        ax.set_ylim(0, nc.max() * 1.28)
    ax.set_xlabel("Grade $k$", fontsize=6.5)
    ax.set_ylabel("Admissible identities", fontsize=6.5)
    ax.set_xticks(range(0, P + 1, max(1, P // 6)))
    _bare(ax)
    ax.set_title(f"Level 1   support geometry", fontsize=7, fontweight="bold",
                 pad=4, loc="left")
    ax.text(0.97, 0.95, f"$k \\in [0, {P}]$\n$2^{{{P}}}$ = {thou(2**P)} identities",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.3,
            color=C_INK, linespacing=1.4)
    ax.text(0.97, 0.66, "a possibility statement:\nwhich grades occur in the lattice",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.0,
            color=C_RULE, style="italic", linespacing=1.3)

    # ---- B. level 3: the grade envelope ------------------------------------
    mp.panel("B", width=196, height=112, pad_left=48, pad_top=26,
             margin_top=16, margin_right=0, margin_bottom=30)
    ax = plt.gca()
    up = env.loc[env.upper > 0, "upper"]
    floor = 10.0 ** np.floor(np.log10(up.min())) if len(up) else 1e-8
    for _, r in env.iterrows():
        hi_ = max(r.upper, floor)
        forced = r.lower > 1e-12
        col = C_ZERO if forced else C_PARTIAL
        # light cap: the additional weight the data merely permit
        ax.vlines(r.k, max(r.lower, floor), hi_, color=col, lw=3.0, alpha=0.30,
                  capstyle="butt", zorder=2)
        # solid: the weight every feasible distribution must place here
        if forced:
            ax.vlines(r.k, floor, r.lower, color=col, lw=3.0,
                      capstyle="butt", zorder=3)
            ax.plot([r.k], [r.lower], marker="_", ms=5, mew=1.0, color=col,
                    zorder=5)
        ax.plot([r.k], [hi_], marker="_", ms=5, mew=1.0, color=col, alpha=0.55,
                zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(floor, 3.0)
    ax.set_xlim(-0.7, P + 0.7)
    ax.set_xticks(range(0, P + 1, max(1, P // 6)))
    ax.set_xlabel("Grade $k$", fontsize=6.5)
    ax.set_ylabel("Permitted weight $q_k$", fontsize=6.5)
    _bare(ax)
    ax.axhline(1.0, color=C_RULE, lw=0.4, ls=(0, (2, 2)), zorder=1)
    ax.legend(handles=[Line2D([], [], color=C_ZERO, lw=3.0,
                              label="forced   every $p$ places $\\geq q_k^{min}$"),
                       Line2D([], [], color=C_PARTIAL, lw=3.0, alpha=0.30,
                              label="permitted   up to $q_k^{max}$")],
              fontsize=5.0, frameon=False, loc="center right", handlelength=1.2,
              handletextpad=0.5, borderpad=0.2, labelspacing=0.5)
    ax.text(0.97, 0.95,
            f"$E[K]$ = {S:.3e}\nexact, not bounded",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.3,
            color=C_INK, linespacing=1.4)
    ax.set_title("Levels 2-3   exact moment, grade envelope", fontsize=7,
                 fontweight="bold", pad=4, loc="left")

    # ---- C. level 4: modified fraction -------------------------------------
    mp.panel("C", width=196, height=88, pad_left=46, pad_top=24,
             margin_right=26, margin_bottom=14)
    ax = plt.gca()
    lo_, hi_ = float(hr.rho_lower), float(hr.rho_upper)
    ax.hlines(0, lo_, hi_, color=C_ZERO, lw=4.5, capstyle="butt", zorder=3)
    ax.plot([lo_], [0], marker="|", ms=9, mew=1.2, color=C_ZERO, zorder=4)
    ax.plot([hi_], [0], marker="|", ms=9, mew=1.2, color=C_ZERO, zorder=4)
    ax.text(lo_, 0.22, f"$\\max_j m_j$\n{lo_:.3e}", ha="right", va="bottom",
            fontsize=5.0, color=C_INK, linespacing=1.4)
    ax.text(hi_, -0.16, f"$\\min(1,S)$\n{hi_:.3e}", ha="left", va="top",
            fontsize=5.0, color=C_INK, linespacing=1.4)
    ax.set_xscale("log")
    ax.set_xlim(lo_ * 0.35, hi_ * 3.2)
    ax.set_ylim(-1.15, 1.35)
    ax.set_yticks([])
    ax.set_xlabel("$\\rho = P(K \\geq 1)$", fontsize=6.5)
    _bare(ax, left=False)
    ax.text(0.5, -0.30, f"width {hi_ - lo_:.3e}", transform=ax.transAxes,
            ha="center", va="top", fontsize=5.2, color=C_RULE)
    ax.set_title("Level 4   modified fraction", fontsize=7, fontweight="bold",
                 pad=4, loc="left")

    # ---- D. level 5: conditional mean grade --------------------------------
    mp.panel("D", width=196, height=88, pad_left=48, pad_top=24,
             margin_right=0, margin_bottom=14)
    ax = plt.gca()
    cl, cu = float(hr.cond_mean_lower), float(hr.cond_mean_upper)
    ax.axvspan(1, P, color="#F2F1EE", zorder=0)
    ax.hlines(0, cl, cu, color=C_UNIQUE, lw=4.5, capstyle="butt", zorder=3)
    for v in (cl, cu):
        ax.plot([v], [0], marker="|", ms=9, mew=1.2, color=C_UNIQUE, zorder=4)
    ax.text(cl, 0.22, f"{cl:.3f}", ha="right", va="bottom", fontsize=5.2,
            color=C_INK)
    ax.text(cu, -0.16, f"{cu:.3f}", ha="left", va="top", fontsize=5.2,
            color=C_INK)
    ax.set_xlim(0.75, P + 0.35)
    ax.set_ylim(-1.15, 1.35)
    ax.set_yticks([])
    ax.set_xticks([1] + list(range(2, P + 1, max(1, P // 5))))
    ax.set_xlabel("$E[K \\mid K \\geq 1]$", fontsize=6.5)
    _bare(ax, left=False)
    ax.text(0.5, -0.30,
            f"of a possible [1, {P}]   normalised spread {float(hr.cond_spread):.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.2,
            color=C_RULE)
    ax.set_title("Level 5   conditional mean grade", fontsize=7,
                 fontweight="bold", pad=4, loc="left")

    mp.fig.text(0.012, 0.995,
                f"{nm} ({focus}){tag}     $R$ = {int(s.R)} · $B$ = {int(s.B)} · "
                f"$P$ = {P} · $U$ = {int(s.U)}",
                fontsize=7.6, fontweight="bold", color=C_INK, ha="left", va="top")
    mp.save(out)


# ==============================================================================
# 6. FIGURE 1 — the proteome
# ==============================================================================

def figure1_proteome(summ, formnet_sites, highlight=("Q09666", "Q15149", "P12270"),
                     out="formnet_figure1_proteome", max_width=540):
    counts = formnet_sites["state_class"].value_counts()
    total = int(counts.sum())
    quant = summ[summ.M > 0].copy()
    mp = MultiPanel(max_width=max_width)

    # ---- A. coordinate ledger ----------------------------------------------
    mp.panel("A", width=140, height=88, pad_left=30, pad_top=14,
             margin_right=10, margin_bottom=54)
    ax = plt.gca()
    left = 0.0
    for st in STATE_ORDER:
        w = counts.get(st, 0) / total
        ax.barh(0, w, left=left, height=0.72, color=STATE_COLORS[st],
                edgecolor="white", linewidth=0.9,
                hatch="//" if st == "unmeasured_X" else None)
        left += w
    cf = (counts.get("unmodified_0", 0) + counts.get("partial", 0)) / total
    ax.plot([0, cf], [0.50, 0.50], color=C_INK, lw=0.8, solid_capstyle="butt")
    ax.text(cf / 2, 0.58, f"directly constrained\n{100*cf:.2f}%", ha="center",
            va="bottom", fontsize=5.6, color=C_INK, linespacing=1.15)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.9, 1.15)
    ax.set_yticks([])
    ax.set_xticks([0, .25, .5, .75, 1.])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])
    ax.set_xlabel(f"Lysine coordinates (%)   n = {thou(total)}", fontsize=6.5)
    _bare(ax, left=False)
    for i, st in enumerate(STATE_ORDER):
        cx, cy = 0.02 + 0.52 * (i % 2), -1.62 - 0.32 * (i // 2)
        ax.add_patch(plt.Rectangle((cx, cy), 0.030, 0.20, transform=ax.transData,
                                   facecolor=STATE_COLORS[st], clip_on=False,
                                   edgecolor="white", lw=0.4,
                                   hatch="//" if st == "unmeasured_X" else None))
        ax.text(cx + 0.048, cy + 0.10, STATE_LABELS[st], fontsize=5.3,
                va="center", ha="left", color=C_INK)
    ax.set_title("Coordinate ledger", fontsize=7, fontweight="bold", pad=4,
                 loc="left")

    # ---- B. bit ledger ------------------------------------------------------
    mp.panel("B", width=98, height=88, pad_left=34, pad_top=14,
             margin_right=10, margin_bottom=26)
    ax = plt.gca()
    vals = [int(summ.C.sum()), int(summ.B.sum()), int(summ.M.sum())]
    ax.bar(range(3), vals, width=0.6, color=[C_INK, C_ZERO, C_PARTIAL])
    for i, v in enumerate(vals):
        ax.text(i, v * 1.06, thou(v), ha="center", va="bottom", fontsize=5.8,
                color=C_INK)
    ax.set_yscale("log")
    ax.set_ylim(max(vals[2], 1) * 0.35, vals[0] * 4)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Constrained", "Eliminated", "Surviving"], fontsize=6,
                       rotation=30, ha="right")
    ax.set_ylabel("Binary degrees of freedom", fontsize=6.5)
    _bare(ax)
    ax.set_title("Support compression", fontsize=7, fontweight="bold", pad=4,
                 loc="left")

    # ---- C. per-protein compression ----------------------------------------
    mp.panel("C", width=140, height=88, pad_left=34, pad_top=14,
             margin_right=0, margin_bottom=26)
    ax = plt.gca()
    b = summ["B"].to_numpy()
    ax.hist(b, bins=np.arange(0, b.max() + 2) - 0.5, color=C_ZERO,
            edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlabel("Bits eliminated per protein, $B$", fontsize=6.5)
    ax.set_ylabel("Proteins", fontsize=6.5)
    _bare(ax)
    row = summ.loc[summ.protein_id == highlight[0]]
    if len(row):
        ax.annotate(highlight[0], xy=(float(row.B.iloc[0]), 1.15), xytext=(0, 16),
                    textcoords="offset points", fontsize=5.8, color=C_INK,
                    ha="center", arrowprops=dict(arrowstyle="-", lw=0.5,
                                                 color=C_RULE, shrinkA=1, shrinkB=1))
    ax.set_title(f"n = {thou(len(summ))} proteins", fontsize=7,
                 fontweight="bold", pad=4, loc="left")

    # ---- D. solution-class census ------------------------------------------
    order_cls = ["trivially_determined", "uniquely_determined", "bounded_family"]
    lab = {"trivially_determined": "No quantitative\nmarginal\n($M$ = 0)",
           "uniquely_determined": "Unique weighted\nsolution\n($M$ = 1)",
           "bounded_family": "Bounded non-unique\nsolution\n($M \\geq$ 2)"}
    col = {"trivially_determined": C_UNMEAS, "uniquely_determined": C_UNIQUE,
           "bounded_family": C_PARTIAL}
    mp.panel("D", width=86, height=150, pad_left=104, pad_top=18,
             margin_right=16, margin_bottom=40)
    ax = plt.gca()
    v = [int((summ.solution_class == c).sum()) for c in order_cls]
    yp = np.arange(3)[::-1]
    ax.barh(yp, v, height=0.58, color=[col[c] for c in order_cls])
    for yy, vv in zip(yp, v):
        ax.text(max(vv, 1) * 1.25, yy, f"{vv:,}   {100*vv/sum(v):.1f}%",
                ha="left", va="center", fontsize=5.4, color=C_INK)
    ax.set_xscale("log")
    ax.set_xlim(0.6, max(v) * 22)
    ax.set_yticks(yp)
    ax.set_yticklabels([lab[c] for c in order_cls], fontsize=5.2, linespacing=1.35)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel("Proteins", fontsize=6.5)
    _bare(ax, left=False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Measured projection", fontsize=7, fontweight="bold", pad=4,
                 loc="left")

    # ---- E. the proteome map -----------------------------------------------
    mp.panel("E", width=232, height=150, pad_left=44, pad_top=18,
             margin_right=54, margin_bottom=40)
    ax = plt.gca()
    x = quant["B"].to_numpy(float)
    y = quant["W"].to_numpy(float)
    sz = 3 + 5.0 * quant["M"].to_numpy(float)
    det = y <= 0
    pos = y[~det]
    lin = 10.0 ** np.floor(np.log10(pos.min())) if pos.size else 1e-5
    rng = np.random.default_rng(0)
    jx = 1.0 + rng.uniform(-0.16, 0.16, size=x.size)
    ax.axhline(0, color=C_UNIQUE, lw=0.6, zorder=1)
    ax.scatter((x[~det] + 0.4) * jx[~det], y[~det], s=sz[~det], marker="o",
               facecolor=C_PARTIAL, edgecolor="white", linewidth=0.25, alpha=0.65,
               zorder=3, label="Bounded non-unique solution")
    ax.scatter((x[det] + 0.4) * jx[det], y[det], s=sz[det] * 1.5, marker="D",
               facecolor="none", edgecolor=C_UNIQUE, linewidth=0.55, alpha=0.8,
               zorder=4, label="Unique weighted solution")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=lin, linscale=0.55)
    ax.set_ylim(0, max(pos.max() * 3 if pos.size else 1, lin * 10))
    ax.set_xlabel("Support constraint   bits eliminated, $B$", fontsize=6.5)
    ax.set_ylabel("Weight freedom   $W = \\sum_j m_j - \\max_j m_j$", fontsize=6.5)
    _bare(ax)
    yt = [0.0] + [lin * 10.0 ** i for i in range(4)]
    ax.set_yticks([t for t in yt if t <= ax.get_ylim()[1]])
    for pid in highlight:
        r = quant.loc[quant.protein_id == pid]
        if not len(r) or float(r.W.iloc[0]) <= 0:
            continue
        ax.annotate(pid, xy=(float(r.B.iloc[0]) + 0.4, float(r.W.iloc[0])),
                    xytext=(7, 6), textcoords="offset points", fontsize=6,
                    color=C_INK, arrowprops=dict(arrowstyle="-", lw=0.5,
                                                 color=C_INK, shrinkA=0, shrinkB=1.5))
    h1, l1 = ax.get_legend_handles_labels()
    ax.add_artist(ax.legend(h1, l1, fontsize=5.4, frameon=False, loc="upper left",
                            handletextpad=0.4, borderpad=0.2, labelspacing=0.5,
                            scatterpoints=1))
    hs = [ax.scatter([], [], s=3 + 5.0 * k, facecolor=C_PARTIAL,
                     edgecolor="white", linewidth=0.25) for k in (1, 5, 10)]
    ax.legend(hs, ["1", "5", "10"], title="Constrained\ncoordinates, $M$",
              fontsize=5.4, title_fontsize=5.4, frameon=False, loc="center left",
              bbox_to_anchor=(1.01, 0.5), labelspacing=0.9, handletextpad=0.5,
              borderpad=0.2)
    ax.set_title(f"Proteins with $M \\geq$ 1   n = {thou(len(quant))}",
                 fontsize=7, fontweight="bold", pad=4, loc="left")

    mp.save(out)


# ==============================================================================
# 7. FIGURE 2 — one per exemplar
# ==============================================================================

def figure2_exemplar(formnet_sites, summ, focus, comparators=(),
                     domains=None, out=None, max_width=540, max_lp=MAX_M_LP):
    blk = formnet_sites[formnet_sites.protein_id == focus]
    if not len(blk):
        print(f"    {focus}: not present, skipped")
        return
    part = blk[blk.state_class == "partial"].sort_values("position")
    m = part["marginal"].to_numpy(float)
    labels = [f"K{int(p)}" for p in part["position"]]
    row = summ.loc[summ.protein_id == focus].iloc[0]
    if len(m) == 0:
        print(f"    {focus}: M = 0, no weight panels to draw, skipped")
        return

    env, _ = grade_envelope(m, max_lp)
    cfg = frechet_configuration_bounds(m, labels)
    mu = float(m.sum())
    out = out or f"formnet_figure2_{focus}"
    name = PROTEIN_NAMES.get(focus, focus)
    mp = MultiPanel(max_width=max_width)

    # ---- A. coordinate map along the sequence ------------------------------
    mp.panel("A", width=470, height=52, pad_left=8, pad_top=16,
             margin_right=10, margin_bottom=20)
    ax = plt.gca()
    hgt = {"unmeasured_X": .30, "modified_unquantified": .55,
           "unmodified_0": .95, "partial": .95}
    zo = {"unmeasured_X": 1, "modified_unquantified": 2, "unmodified_0": 3,
          "partial": 4}
    if domains:
        for a_, b_, lb in domains:
            ax.axvspan(a_, b_, color="#F0EEE9", zorder=0)
            ax.text((a_ + b_) / 2, 1.16, lb, ha="center", va="bottom",
                    fontsize=5.4, color=C_RULE)
    for st in STATE_ORDER[::-1]:
        sub = blk[blk.state_class == st]
        if len(sub):
            ax.vlines(sub["position"], 0, hgt[st], color=STATE_COLORS[st],
                      lw=0.55, zorder=zo[st])
    ax.set_xlim(0, blk["position"].max() * 1.005)
    ax.set_ylim(0, 1.32)
    ax.set_yticks([])
    ax.set_xlabel("Lysine position in FASTA sequence", fontsize=6.5)
    _bare(ax, left=False)
    ax.set_title(f"{name} ({focus})   $R$ = {int(row.R)} · $B$ = {int(row.B)} "
                 f"· $M$ = {int(row.M)} · $U$ = {int(row.U)}   "
                 f"log$_2|\\mathcal{{A}}(m)|$ = {int(row.bits_admissible_full)} bits",
                 fontsize=7, fontweight="bold", pad=4, loc="left")

    # ---- B. grade envelope --------------------------------------------------
    mp.panel("B", width=132, height=118, pad_left=46, pad_top=16,
             margin_right=22, margin_bottom=26)
    ax = plt.gca()
    up = env.loc[env.upper > 0, "upper"]
    floor = 10.0 ** np.floor(np.log10(up.min())) if len(up) else 1e-6
    for _, r in env.iterrows():
        col = C_ZERO if r.k == 0 else C_PARTIAL
        hi = max(r.upper, floor)
        if r.lower > 1e-12:
            ax.vlines(r.k, max(r.lower, floor), hi, color=col, lw=2.6,
                      capstyle="butt", zorder=3)
        else:
            ax.vlines(r.k, floor, hi, color=col, lw=2.6, alpha=0.35,
                      capstyle="butt", zorder=2)
        ax.plot([r.k], [hi], marker="_", ms=4.5, mew=0.9, color=col, zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(floor, 3.0)
    ax.set_xlim(-0.7, len(m) + 0.7)
    ax.set_xticks(range(0, len(m) + 1, max(1, len(m) // 6)))
    ax.set_xlabel("Grade $k$", fontsize=6.5)
    ax.set_ylabel("Permitted weight  $q_k$", fontsize=6.5)
    _bare(ax)
    ax.axhline(1.0, color=C_RULE, lw=0.4, ls=(0, (2, 2)), zorder=1)
    r0 = env.iloc[0]
    ax.annotate(f"[{r0.lower:.4f}, {r0.upper:.4f}]", xy=(0, r0.upper),
                xytext=(9, -2), textcoords="offset points", fontsize=5.2,
                color=C_ZERO, ha="left", va="center")
    ax.text(0.97, 0.87, f"$E[k]$ = {mu:.2e}\nfixed, not bounded",
            transform=ax.transAxes, fontsize=5.2, color=C_RULE, ha="right",
            va="top", linespacing=1.3)
    ax.set_title("Grade envelope", fontsize=7, fontweight="bold", pad=4,
                 loc="left")

    # ---- C. configuration bound ladder -------------------------------------
    mp.panel("C", width=238, height=118, pad_left=112, pad_top=16,
             margin_right=0, margin_bottom=26)
    ax = plt.gca()
    top = cfg.head(TOP_CONFIGS).iloc[::-1].reset_index(drop=True)
    for i, r in top.iterrows():
        nec = r.status == "necessary"
        col = C_ZERO if r.k == 0 else C_PARTIAL
        ax.hlines(i, r.lower, r.upper, color=col, lw=2.2,
                  alpha=1.0 if nec else 0.45, capstyle="butt")
        ax.plot([r.upper], [i], marker="|", ms=4.5, mew=0.9, color=col)
        ax.plot([r.lower], [i], marker="|", ms=4.5, mew=0.9, color=col,
                alpha=1.0 if nec else 0.45)
        if nec:
            ax.annotate(f"[{r.lower:.4f}, {r.upper:.4f}]", xy=(r.lower, i),
                        xytext=(-6, 0), textcoords="offset points",
                        fontsize=5.2, color=col, ha="right", va="center")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["label"], fontsize=5.6)
    ax.set_xscale("symlog", linthresh=floor, linscale=0.45)
    ax.set_xlim(0, 3.0)
    ax.set_ylim(-0.8, len(top) - 0.2)
    ax.set_xlabel("Permitted weight interval", fontsize=6.5)
    ax.set_xticks([0, floor, floor * 10, floor * 100, 1])
    _bare(ax)
    ax.legend(handles=[Line2D([], [], color=C_ZERO, lw=2.2,
                              label="Required (lower bound > 0)"),
                       Line2D([], [], color=C_PARTIAL, lw=2.2, alpha=0.45,
                              label="Permitted (capped)")],
              fontsize=5.4, frameon=False, loc="center right", handlelength=1.4,
              handletextpad=0.5, borderpad=0.2)
    ax.set_title("Configuration bounds", fontsize=7, fontweight="bold", pad=4,
                 loc="left")

    # ---- D. cross-protein ceilings -----------------------------------------
    mp.panel("D", width=470, height=112, pad_left=48, pad_top=16,
             margin_right=10, margin_bottom=24)
    ax = plt.gca()
    cyc = [C_ZERO, C_PARTIAL, "#5B7A54", "#7A5C86", "#A8843F"]
    lows = []
    for i, pid in enumerate([focus] + [p for p in comparators if p != focus]):
        pb = formnet_sites[formnet_sites.protein_id == pid]
        mm = pb.loc[pb.state_class == "partial", "marginal"].dropna().to_numpy(float)
        if len(mm) == 0:
            continue
        e, _ = grade_envelope(mm, max_lp)
        ks = e.k.to_numpy()
        tail = np.array([e.loc[e.k >= k, "upper"].max() for k in ks])
        tail = np.maximum(tail, 1e-12)
        lows.append(tail.min())
        ax.plot(ks, tail, marker="o", ms=2.2, lw=0.9, color=cyc[i % len(cyc)],
                label=f"{PROTEIN_NAMES.get(pid, pid)} ({pid})  $M$={len(mm)}")
    ax.set_yscale("log")
    ax.set_ylim(10.0 ** np.floor(np.log10(min(lows))) if lows else 1e-6, 4.0)
    ax.set_xlabel("Grade threshold $k$", fontsize=6.5)
    ax.set_ylabel("$U[\\,P(k' \\geq k)\\,]$", fontsize=6.5)
    _bare(ax)
    ax.legend(fontsize=5.4, frameon=False, ncol=2, loc="lower left",
              handlelength=1.4, handletextpad=0.5, columnspacing=1.2,
              borderpad=0.2)
    ax.set_title("Ceiling on higher-grade combinations", fontsize=7,
                 fontweight="bold", pad=4, loc="left")

    mp.save(out)


# ==============================================================================
# 8. MAIN
# ==============================================================================

def main():
    setup_matplotlib()
    rule("FormNet — ACETYLATION EXPERIMENT 2 — FULL RECOVERY")
    print()

    fs = load_formnet_sites()
    need = {"protein_id", "site_id", "position", "state_class", "marginal"}
    missing = need - set(fs.columns)
    if missing:
        raise ValueError(f"formnet_sites missing columns: {missing}")
    print(f"formnet_sites: {fs.shape[0]:,} coordinates, "
          f"{fs.protein_id.nunique():,} protein groups")
    print()

    print("Building per-protein summary ...")
    summ = protein_summary(fs)
    summ.to_csv("formnet_protein_summary.csv", index=False)
    print(f"  wrote formnet_protein_summary.csv  ({len(summ):,} rows)")
    print()

    section1_coverage(summ, fs)
    section2_boundary(summ)
    section3_identity_vs_weight(summ)
    env_df, cfg_df, targets = section4_grades(fs, summ, EXEMPLARS)

    print("Building the grade hierarchy ...")
    hier = hierarchy_table(fs, summ)
    hier.to_csv("formnet_grade_hierarchy.csv", index=False)
    print(f"  wrote formnet_grade_hierarchy.csv  ({len(hier):,} rows)")
    print()
    regimes = select_exemplars(hier, summ)
    section5_hierarchy(fs, summ, hier, regimes)

    if len(env_df):
        env_df.to_csv("formnet_grade_envelopes.csv", index=False)
        print("wrote formnet_grade_envelopes.csv")
    if len(cfg_df):
        cfg_df.to_csv("formnet_configuration_bounds.csv", index=False)
        print("wrote formnet_configuration_bounds.csv")
    print()

    if MAKE_FIGURES:
        rule("FIGURES")
        hl = tuple(summ.nlargest(3, "B").protein_id) if not set(
            EXEMPLARS) & set(summ.protein_id) else tuple(EXEMPLARS[:3])
        figure1_proteome(summ, fs, highlight=hl)
        for pid in targets:
            figure2_exemplar(fs, summ, focus=pid,
                             comparators=[p for p in targets if p != pid][:3])
        figure3_hierarchy_map(hier, summ, regimes)
        for regime, pid in regimes.items():
            figure4_hierarchy_ladder(fs, summ, hier, focus=pid, regime=regime)
        print()

    rule("DONE")
    return {"formnet_sites": fs, "protein_summary": summ,
            "grade_envelopes": env_df, "configuration_bounds": cfg_df,
            "grade_hierarchy": hier, "regimes": regimes}


if __name__ == "__main__":
    RESULTS = main()
