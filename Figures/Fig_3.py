import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import linprog

# ----------------------------------------------------------------------
SITES_CSV = "formnet_sites.csv"

# one protein per class, chosen by the selection rules in the pipeline script
COLUMNS = [
    dict(pid="Q9NZM1", cls="support only",          rule="$M$ = 0, largest $B$"),
    dict(pid="Q8WZ42", cls="exact projection",      rule="$M$ = 1, largest $U$"),
    dict(pid="P12270", cls="bounded, grade forced", rule="$P \\geq$ 5, smallest spread"),
    dict(pid="Q13813", cls="bounded, permissive",   rule="$P \\geq$ 5, largest spread"),
]

# VERIFY these against UniProt before submission
PROTEIN_NAMES = {"Q9NZM1": "MYOF", "Q8WZ42": "TTN",
                 "P12270": "TPR",  "Q13813": "SPTAN1"}

CLASS_N = {"support only": (3381, 78.6), "exact projection": (685, 15.9),
           "bounded, grade forced": (180, 4.2), "bounded, permissive": (54, 1.3)}
# ----------------------------------------------------------------------

PT_PER_INCH = 72.0

C_ZERO    = "#3B6E8F"
C_PARTIAL = "#C8562B"
C_UNQ     = "#8A7A3D"
C_UNMEAS  = "#E3E0D9"
C_UNIQUE  = "#4E6E58"
C_INK     = "#1A1A1A"
C_RULE    = "#8C8C8C"

STATE_COLORS = {"unmodified_0": C_ZERO, "partial": C_PARTIAL,
                "modified_unquantified": C_UNQ, "unmeasured_X": C_UNMEAS}
STATE_LABELS = {"unmodified_0": "Boundary-fixed (0)",
                "partial": "Quantitatively constrained",
                "modified_unquantified": "Modified, unquantified",
                "unmeasured_X": "Unmeasured"}
STATE_ORDER = ["unmodified_0", "partial", "modified_unquantified", "unmeasured_X"]


class MultiPanel:
    """540 pt = 7.5 in = 190 mm; every fontsize below is a literal point size."""

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
        x, y = self._x + margin_left + pad_left, self._row_top + margin_top + pad_top
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
        print(f"wrote {stem}.pdf and {stem}.png")


def setup_matplotlib():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def _bare(ax, left=True, bottom=True, labelsize=5.0):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(length=2.0, width=0.5, labelsize=labelsize, pad=1.5)
    for s in ("left", "bottom"):
        if ax.spines[s].get_visible():
            ax.spines[s].set_linewidth(0.6)


# ======================================================================
# exact bounds
# ======================================================================

def grade_envelope(m):
    """Exact [q_k^min, q_k^max] by LP over P(m). Returns (P+1, 2)."""
    m = np.asarray(m, float)
    P = len(m)
    if P == 0:
        return np.array([[1.0, 1.0]])
    S = np.array(list(itertools.product([0, 1], repeat=P)), float)
    A = np.vstack([np.ones(len(S)), S.T])
    b = np.concatenate([[1.0], m])
    k = S.sum(1)
    out = []
    for j in range(P + 1):
        c = (k == j).astype(float)
        lo = linprog(c, A_eq=A, b_eq=b, bounds=(0, 1), method="highs").fun
        hi = -linprog(-c, A_eq=A, b_eq=b, bounds=(0, 1), method="highs").fun
        out.append([float(lo), float(hi)])
    env = np.array(out)
    # closed-form cross-checks
    S_, mx = float(m.sum()), float(m.max())
    assert abs(env[0, 0] - max(0.0, 1 - S_)) < 1e-9
    assert abs(env[0, 1] - (1 - mx)) < 1e-9
    assert abs(env[1, 0] - max(0.0, 2 * mx - S_)) < 1e-9
    return env


def protein_block(sites, pid):
    blk = sites[sites.protein_id == pid].sort_values("position")
    if not len(blk):
        raise KeyError(f"{pid} not in {SITES_CSV}")
    m = blk.loc[blk.state_class == "partial", "marginal"].dropna().to_numpy(float)
    R = int(len(blk))
    B = int((blk.state_class == "unmodified_0").sum())
    P = len(m)
    S = float(m.sum()) if P else 0.0
    mx = float(m.max()) if P else 0.0
    return dict(pid=pid, blk=blk, m=m, R=R, B=B, P=P, U=R - B - P,
                S=S, m_max=mx, env=grade_envelope(m),
                bits=R - B,
                rho=(mx, min(1.0, S)) if P else (0.0, 0.0),
                cond=(max(1.0, S), S / mx) if P else (np.nan, np.nan),
                spread=((S / mx - 1) / (P - 1)) if P > 1 else 0.0)


# ======================================================================
# figure
# ======================================================================

def build(stem="formnet_resolution_gradient", max_width=540):
    sites = pd.read_csv(SITES_CSV)
    cols = []
    for spec in COLUMNS:
        d = protein_block(sites, spec["pid"])
        d.update(spec)
        cols.append(d)

    # shared log floor across every column, so panel b is comparable
    pos = [d["env"][d["env"][:, 1] > 0, 1].min() for d in cols]
    floor = 10.0 ** np.floor(np.log10(min(pos)))

    mp = MultiPanel(max_width=max_width)
    for i, d in enumerate(cols):
        first = (i == 0)
        mp.panel(chr(ord("a") + i), width=106, height=268,
                 pad_left=40 if first else 16, pad_top=76,
                 margin_right=6, margin_bottom=58)
        host = plt.gca()
        host.axis("off")
        host.set_xlim(0, 1)
        host.set_ylim(0, 1)

        n, pct = CLASS_N.get(d["cls"], (np.nan, np.nan))
        nm = PROTEIN_NAMES.get(d["pid"], d["pid"])
        host.text(0, 1.246, d["cls"], transform=host.transAxes, fontsize=6.6,
                  fontweight="bold", color=C_INK, ha="left", va="bottom")
        host.text(0, 1.194, f"n = {n:,}  ({pct:.1f}%)   {d['rule']}",
                  transform=host.transAxes, fontsize=4.9, color=C_RULE,
                  ha="left", va="bottom")
        host.text(0, 1.127, f"{nm} ({d['pid']})", transform=host.transAxes,
                  fontsize=6.0, fontweight="bold", color=C_INK, ha="left",
                  va="bottom")
        host.text(0, 1.030,
                  f"$R$={d['R']}  $B$={d['B']}  $M$={d['P']}  $U$={d['U']}\n"
                  f"log$_2|\\mathcal{{A}}(m)|$ = {d['bits']:,} bits",
                  transform=host.transAxes, fontsize=4.9, color=C_INK,
                  ha="left", va="bottom", linespacing=1.4)

        # ---- row a: coverage along the sequence -------------------------
        a = host.inset_axes([0, 0.885, 1, 0.075])
        hgt = {"unmeasured_X": .34, "modified_unquantified": .58,
               "unmodified_0": 1.0, "partial": 1.0}
        zo = {"unmeasured_X": 1, "modified_unquantified": 2,
              "unmodified_0": 3, "partial": 4}
        for st in STATE_ORDER[::-1]:
            sub = d["blk"][d["blk"].state_class == st]
            if len(sub):
                a.vlines(sub["position"], 0, hgt[st], color=STATE_COLORS[st],
                         lw=0.45, zorder=zo[st])
        a.set_xlim(0, d["blk"]["position"].max() * 1.01)
        a.set_ylim(0, 1.12)
        a.set_yticks([])
        a.set_xticks([0, int(d["blk"]["position"].max())])
        _bare(a, left=False, labelsize=4.6)
        a.set_xlabel("lysine position", fontsize=5.0, labelpad=0.5)

        # ---- row b: grade envelope --------------------------------------
        b = host.inset_axes([0, 0.38, 1, 0.42])
        env, P = d["env"], d["P"]
        for j in range(P + 1):
            lo, hi = env[j]
            forced = lo > 1e-12
            col = C_ZERO if forced else C_PARTIAL
            b.vlines(j, max(lo, floor), max(hi, floor), color=col, lw=2.4,
                     alpha=0.30, capstyle="butt", zorder=2)
            if forced:
                b.vlines(j, floor, lo, color=col, lw=2.4, capstyle="butt",
                         zorder=3)
                b.plot([j], [lo], marker="_", ms=4, mew=0.9, color=col, zorder=5)
            b.plot([j], [max(hi, floor)], marker="_", ms=4, mew=0.9, color=col,
                   alpha=0.55, zorder=4)
        b.set_yscale("log")
        b.set_ylim(floor, 3.0)
        b.set_xlim(-0.7, max(P, 1) + 0.7)
        b.set_xticks(range(0, P + 1, max(1, P // 4)) if P else [0])
        _bare(b, left=first)
        if not first:
            b.tick_params(left=False, labelleft=False, which="both")
        else:
            b.set_ylabel("permitted weight $q_k$", fontsize=5.4, labelpad=1)
        b.set_xlabel("grade $k$", fontsize=5.4, labelpad=0.5)
        b.axhline(1.0, color=C_RULE, lw=0.35, ls=(0, (2, 2)), zorder=1)

        if P == 0:
            b.text(0.97, 0.42, "no interior marginal\n$q_0$ = 1 by construction\n"
                                "no weight information",
                   transform=b.transAxes, ha="right", va="center",
                   fontsize=4.9, color=C_RULE, linespacing=1.5)
        else:
            b.text(0.97, 0.06, f"$E[K]$ = {d['S']:.2e}", transform=b.transAxes,
                   ha="right", va="bottom", fontsize=4.9, color=C_INK)

        # ---- row c: conditional mean grade ------------------------------
        c = host.inset_axes([0, 0.115, 1, 0.080])
        if P == 0:
            c.axis("off")
            c.text(0.5, 0.5, "not defined", transform=c.transAxes, ha="center",
                   va="center", fontsize=4.9, color=C_RULE, style="italic")
        else:
            cl, cu = d["cond"]
            c.axvspan(1, max(P, 1.02), color="#F2F1EE", zorder=0)
            c.hlines(0, cl, cu, color=C_UNIQUE, lw=3.6, capstyle="butt", zorder=3)
            for v in (cl, cu):
                c.plot([v], [0], marker="|", ms=7, mew=1.1, color=C_UNIQUE,
                       zorder=4)
            c.text(cl, 0.5, f"{cl:.2f}", ha="right", va="bottom", fontsize=4.7,
                   color=C_INK)
            c.text(cu, -0.6, f"{cu:.2f}", ha="left", va="top", fontsize=4.7,
                   color=C_INK)
            c.set_xlim(0.6, max(P, 1) + 0.4)
            c.set_ylim(-1.9, 1.9)
            c.set_yticks([])
            c.set_xticks([1, max(P, 1)] if P > 1 else [1])
            _bare(c, left=False, labelsize=4.6)
            c.set_xlabel("$E[K \\mid K \\geq 1]$   of $[1, M]$", fontsize=5.0,
                         labelpad=0.5)
            host.text(0.5, 0.030, f"normalised spread {d['spread']:.3f}",
                      transform=host.transAxes, ha="center", va="bottom",
                      fontsize=4.9, color=C_RULE)

    # shared legends, drawn once under the first column
    fig = mp.fig
    h_state = [Line2D([], [], color=STATE_COLORS[s], lw=2.4, label=STATE_LABELS[s])
               for s in STATE_ORDER]
    fig.legend(handles=h_state, fontsize=4.9, frameon=False, ncol=4,
               loc="lower left", bbox_to_anchor=(0.012, 0.008),
               handlelength=1.2, handletextpad=0.5, columnspacing=1.4)
    fig.legend(handles=[Line2D([], [], color=C_ZERO, lw=2.4,
                               label="forced   every $p$ places $\\geq q_k^{min}$"),
                        Line2D([], [], color=C_PARTIAL, lw=2.4, alpha=0.30,
                               label="permitted   up to $q_k^{max}$")],
               fontsize=4.9, frameon=False, ncol=2, loc="lower right",
               bbox_to_anchor=(0.99, 0.008), handlelength=1.2,
               handletextpad=0.5, columnspacing=1.4)

    fig.text(0.012, 0.062,
             "Resolution decreases left to right. Identity is stated numerically in "
             "each header; every panel reports weight.\nPanel b shares one log scale "
             "across all four columns. All bounds are exact, by linear programming "
             "over $\\mathcal{P}(m)$.",
             fontsize=5.0, color="#4A4A4A", ha="left", va="bottom",
             linespacing=1.6)
    mp.save(stem)
    return cols


setup_matplotlib()
COLS = build()

print("\nColumn summary")
print(f"{'protein':<10}{'R':>6}{'B':>5}{'M':>4}{'U':>7}{'bits':>8}"
      f"{'E[K]':>12}{'spread':>9}{'forced grades':>16}")
for d in COLS:
    forced = [j for j in range(d["P"] + 1) if d["env"][j, 0] > 1e-12]
    print(f"{d['pid']:<10}{d['R']:>6}{d['B']:>5}{d['P']:>4}{d['U']:>7}"
          f"{d['bits']:>8,}{d['S']:>12.3e}{d['spread']:>9.3f}{str(forced):>16}")

try:
    from IPython.display import Image, display
    display(Image("formnet_resolution_gradient.png", width=1000))
except Exception:
    pass
