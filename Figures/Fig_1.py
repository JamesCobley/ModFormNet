import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

PT_PER_INCH = 72.0

# ----------------------------------------------------------------------
# palette
# ----------------------------------------------------------------------

C_ZERO    = "#3B6E8F"   # boundary-fixed coordinate  (m = 0 or 1)
C_PARTIAL = "#C8562B"   # quantitatively constrained coordinate (0 < m < 1)
C_UNIQUE  = "#4E6E58"   # unique weighted solution
C_INK     = "#1A1A1A"
C_RULE    = "#8C8C8C"
C_PALE    = "#D8D5CE"   # excluded by the data
C_BOX     = "#F4F2EE"
C_MATHBG  = "#EDF1F4"
C_MATHED  = "#C9D6DE"

MONO = {"family": "DejaVu Sans Mono"}


# ======================================================================
# minimal pixel-exact multipanel layout (replaces cnsplots)
# ======================================================================

class MultiPanel:
    """
    Panels flow left to right in POINTS and wrap at max_width.
    max_width=540 pt = 7.5 in = 190 mm, a Nature double-column figure, so the
    font sizes below are literal point sizes.
    """

    def __init__(self, max_width=540, label_size=8.5):
        self.max_width = float(max_width)
        self.label_size = label_size
        self.fig = plt.figure(figsize=(max_width / PT_PER_INCH, 1.0),
                              facecolor="white")
        self._axes, self._labels = [], []
        self._x = self._row_top = self._row_h = 0.0

    def panel(self, label=None, width=150, height=120, pad_left=30, pad_top=16,
              margin_right=0, margin_bottom=0, margin_left=0, margin_top=0):
        cell_w = margin_left + pad_left + width + margin_right
        cell_h = margin_top + pad_top + height + margin_bottom
        if self._x > 0 and self._x + cell_w > self.max_width + 1e-6:
            self._row_top += self._row_h
            self._x = self._row_h = 0.0
        x = self._x + margin_left + pad_left
        y = self._row_top + margin_top + pad_top
        ax = self.fig.add_axes([0, 0, 1, 1])
        self._axes.append((ax, x, y, float(width), float(height)))
        if label:
            self._labels.append((label, self._x + margin_left,
                                 self._row_top + margin_top))
        self._x += cell_w
        self._row_h = max(self._row_h, cell_h)
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
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans",
                            "DejaVu Sans"],
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


# ======================================================================
# lattice
# ======================================================================

def combinations(R=3):
    return ["".join(map(str, c)) for c in itertools.product([0, 1], repeat=R)]


def lattice_positions(R=3, xgap=1.30):
    pos = {}
    for k in range(R + 1):
        rank = sorted((c for c in combinations(R) if c.count("1") == k),
                      reverse=True)
        for i, c in enumerate(rank):
            pos[c] = ((i - (len(rank) - 1) / 2) * xgap, float(k))
    return pos


def hamming_edges(R=3):
    return [(a, b) for a, b in itertools.combinations(combinations(R), 2)
            if sum(u != v for u, v in zip(a, b)) == 1]


def excluded_by(m, R=3):
    """
    Combinations the data remove from the admissible support.

    A boundary marginal m_j = 0 excludes every combination with coordinate j
    modified; m_j = 1 excludes every combination with coordinate j unmodified.
    Interior marginals exclude nothing — they constrain weight only.
    """
    out = set()
    for c in combinations(R):
        for j, v in enumerate(m):
            if v is None:
                continue
            if (v == 0 and c[j] == "1") or (v == 1 and c[j] == "0"):
                out.add(c)
    return out


def draw_lattice(ax, m=None, R=3, accent=C_PARTIAL, node_w=0.86, node_h=0.40,
                 fontsize=5.4, show_k=False, xgap=1.30, emphasise=()):
    pos = lattice_positions(R, xgap)
    ex = excluded_by(m, R) if m is not None else set()

    for a, b in hamming_edges(R):
        dead = a in ex or b in ex
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=C_PALE if dead else "#BFBCB5", lw=0.45, zorder=1)

    for c, (x, y) in pos.items():
        if c in ex:
            face, edge, txt, lw = "white", C_PALE, C_PALE, 0.5
        elif c in emphasise:
            face, edge, txt, lw = accent, accent, "white", 0.7
        else:
            face, edge, txt, lw = "white", accent, C_INK, 0.7
        ax.add_patch(FancyBboxPatch(
            (x - node_w / 2, y - node_h / 2), node_w, node_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=face, edgecolor=edge, linewidth=lw, zorder=3))
        ax.text(x, y, c, ha="center", va="center", fontsize=fontsize,
                color=txt, zorder=4, **MONO)

    if show_k:
        for k in range(R + 1):
            n = sum(1 for c in combinations(R) if c.count("1") == k)
            ax.text(2.05, k, f"$k$ = {k}    $\\binom{{{R}}}{{{k}}}$ = {n}",
                    ha="left", va="center", fontsize=5.4, color=C_RULE)

    ax.set_xlim(-2.15, 3.55 if show_k else 2.15)
    ax.set_ylim(-0.55, R + 0.55)
    ax.axis("off")


def draw_marginal_stems(ax, m, R=3, accent=C_PARTIAL, fs=5.0):
    """Measured site marginals: bar for interior, dot+0 for boundary, X for unmeasured."""
    ax.axhline(0, color=C_INK, lw=0.5)
    for j, v in enumerate(m):
        if v is None:
            ax.text(j, 0.42, "X", ha="center", va="center", fontsize=6.2,
                    color=C_RULE, fontweight="bold")
        else:
            col = C_ZERO if v in (0, 1) else accent
            if v > 0:
                ax.bar(j, v, width=0.45, color=col, edgecolor="none")
            else:
                ax.plot([j], [0], marker="o", ms=2.6, color=col, zorder=4)
                ax.text(j, 0.10, "0", ha="center", va="bottom", fontsize=fs,
                        color=col)
    ax.set_xticks(range(R))
    ax.set_xticklabels([f"$m_{j+1}$" for j in range(R)], fontsize=fs)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "", "1"], fontsize=fs)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.65, R - 0.35)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.tick_params(length=1.8, width=0.5, pad=1.5)
    ax.set_ylabel("marginal", fontsize=fs, labelpad=1)


# ======================================================================
# figure
# ======================================================================

CASES = [
    dict(key="b", title="Weakest case", m=[None, None, 0.25], accent=C_PARTIAL,
         lines=[r"$p_{001}+p_{011}+p_{101}+p_{111} = 0.25$",
                r"$p_{000}+p_{010}+p_{100}+p_{110} = 0.75$"],
         caption="All eight combinations remain admissible.\n"
                 "Only the total weight on coordinate 3 is fixed;\n"
                 "111 is admissible but cannot exceed 0.25."),
    dict(key="c", title="Boundary constraint", m=[None, None, 0], accent=C_ZERO,
         lines=["excluded:  001,  011,  101,  111",
                r"admissible:  $\{000, 010, 100, 110\}$"],
         caption="One boundary marginal removes half the\n"
                 "combinations exactly, including the all-modified\n"
                 "combination. Exclusion is a statement about what\n"
                 "the data admit, not about what exists."),
    dict(key="d", title="Bounded, non-unique", m=[0, 0.25, 0.25],
         accent=C_PARTIAL,
         lines=[r"$p_{011} = t$,   $0 \leq t \leq 0.25$",
                r"$p_{010} = p_{001} = 0.25 - t$",
                r"$p_{000} = 0.50 + t$"],
         caption="Four combinations survive. One free parameter\n"
                 "remains: the feasible set is a bounded segment,\n"
                 "not a point. Every combination is bounded,\n"
                 "none is determined."),
    dict(key="e", title="Unique weighted solution", m=[0, 0, 0.25],
         accent=C_UNIQUE,
         lines=[r"$p_{000} = 0.75$,   $p_{001} = 0.25$"],
         caption="Two combinations survive and the remaining\n"
                 "marginal fixes their weights. The polytope has\n"
                 "collapsed to a single point: the data determine\n"
                 "the weighting exactly."),
]


def build(stem="formnet_schematic", max_width=540, R=3):
    mp = MultiPanel(max_width=max_width)

    # ---------------- a. combinatorics and the polytope ---------------
    mp.panel("a", width=500, height=176, pad_left=22, pad_top=20,
             margin_right=0, margin_bottom=20)
    host = plt.gca()
    host.axis("off")
    host.set_xlim(0, 1)
    host.set_ylim(0, 1)
    host.set_title("Combinations, $k$-grading and the feasible polytope",
                   fontsize=7.2, fontweight="bold", pad=6, loc="left")

    ax = host.inset_axes([-0.03, 0.04, 0.46, 0.94])
    draw_lattice(ax, m=None, R=R, accent=C_INK, show_k=True, fontsize=5.2)
    ax.annotate("", xy=(-2.02, R + 0.30), xytext=(-2.02, -0.30),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=C_RULE,
                                mutation_scale=6))
    ax.text(-2.18, R / 2, "modification count $k$", rotation=90, ha="right",
            va="center", fontsize=5.6, color=C_RULE)
    ax.text(0, -0.52, "poles: all-unmodified (000), all-modified (111)",
            ha="center", va="top", fontsize=5.3, color=C_RULE, style="italic")

    bx = host.inset_axes([0.50, 0.02, 0.50, 0.96])
    bx.axis("off")
    bx.add_patch(FancyBboxPatch(
        (0.005, 0.02), 0.99, 0.96,
        boxstyle="round,pad=0.008,rounding_size=0.03", transform=bx.transAxes,
        facecolor=C_MATHBG, edgecolor=C_MATHED, linewidth=0.6, zorder=0))
    for yy, txt, fs, col, style in [
        (0.93, "The unknown weighting over combinations", 6.4, C_INK, "bold"),
        (0.80, r"$\Omega_R = \{0,1\}^R$,   $|\Omega_R| = 2^R$", 6.2, C_INK, None),
        (0.665, r"$p_x \geq 0$,   $\sum_{x \in \Omega_R} p_x = 1$", 6.2, C_INK, None),
        (0.505, r"$m_j \;=\; \sum_{x \,:\, x_j = 1} p_x$", 6.6, C_INK, None),
    ]:
        bx.text(0.5, yy, txt, transform=bx.transAxes, ha="center", va="top",
                fontsize=fs, color=col,
                fontweight="bold" if style == "bold" else "normal")
    bx.text(0.5, 0.355,
            "Each site marginal is one linear constraint on the\n"
            "weighting. Their intersection with non-negativity\n"
            "and normalisation is a convex polytope.",
            transform=bx.transAxes, ha="center", va="top", fontsize=5.5,
            color="#4A4A4A", linespacing=1.6)
    bx.text(0.5, 0.175,
            r"$\mathcal{P}(m) = \{\, p \in \mathbb{R}^{2^R}_{\geq 0} \;:\; "
            r"\sum_x p_x = 1, \; \sum_{x : x_j = 1} p_x = m_j \,\}$",
            transform=bx.transAxes, ha="center", va="top", fontsize=5.9,
            color=C_INK)
    bx.text(0.5, 0.075,
            "every point of $\\mathcal{P}(m)$ is data-admissible;\n"
            "the data prefer none of them",
            transform=bx.transAxes, ha="center", va="top", fontsize=5.2,
            color=C_RULE, style="italic", linespacing=1.5)

    # ---------------- b-e. accumulating constraints -------------------
    for i, cs in enumerate(CASES):
        mp.panel(cs["key"], width=232, height=178, pad_left=22, pad_top=20,
                 margin_right=0 if i % 2 else 24, margin_bottom=20)
        ax = plt.gca()
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(cs["title"], fontsize=7.2, fontweight="bold", pad=6,
                     loc="left")

        lat = ax.inset_axes([-0.06, 0.24, 0.52, 0.78])
        draw_lattice(lat, m=cs["m"], R=R, accent=cs["accent"], fontsize=5.0,
                     xgap=1.22,
                     emphasise={"000", "001"} if cs["key"] == "e" else ())

        mg = ax.inset_axes([0.55, 0.72, 0.40, 0.24])
        draw_marginal_stems(mg, cs["m"], R=R, accent=cs["accent"])

        mtxt = ", ".join("X" if v is None else f"{v:g}" for v in cs["m"])
        ax.text(0.55, 0.655, f"$m$ = ({mtxt})", transform=ax.transAxes,
                fontsize=6.0, color=C_INK, ha="left", va="top")
        ax.text(0.55, 0.575, "\n".join(cs["lines"]), transform=ax.transAxes,
                fontsize=5.4, color=C_INK, ha="left", va="top", linespacing=1.7,
                bbox=dict(boxstyle="round,pad=0.45", facecolor=C_BOX,
                          edgecolor="#DEDBD4", linewidth=0.5))
        ax.text(0.0, 0.155, cs["caption"], transform=ax.transAxes,
                fontsize=5.4, color="#4A4A4A", ha="left", va="top",
                linespacing=1.7)

        if cs["key"] == "e":
            bar = ax.inset_axes([0.63, 0.17, 0.29, 0.25])
            bar.bar([0, 1], [0.75, 0.25], width=0.5, color=C_UNIQUE)
            for xx, vv in zip([0, 1], [0.75, 0.25]):
                bar.text(xx, vv + 0.04, f"{vv:.2f}", ha="center", va="bottom",
                         fontsize=5.2, color=C_INK)
            bar.set_xticks([0, 1])
            bar.set_xticklabels(["000", "001"], fontsize=5.0, **MONO)
            bar.set_ylim(0, 1.0)
            bar.set_yticks([0, 0.5, 1.0])
            bar.set_yticklabels(["0", "", "1"], fontsize=5.0)
            for s in ("top", "right"):
                bar.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                bar.spines[s].set_linewidth(0.5)
            bar.tick_params(length=1.8, width=0.5, pad=1.5)
            bar.set_ylabel("permitted\nweight", fontsize=5.0, labelpad=1,
                           linespacing=1.2)

    mp.save(stem)
    return mp


setup_matplotlib()
build()

try:                       # inline preview in Colab
    from IPython.display import Image, display
    display(Image("formnet_schematic.png", width=900))
except Exception:
    pass
