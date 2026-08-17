import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon

# ----------------------------------------------------------------------
NAME = "FormNet"
UNIT = "combination"
UNITS = UNIT + "s"
# ----------------------------------------------------------------------

PT_PER_INCH = 72.0

C_AC      = "#C8562B"   # PTM type T, the projected type (acetylation)
C_PH      = "#7A5C86"   # a second PTM type
C_OX      = "#3E7D80"   # a third PTM type
C_ZERO    = "#3B6E8F"   # boundary-fixed coordinate
C_PARTIAL = "#C8562B"   # quantitatively constrained coordinate
C_UNIQUE  = "#4E6E58"
C_INK     = "#1A1A1A"
C_RULE    = "#8C8C8C"
C_PALE    = "#D8D5CE"
C_BOX     = "#F4F2EE"
C_MATHBG  = "#EDF1F4"
C_MATHED  = "#C9D6DE"

MONO = {"family": "DejaVu Sans Mono"}

# the worked example: N modifiable sites across three PTM types,
# R of them of the projected type T
N_TOTAL = 12
SITES = [   # (sequence position, residue, PTM type, colour)
    (18,  "S", "phospho",     C_PH),
    (32,  "K", "acetylation", C_AC),
    (47,  "C", "oxidation",   C_OX),
    (61,  "T", "phospho",     C_PH),
    (75,  "K", "acetylation", C_AC),
    (88,  "C", "oxidation",   C_OX),
    (103, "Y", "phospho",     C_PH),
    (119, "C", "oxidation",   C_OX),
    (134, "K", "acetylation", C_AC),
    (150, "S", "phospho",     C_PH),
    (168, "C", "oxidation",   C_OX),
    (182, "T", "phospho",     C_PH),
]
SEQ_LEN = 200
T_SITES = [s for s in SITES if s[2] == "acetylation"]
R = len(T_SITES)


# ======================================================================
# minimal pixel-exact multipanel layout
# ======================================================================

class MultiPanel:
    """Panels flow left to right in POINTS and wrap at max_width.
    540 pt = 7.5 in = 190 mm, a Nature double-column figure, so every
    fontsize below is a literal point size."""

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
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


# ======================================================================
# lattice helpers (identical geometry to the main schematic)
# ======================================================================

def combos(R=3):
    return ["".join(map(str, c)) for c in itertools.product([0, 1], repeat=R)]


def lattice_positions(R=3, xgap=1.30):
    pos = {}
    for k in range(R + 1):
        rank = sorted((c for c in combos(R) if c.count("1") == k), reverse=True)
        for i, c in enumerate(rank):
            pos[c] = ((i - (len(rank) - 1) / 2) * xgap, float(k))
    return pos


def hamming_edges(R=3):
    return [(a, b) for a, b in itertools.combinations(combos(R), 2)
            if sum(u != v for u, v in zip(a, b)) == 1]


def excluded_by(m, R=3):
    out = set()
    for c in combos(R):
        for j, v in enumerate(m):
            if v is None:
                continue
            if (v == 0 and c[j] == "1") or (v == 1 and c[j] == "0"):
                out.add(c)
    return out


def draw_lattice(ax, m=None, R=3, accent=C_PARTIAL, node_w=0.86, node_h=0.40,
                 fontsize=5.2, xgap=1.30, emphasise=(), hull=None):
    pos = lattice_positions(R, xgap)
    ex = excluded_by(m, R) if m is not None else set()


    hull = set(hull or ())
    for a, b in hamming_edges(R):
        dead = a in ex or b in ex
        inside = a in hull and b in hull
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=C_ZERO if inside else (C_PALE if dead else "#BFBCB5"),
                lw=1.8 if inside else 0.45, zorder=2 if inside else 1,
                solid_capstyle="round")

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

    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim(-0.55, R + 0.55)
    ax.axis("off")


def bitbox(ax, x, y, w, h, char, color, fontsize=6.0, filled=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
        transform=ax.transAxes, facecolor=color if filled else "white",
        edgecolor=color, linewidth=0.7, zorder=3))
    ax.text(x + w / 2, y + h / 2, char, transform=ax.transAxes, ha="center",
            va="center", fontsize=fontsize, zorder=4,
            color="white" if filled else C_INK, **MONO)


# ======================================================================
# figure
# ======================================================================

def build(stem="formnet_supp1", max_width=540):
    mp = MultiPanel(max_width=max_width)

    # ================== a. proteoform -> projection ===================
    mp.panel("a", width=498, height=118, pad_left=24, pad_top=20,
             margin_right=0, margin_bottom=24)
    ax = plt.gca()
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Each {UNIT} is a projection of a fully specified proteoform",
                 fontsize=7.2, fontweight="bold", pad=6, loc="left")

    # --- the sequence, carrying every modifiable site of every type
    x0, x1, ybar = 0.035, 0.60, 0.66
    ax.add_patch(FancyBboxPatch(
        (x0, ybar), x1 - x0, 0.055,
        boxstyle="round,pad=0.002,rounding_size=0.02", transform=ax.transAxes,
        facecolor="#F0EEE9", edgecolor="#C9C6BF", linewidth=0.6, zorder=1))

    def sx(p):
        return x0 + (x1 - x0) * p / SEQ_LEN

    for pos_, res, ptm, col in SITES:
        isT = (ptm == "acetylation")
        ax.plot([sx(pos_), sx(pos_)], [ybar - 0.055 if isT else ybar - 0.03,
                                       ybar + 0.055],
                color=col, lw=1.5 if isT else 0.9, zorder=3,
                solid_capstyle="butt")
        if isT:
            ax.text(sx(pos_), ybar + 0.075, f"{res}{pos_}", ha="center",
                    va="bottom", fontsize=5.0, color=col, **MONO)

    ax.text(x0, ybar - 0.115,
            f"$N$ = {N_TOTAL} modifiable sites across three PTM types",
            fontsize=5.6, color=C_INK, ha="left", va="top")
    for i, (lab, col) in enumerate([("acetylation (type $T$)", C_AC),
                                    ("phosphorylation", C_PH),
                                    ("oxidation", C_OX)]):
        cx = x0 + 0.205 * i
        ax.plot([cx, cx + 0.016], [ybar - 0.215, ybar - 0.215], color=col,
                lw=1.5, solid_capstyle="butt")
        ax.text(cx + 0.026, ybar - 0.215, lab, fontsize=5.2, color=C_RULE,
                ha="left", va="center")

    # --- the projection arrow
    ax.annotate("", xy=(0.705, ybar + 0.02), xytext=(0.625, ybar + 0.02),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=C_INK,
                                mutation_scale=7))
    ax.text(0.665, ybar + 0.085, r"$\pi_T$", ha="center", va="bottom",
            fontsize=7.0, color=C_INK)
    ax.text(0.665, ybar - 0.045, "project onto\ntype $T$", ha="center",
            va="top", fontsize=5.0, color=C_RULE, linespacing=1.4)

    # --- the resulting binary string, with labels retained per coordinate
    bw, bh, bx = 0.048, 0.085, 0.735
    for j, (pos_, res, ptm, col) in enumerate(T_SITES):
        bitbox(ax, bx + j * (bw + 0.012), ybar - 0.015, bw, bh,
               ["1", "0", "1"][j], C_AC, fontsize=6.4,
               filled=(["1", "0", "1"][j] == "1"))
        ax.text(bx + j * (bw + 0.012) + bw / 2, ybar - 0.045,
                f"$\\mathrm{{{res}}}^{{{pos_},\\,\\mathrm{{ac}}}}$",
                ha="center", va="top", fontsize=5.0, color=C_RULE)

    ax.text(bx + 1.5 * (bw + 0.012) - 0.006, ybar + 0.115,
            r"$x \in \Omega_R = \{0,1\}^R$,   $R$ = " + str(R),
            ha="center", va="bottom", fontsize=5.8, color=C_INK)

    ax.text(0.035, 0.305,
            f"Residue identity, sequence position and modification type are not discarded by the projection: they remain attached to each\n"
            f"coordinate as labels. The binary string is the {UNIT}; the labels say which {UNIT} it is. Two coordinates may be exchanged only\n"
            f"if their labels are exchanged with them.",
            fontsize=5.4, color="#4A4A4A", ha="left", va="top", linespacing=1.7)

    # ================== b. the fibre ==================================
    mp.panel("b", width=498, height=126, pad_left=24, pad_top=20,
             margin_right=0, margin_bottom=22)
    ax = plt.gca()
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"A bound on a {UNIT} is a bound on its entire fibre",
                 fontsize=7.2, fontweight="bold", pad=6, loc="left")

    # full proteoforms above
    n_show, fx0, fw = 5, 0.055, 0.030
    rng = np.random.default_rng(4)
    for i in range(n_show):
        fx = fx0 + i * (fw + 0.020)
        for j in range(N_TOTAL):
            isT = SITES[j][2] == "acetylation"
            if isT:                      # the projected coordinates agree: 1,0,1
                on = [1, 0, 1][[k for k, sc in enumerate(SITES)
                                if sc[2] == "acetylation"].index(j)]
                cc = C_AC if on else "#EBD8CF"
                lw = 1.1
            else:                        # everything else is free to vary
                cc = C_RULE if rng.random() < 0.4 else "#DCDAD4"
                lw = 0.6
            ax.plot([fx, fx + fw], [0.88 - j * 0.030] * 2, color=cc, lw=lw,
                    solid_capstyle="butt")
    ax.text(fx0 + n_show * (fw + 0.020) + 0.008, 0.70, "...", ha="left",
            va="center", fontsize=8, color=C_RULE)
    ax.text(fx0, 0.955, r"$z \in \{0,1\}^N$   full proteoforms",
            fontsize=5.6, color=C_INK, ha="left", va="bottom")

    # bracket down to the single combination
    ax.annotate("", xy=(0.335, 0.42), xytext=(0.215, 0.42),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color=C_INK,
                                mutation_scale=7))
    ax.text(0.275, 0.465, r"$\pi_T$", ha="center", va="bottom", fontsize=7.0,
            color=C_INK)

    for j, ch in enumerate("101"):
        bitbox(ax, 0.355 + j * 0.055, 0.36, 0.046, 0.14, ch, C_AC,
               fontsize=6.4, filled=(ch == "1"))
    ax.text(0.437, 0.545, r"$x$", ha="center", va="bottom", fontsize=6.4,
            color=C_INK)

    ax.text(0.545, 0.80,
            r"$\pi_T^{-1}(x) = \{\, z : \pi_T(z) = x \,\}$",
            fontsize=6.2, color=C_INK, ha="left", va="top")
    ax.text(0.545, 0.585,
            r"$|\pi_T^{-1}(x)| = 2^{\,N-R} = 2^{\,%d} = %d$" % (N_TOTAL - R,
                                                                2 ** (N_TOTAL - R)),
            fontsize=6.2, color=C_INK, ha="left", va="top")
    ax.text(0.545, 0.375,
            r"$p_x \;=\; \sum_{z \,:\, \pi_T(z) = x} q_z \;\leq\; u$",
            fontsize=6.6, color=C_INK, ha="left", va="top")
    ax.text(0.545, 0.155,
            f"The weight carried by one {UNIT} is the summed weight of every\n"
            f"full proteoform projecting onto it. A {NAME} ceiling therefore\n"
            f"caps the whole fibre at once, without resolving anything inside it.",
            fontsize=5.4, color="#4A4A4A", ha="left", va="top", linespacing=1.7)

    # ================== c. A(m), the support object ===================
    mp.panel("c", width=222, height=186, pad_left=24, pad_top=20,
             margin_right=28, margin_bottom=10)
    ax = plt.gca()
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(r"$\mathcal{A}(m)$   which " + UNITS,
                 fontsize=7.2, fontweight="bold", pad=6, loc="left")

    m_demo = [0, 0.25, 0.25]
    lat = ax.inset_axes([0.02, 0.30, 0.96, 0.70])
    draw_lattice(lat, m=m_demo, R=3, accent=C_PARTIAL, xgap=1.22,
                 hull=["000", "010", "011", "001"])
    lat.text(2.05, 1.5, "admissible\nsubcube\n$2^{R-B}$ vertices", fontsize=5.0,
             color=C_ZERO, ha="right", va="center", linespacing=1.35)

    ax.text(0.02, 0.245,
            r"$m = (0,\ 0.25,\ 0.25)$,   $B$ = 1 boundary coordinate",
            fontsize=5.6, color=C_INK, ha="left", va="top")
    ax.text(0.02, 0.165,
            r"$|\mathcal{A}(m)| = 2^{\,R-B} = 2^{\,2} = 4$   exactly",
            fontsize=5.8, color=C_ZERO, ha="left", va="top")
    ax.text(0.02, 0.085,
            f"An $(R-B)$-dimensional subcube. Every one of its\n"
            f"vertices is admissible, so the count is exact, not\n"
            f"an upper limit. $B$ bits of freedom are removed.",
            fontsize=5.3, color="#4A4A4A", ha="left", va="top", linespacing=1.7)

    # ================== d. P(m), the weight object ====================
    mp.panel("d", width=222, height=186, pad_left=42, pad_top=20,
             margin_right=0, margin_bottom=10)
    ax = plt.gca()
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(r"$\mathcal{P}(m)$   how much weight",
                 fontsize=7.2, fontweight="bold", pad=6, loc="left")

    pl = ax.inset_axes([0.0, 0.42, 0.98, 0.56])
    t = np.linspace(0, 0.25, 200)
    for lbl, vals, col, ls in [
        (r"$p_{000}$", 0.50 + t, C_ZERO, "-"),
        (r"$p_{001} = p_{010}$", 0.25 - t, C_PARTIAL, "-"),
        (r"$p_{011}$", t, C_UNIQUE, "-"),
    ]:
        pl.plot(t, vals, color=col, lw=1.1, ls=ls, label=lbl)
    pl.set_xlim(-0.008, 0.258)
    pl.set_ylim(-0.03, 0.80)
    pl.set_xticks([0, 0.125, 0.25])
    pl.set_xticklabels(["0", "", "0.25"], fontsize=5.2)
    pl.set_yticks([0, 0.25, 0.5, 0.75])
    pl.set_yticklabels(["0", "0.25", "0.50", "0.75"], fontsize=5.2)
    pl.set_xlabel("free parameter $t$", fontsize=5.6, labelpad=1)
    pl.set_ylabel("permitted weight", fontsize=5.6, labelpad=2)
    for s in ("top", "right"):
        pl.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        pl.spines[s].set_linewidth(0.6)
    pl.tick_params(length=2, width=0.6, pad=1.5)
    pl.legend(fontsize=4.9, frameon=False, loc="upper right",
              handlelength=1.2, handletextpad=0.4, labelspacing=0.35,
              borderpad=0.1)

    ax.text(0.0, 0.245,
            r"same $m$; $\dim \mathcal{P}(m) = 4 - 3 = 1$",
            fontsize=5.8, color=C_INK, ha="left", va="top")
    ax.text(0.0, 0.165,
            f"The four admissible {UNITS} are fixed, but their\n"
            f"weights are not. One free parameter survives, so\n"
            f"$\\mathcal{{P}}(m)$ is a bounded segment, not a point. Every\n"
            f"{UNIT} is bounded; none is determined.",
            fontsize=5.3, color="#4A4A4A", ha="left", va="top", linespacing=1.7)

    mp.save(stem)
    return mp


setup_matplotlib()
build()

try:
    from IPython.display import Image, display
    display(Image("formnet_supp1.png", width=900))
except Exception:
    pass
