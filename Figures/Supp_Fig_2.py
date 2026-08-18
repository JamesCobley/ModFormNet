import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy.optimize import linprog

# ----------------------------------------------------------------------
NAME = "FormNet"
UNIT = "combination"
UNITS = UNIT + "s"

M_FULL = np.array([0.0, 0.25, 0.10, 0.05])     # one boundary zero, three interior
SITE_LABELS = ["K12", "K31", "K64", "K88"]
# ----------------------------------------------------------------------

PT_PER_INCH = 72.0

C_ZERO    = "#3B6E8F"   # boundary-fixed / forced weight
C_PARTIAL = "#C8562B"   # interior coordinate / permitted weight
C_UNIQUE  = "#4E6E58"   # conditional statements
C_INK     = "#1A1A1A"
C_RULE    = "#8C8C8C"
C_PALE    = "#D8D5CE"
C_BOX     = "#F4F2EE"
C_MATHBG  = "#EDF1F4"
C_MATHED  = "#C9D6DE"
MONO = {"family": "DejaVu Sans Mono"}


# ======================================================================
# layout
# ======================================================================

class MultiPanel:
    """540 pt = 7.5 in = 190 mm, a Nature double-column figure, so every
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


def _bare(ax, left=True, bottom=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(length=2.5, width=0.6, labelsize=5.6)
    for s in ("left", "bottom"):
        if ax.spines[s].get_visible():
            ax.spines[s].set_linewidth(0.6)


# ======================================================================
# the mathematics — LP and closed form, cross-checked
# ======================================================================

def polytope(m):
    P = len(m)
    S = np.array(list(itertools.product([0, 1], repeat=P)), float)
    A = np.vstack([np.ones(len(S)), S.T])
    b = np.concatenate([[1.0], np.asarray(m, float)])
    return S, A, b


def _lp(c, A, b, sense="min"):
    r = linprog(c if sense == "min" else -c, A_eq=A, b_eq=b,
                bounds=(0, 1), method="highs")
    return float(r.fun if sense == "min" else -r.fun)


def grade_envelope(m):
    """Exact [q_k^min, q_k^max] by LP over the feasible polytope."""
    S, A, b = polytope(m)
    k = S.sum(1)
    return np.array([[_lp((k == j).astype(float), A, b, "min"),
                      _lp((k == j).astype(float), A, b, "max")]
                     for j in range(len(m) + 1)])


def hierarchy(m_full):
    """All five levels, with the LP and the closed forms asserted equal."""
    m_full = np.asarray(m_full, float)
    interior = m_full[(m_full > 0) & (m_full < 1)]
    B1 = int((m_full == 1).sum())
    B0 = int((m_full == 0).sum())
    P = len(interior)
    S = float(interior.sum())
    mx = float(interior.max())

    env = grade_envelope(interior)

    # closed forms
    assert abs(env[0, 0] - max(0.0, 1 - S)) < 1e-9, "q0_min"
    assert abs(env[0, 1] - (1 - mx)) < 1e-9, "q0_max"
    assert abs(env[1, 0] - max(0.0, 2 * mx - S)) < 1e-9, "q1_min"

    # level 4 by LP on p_0, and by closed form
    Sm, A, b = polytope(interior)
    c0 = np.zeros(len(Sm)); c0[0] = 1.0
    rho_lo_lp = 1 - _lp(c0, A, b, "max")
    rho_hi_lp = 1 - _lp(c0, A, b, "min")
    assert abs(rho_lo_lp - mx) < 1e-9 and abs(rho_hi_lp - min(1.0, S)) < 1e-9

    return dict(m_full=m_full, interior=interior, B0=B0, B1=B1, P=P, S=S, m_max=mx,
                k_min=B1, k_max=B1 + P, envelope=env,
                rho=(mx, min(1.0, S)),
                cond=(max(1.0, S), S / mx),
                forced=[j for j in range(P + 1) if env[j, 0] > 1e-12])


def feasible_examples(m):
    """
    Three genuinely feasible weightings with the same marginals, so the same
    exact mean grade, but different grade distributions.
    """
    S, A, b = polytope(m)
    k = S.sum(1).astype(int)
    P = len(m)
    out = []

    # 1. all weight at grade 0 and 1: put m_j on each singleton
    p = np.zeros(len(S))
    for j in range(P):
        idx = np.where((S.sum(1) == 1) & (S[:, j] == 1))[0][0]
        p[idx] = m[j]
    p[0] = 1 - m.sum()
    out.append(("all singly modified", p))

    # 2. maximum entropy: the product distribution, exact on every marginal
    p = np.prod(np.where(S == 1, m, 1 - m), axis=1)
    out.append(("maximum entropy", p / p.sum()))

    # 3. push weight as high in grade as the marginals allow
    c = -(k.astype(float) ** 2)
    r = linprog(c, A_eq=A, b_eq=b, bounds=(0, 1), method="highs")
    out.append(("weight pushed high", r.x))

    res = []
    for lab, p in out:
        assert np.allclose(S.T @ p, m, atol=1e-9), lab
        assert abs(p.sum() - 1) < 1e-9 and (p > -1e-12).all(), lab
        q = np.bincount(k, weights=p, minlength=P + 1)
        assert abs(float((np.arange(P + 1) * q).sum()) - m.sum()) < 1e-9, lab
        res.append((lab, q))
    return res


# ======================================================================
# lattice drawing
# ======================================================================

def draw_subcube(ax, m_full, site_labels, node_w=0.80, node_h=0.38, fs=4.8):
    """
    Full R-coordinate lattice with the boundary-zero coordinates greying out
    half the space, and the admissible subcube drawn in bold.
    """
    R = len(m_full)
    combos = ["".join(map(str, c)) for c in itertools.product([0, 1], repeat=R)]
    pos = {}
    for kk in range(R + 1):
        rank = sorted([c for c in combos if c.count("1") == kk], reverse=True)
        for i, c in enumerate(rank):
            pos[c] = ((i - (len(rank) - 1) / 2) * 1.02, float(kk))

    zeros = [j for j, v in enumerate(m_full) if v == 0]
    dead = {c for c in combos if any(c[j] == "1" for j in zeros)}
    live = set(combos) - dead

    for a, b in itertools.combinations(combos, 2):
        if sum(u != v for u, v in zip(a, b)) != 1:
            continue
        both = a in live and b in live
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                color=C_ZERO if both else C_PALE,
                lw=1.5 if both else 0.4, zorder=2 if both else 1,
                solid_capstyle="round")

    for c, (x, y) in pos.items():
        alive = c in live
        face, edge, txt = ("white", C_PARTIAL, C_INK) if alive else ("white", C_PALE, C_PALE)
        ax.add_patch(FancyBboxPatch((x - node_w / 2, y - node_h / 2), node_w, node_h,
                                    boxstyle="round,pad=0.02,rounding_size=0.10",
                                    facecolor=face, edgecolor=edge,
                                    linewidth=0.7 if alive else 0.45, zorder=3))
        ax.text(x, y, c, ha="center", va="center", fontsize=fs, color=txt,
                zorder=4, **MONO)

    for kk in range(R + 1):
        n_live = len([c for c in live if c.count("1") == kk])
        if n_live:
            ax.text(4.1, kk, f"$k$={kk}   {n_live}", ha="right", va="center",
                    fontsize=4.9, color=C_ZERO)
    ax.set_xlim(-4.4, 4.4)
    ax.set_ylim(-0.6, R + 0.6)
    ax.axis("off")


# ======================================================================
# the figure
# ======================================================================

def build(stem="formnet_supp2", max_width=540):
    H = hierarchy(M_FULL)
    interior = H["interior"]
    P, S, mx = H["P"], H["S"], H["m_max"]
    env = H["envelope"]
    examples = feasible_examples(interior)
    ks = np.arange(P + 1)
    mp = MultiPanel(max_width=max_width)

    mtxt = ", ".join(f"{v:g}" for v in H["m_full"])
    mp.fig.text(0.012, 0.995,
                f"Worked example   $m$ = ({mtxt})   "
                f"$R$ = {len(H['m_full'])} · $B$ = {H['B0']} · $P$ = {P} · "
                f"$S$ = {S:g}",
                fontsize=7.6, fontweight="bold", color=C_INK, ha="left", va="top")

    # ================= a. level 1 — support geometry ==================
    mp.panel("a", width=210, height=150, pad_left=14, pad_top=24,
             margin_top=16, margin_right=18, margin_bottom=86)
    ax = plt.gca()
    draw_subcube(ax, H["m_full"], SITE_LABELS)
    ax.set_title("Level 1   support geometry", fontsize=7.2, fontweight="bold",
                 pad=4, loc="left")
    ax.text(0.0, -0.03,
            f"One boundary-zero coordinate removes half the lattice exactly.\n"
            f"The admissible subcube has $2^{{R-B}}$ = {2**P} {UNITS}, spanning\n"
            f"grades $k \\in [{H['k_min']}, {H['k_max']}]$. This is a possibility statement: it says\n"
            f"which grades occur in the lattice, not that any weighting uses them.",
            transform=ax.transAxes, fontsize=5.2, color="#4A4A4A", ha="left",
            va="top", linespacing=1.7)

    # ================= b. level 2 — the exact first moment ============
    mp.panel("b", width=240, height=150, pad_left=40, pad_top=24,
             margin_top=16, margin_right=0, margin_bottom=86)
    ax = plt.gca()
    w = 0.26
    cols = [C_ZERO, C_PARTIAL, C_UNIQUE]
    for i, (lab, q) in enumerate(examples):
        ax.bar(ks + (i - 1) * w, q, width=w, color=cols[i], edgecolor="none",
               label=lab, zorder=3)
    ax.set_xticks(ks)
    ax.set_xlabel("Grade $k$", fontsize=6.2)
    ax.set_ylabel("Weight $q_k$", fontsize=6.2)
    ax.set_ylim(0, 0.86)
    _bare(ax)
    ax.legend(fontsize=5.0, frameon=False, loc="upper right", handlelength=1.1,
              handletextpad=0.5, labelspacing=0.4, borderpad=0.2)
    ax.set_title("Level 2   exact first moment", fontsize=7.2, fontweight="bold",
                 pad=4, loc="left")

    strip = ax.inset_axes([0, -0.30, 1.0, 0.11])
    strip.axhline(0, color=C_RULE, lw=0.5)
    for i, (lab, q) in enumerate(examples):
        strip.plot([float((ks * q).sum())], [0], marker="v", ms=4,
                   color=cols[i], zorder=3)
    strip.set_xlim(-0.5, P + 0.5)
    strip.set_ylim(-0.5, 0.9)
    strip.set_yticks([])
    strip.set_xticks([S])
    strip.set_xticklabels([f"$E[K]$ = {S:g}"], fontsize=5.4)
    for sp in ("top", "right", "left", "bottom"):
        strip.spines[sp].set_visible(False)
    strip.tick_params(length=0, pad=1)

    ax.text(0.0, -0.46,
            f"Three weightings, all satisfying the same marginals, all with different\n"
            f"grade distributions. Every one has its centre of mass at exactly\n"
            f"$E[K] = \\sum_j m_j$ = {S:g}. Degenerate identity, exact mean grade.",
            transform=ax.transAxes, fontsize=5.2, color="#4A4A4A", ha="left",
            va="top", linespacing=1.7)

    # ================= c. level 3 — the grade envelope ================
    mp.panel("c", width=208, height=140, pad_left=38, pad_top=24,
             margin_right=18, margin_bottom=48)
    ax = plt.gca()
    for j in range(P + 1):
        lo, hi = env[j]
        forced = lo > 1e-12
        col = C_ZERO if forced else C_PARTIAL
        ax.bar(j, hi - lo, bottom=lo, width=0.52, color=col, alpha=0.30,
               edgecolor="none", zorder=2)
        if forced:
            ax.bar(j, lo, width=0.52, color=col, edgecolor="none", zorder=3)
        ax.plot([j - 0.30, j + 0.30], [hi, hi], color=col, lw=1.0, zorder=4)
        ax.text(j, hi + 0.022, f"{hi:.2f}", ha="center", va="bottom",
                fontsize=4.9, color=col)
        if forced:
            ax.text(j, lo - 0.022, f"{lo:.2f}", ha="center", va="top",
                    fontsize=4.9, color="white" if lo > 0.15 else col,
                    zorder=5)
    ax.set_xticks(ks)
    ax.set_xlabel("Grade $k$", fontsize=6.2)
    ax.set_ylabel("Weight $q_k$", fontsize=6.2)
    ax.set_ylim(0, 0.92)
    _bare(ax)
    ax.legend(handles=[Line2D([], [], color=C_ZERO, lw=4,
                              label="forced   every $p$ places $\\geq q_k^{min}$"),
                       Line2D([], [], color=C_PARTIAL, lw=4, alpha=0.30,
                              label="permitted   up to $q_k^{max}$")],
              fontsize=5.0, frameon=False, loc="upper right", handlelength=1.1,
              handletextpad=0.5, labelspacing=0.4, borderpad=0.2)
    ax.set_title("Level 3   grade envelope", fontsize=7.2, fontweight="bold",
                 pad=4, loc="left")
    ax.annotate(f"$q_1^{{min}} = 2\\max_j m_j - S$ = {env[1,0]:.2f}",
                xy=(1, env[1, 0]), xytext=(30, 26), textcoords="offset points",
                fontsize=5.2, color=C_ZERO,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C_ZERO,
                                shrinkA=0, shrinkB=2))
    ax.text(0.0, -0.245,
            f"Grade 1 is not merely possible. Every weighting consistent with\n"
            f"the marginals must place at least {env[1,0]:.2f} of the weight there.",
            transform=ax.transAxes, fontsize=5.2, color="#4A4A4A", ha="left",
            va="top", linespacing=1.7)

    # ================= d. levels 4-5 — the modified subpopulation =====
    mp.panel("d", width=232, height=140, pad_left=40, pad_top=24,
             margin_right=0, margin_bottom=48)
    ax = plt.gca()
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Levels 4-5   the modified subpopulation", fontsize=7.2,
                 fontweight="bold", pad=4, loc="left")

    ax.add_patch(FancyBboxPatch((0.01, 0.80), 0.98, 0.19,
                                boxstyle="round,pad=0.012,rounding_size=0.04",
                                transform=ax.transAxes, facecolor=C_MATHBG,
                                edgecolor=C_MATHED, linewidth=0.6, zorder=0))
    ax.text(0.5, 0.945, r"$S \;=\; \rho \cdot E[K \mid K \geq 1]$,"
                        r"$\qquad \rho = P(K \geq 1) = 1 - q_0$",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.2,
            color=C_INK)
    ax.text(0.5, 0.855, "copies at $k$ = 0 contribute nothing to the mean grade",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.0,
            color=C_RULE, style="italic")

    rlo, rhi = H["rho"]
    r_ax = ax.inset_axes([0.04, 0.56, 0.92, 0.10])
    r_ax.hlines(0, rlo, rhi, color=C_ZERO, lw=5, capstyle="butt", zorder=3)
    for v, lab, ha_ in [(rlo, "$\\max_j m_j$", "right"), (rhi, "$\\min(1,S)$", "left")]:
        r_ax.plot([v], [0], marker="|", ms=9, mew=1.2, color=C_ZERO, zorder=4)
        r_ax.text(v, 0.55, f"{v:g}\n{lab}", ha=ha_, va="bottom", fontsize=5.0,
                  color=C_INK, linespacing=1.35)
    r_ax.set_xlim(0, 1)
    r_ax.set_ylim(-1.5, 2.2)
    r_ax.set_yticks([])
    r_ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    r_ax.tick_params(labelsize=5.0, length=2, width=0.5, pad=1)
    for sp in ("top", "right", "left"):
        r_ax.spines[sp].set_visible(False)
    r_ax.spines["bottom"].set_linewidth(0.5)
    r_ax.set_xlabel("Level 4    $\\rho$, the modified fraction", fontsize=5.8,
                    labelpad=1)

    clo, chi = H["cond"]
    c_ax = ax.inset_axes([0.04, 0.22, 0.92, 0.10])
    c_ax.axvspan(1, P, color="#F2F1EE", zorder=0)
    c_ax.hlines(0, clo, chi, color=C_UNIQUE, lw=5, capstyle="butt", zorder=3)
    for v, lab, ha_ in [(clo, "$\\max(1,S)$", "right"), (chi, "$S/\\max_j m_j$", "left")]:
        c_ax.plot([v], [0], marker="|", ms=9, mew=1.2, color=C_UNIQUE, zorder=4)
        c_ax.text(v, 0.55, f"{v:g}\n{lab}", ha=ha_, va="bottom", fontsize=5.0,
                  color=C_INK, linespacing=1.35)
    c_ax.set_xlim(0.6, P + 0.4)
    c_ax.set_ylim(-1.5, 2.2)
    c_ax.set_yticks([])
    c_ax.set_xticks(range(1, P + 1))
    c_ax.tick_params(labelsize=5.0, length=2, width=0.5, pad=1)
    for sp in ("top", "right", "left"):
        c_ax.spines[sp].set_visible(False)
    c_ax.spines["bottom"].set_linewidth(0.5)
    c_ax.set_xlabel("Level 5    $E[K \\mid K \\geq 1]$, of a possible $[1, P]$",
                    fontsize=5.8, labelpad=1)

    ax.text(0.0, -0.03,
            f"Among copies carrying at least one modification the mean grade is\n"
            f"bounded to [{clo:g}, {chi:g}] of a possible [1, {P}]. Levels 4-5 have content only\n"
            f"when $B_1$ = 0; if any coordinate is fixed modified then $\\rho$ = 1 and\n"
            f"$E[K \\mid K \\geq 1]$ collapses onto the exact first moment of level 2.",
            transform=ax.transAxes, fontsize=5.2, color="#4A4A4A", ha="left",
            va="top", linespacing=1.7)

    mp.save(stem)
    return H


setup_matplotlib()
H = build()

print("\nAll bounds cross-checked, LP against closed form:")
print(f"  q_0 in [{H['envelope'][0,0]:.4f}, {H['envelope'][0,1]:.4f}]"
      f"   closed [max(0,1-S), 1-max m]")
print(f"  q_1 in [{H['envelope'][1,0]:.4f}, {H['envelope'][1,1]:.4f}]"
      f"   closed q_1^min = max(0, 2 max m - S)")
print(f"  rho in [{H['rho'][0]:.4f}, {H['rho'][1]:.4f}]")
print(f"  E[K|K>=1] in [{H['cond'][0]:.4f}, {H['cond'][1]:.4f}]")
print(f"  forced grades: {H['forced']}")

try:
    from IPython.display import Image, display
    display(Image("formnet_supp2.png", width=900))
except Exception:
    pass
