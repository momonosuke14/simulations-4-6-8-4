"""Gera as figuras e a tabela de resultados (RESULTADOS.md) a partir dos pickles de run_ens.py.
Estilo: violeta -> vermelho, superficies 3D como no artigo, mais painéis 2D de comparacao entre volumes."""
import os, glob, pickle, json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa

OUT = "figs7"; os.makedirs(OUT, exist_ok=True)
CM = LinearSegmentedColormap.from_list("vr", ["#fbf3ee", "#e8a99a", "#d6283c", "#8e2a8a", "#3b1a78"])
COL = {4: "#5b3cc4", 6: "#b02a8f", 8: "#e0352b"}
GRID = np.round(np.linspace(0, 8, 81), 10)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130,
                     "axes.spines.top": False, "axes.spines.right": False})


def load(L):
    d = {}
    for f in glob.glob(f"res_L{L}/beta_*.pkl"):
        b = float(os.path.basename(f)[5:-4]); d[b] = pickle.load(open(f, "rb"))
    return dict(sorted(d.items()))


DATA = {L: load(L) for L in (4, 6, 8)}
DATA = {L: d for L, d in DATA.items() if d}


def ms(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan) if len(x) else (np.nan, np.nan)


def series(L, fn):
    bs = list(DATA[L].keys()); m, e = [], []
    for b in bs:
        a, s = ms([fn(r) for r in DATA[L][b]]); m.append(a); e.append(s)
    return np.array(bs), np.array(m), np.array(e)


def cold_N(L):
    p = 2 * np.pi * np.arange(L) / L
    g = np.meshgrid(*([1 - np.cos(p)] * 4), indexing='ij'); lam = np.sort(np.repeat(sum(g).ravel(), 8))
    return np.array([np.sum(lam <= x + 1e-12) for x in GRID]) / L ** 4


def dimp(r, Lm):  # self-consistent normalisation (see README)
    return r[f'c1_{Lm}']['tr2']


def trXn(r, Lm):
    return r[f'c1_{Lm}']['trX'] / dimp(r, Lm)


def f1(r, Lm): return r[f'c1_{Lm}']['f'][0]
def lam1ray(r, Lm): return r['lamM'][0] * r[f'c1_{Lm}']['ray']


# ------------------------------------------------------------------ fig 1: lambda_1 vs beta, volumes
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
for L in DATA:
    b, m, e = series(L, lambda r: r['lamM'][0]); ax[0].errorbar(b, m, e, marker='o', color=COL[L], label=f"{L}$^4$", capsize=2)
    b, m, e = series(L, lambda r: r['lam1D']); ax[1].errorbar(b, m, e, marker='s', color=COL[L], label=f"{L}$^4$", capsize=2)
ax[0].set_yscale('log'); ax[0].set_xscale('log'); ax[1].set_xscale('log'); ax[1].set_yscale('log')
ax[0].set_xlabel(r"$\beta$"); ax[0].set_ylabel(r"$\langle\lambda_1(M_A)\rangle$ (best gauge copy)"); ax[0].legend(title="lattice")
ax[1].set_xlabel(r"$\beta$"); ax[1].set_ylabel(r"$\langle\lambda_1(\Delta_A)\rangle$")
ax[0].set_title("distance to the Gribov horizon"); ax[1].set_title("covariant Laplacian (gauge invariant)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig01_lambda1_volumes.png"); plt.close(fig)

# ------------------------------------------------------------------ fig 2: Gribov copies
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
for L in DATA:
    b, m, e = series(L, lambda r: float(r['ndistinct'] >= 2)); ax[0].errorbar(b, m, e, marker='o', color=COL[L], label=f"{L}$^4$", capsize=2)
    b, m, e = series(L, lambda r: r['ndistinct']); ax[1].errorbar(b, m, e, marker='o', color=COL[L], label=f"{L}$^4$", capsize=2)
for a in ax: a.set_xscale('log'); a.set_xlabel(r"$\beta$")
ax[0].set_ylabel("fraction of configurations with $\\geq 2$ minima"); ax[1].set_ylabel("distinct minima per configuration")
ax[0].legend(title="lattice"); ax[0].set_title("Gribov copies among the gauge starts"); ax[1].set_title("(lower bound: finite number of starts)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig02_gribov_copies.png"); plt.close(fig)

# ------------------------------------------------------------------ 3D helpers
def surf(ax, X, Y, Z, **kw):
    ax.plot_surface(X, Y, Z, cmap=CM, edgecolor='none', alpha=0.95, **kw)


def tick_beta(ax, bs):
    ax.set_xticks(np.arange(len(bs))); ax.set_xticklabels([f"{b:g}" for b in bs])


# ------------------------------------------------------------------ fig 3: distance to horizon per configuration (3D) -- largest L with data
Lmax = 6 if 6 in DATA else max(DATA)
bs = list(DATA[Lmax].keys())
fig = plt.figure(figsize=(6.4, 5)); ax = fig.add_subplot(111, projection='3d')
for i, b in enumerate(bs):
    rr = DATA[Lmax][b]; lam = np.array([r['lamM'][0] for r in rr]); ch = np.array([r['chain'] for r in rr])
    ax.scatter(np.full(len(lam), i), np.arange(len(lam)), lam, c=lam, cmap=CM, vmin=0, vmax=max(0.2, lam.max()), s=26, depthshade=False, edgecolor='k', linewidth=0.3)
    ax.plot([i, i], [-0.5, len(lam) - 0.5], [lam.mean()] * 2, color="#3b1a78", lw=2.2)
tick_beta(ax, bs); ax.set_xlabel(r"$\beta$"); ax.set_ylabel("configuration"); ax.set_zlabel(r"$\lambda_1$ (best gauge copy)")
ax.set_title(f"Distance to the horizon per configuration, {Lmax}$^4$ (bar = mean)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig03_horizon_distance_3d.png"); plt.close(fig)

# ------------------------------------------------------------------ fig 4: Gribov copies (3D scatter)
fig = plt.figure(figsize=(6.6, 5)); ax = fig.add_subplot(111, projection='3d')
for i, b in enumerate(bs):
    for r in DATA[Lmax][b]:
        dF = r['copies_F']; l1 = r['copies_l1']
        ax.scatter(np.full(len(dF), i), dF, l1, c=np.where(dF < 1e-6, "#4b2aa8", "#d6283c"), s=14, depthshade=False)
tick_beta(ax, bs); ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$F-F_{\min}$ (same configuration)"); ax.set_zlabel(r"$\lambda_1$ of the copy")
ax.set_title(f"Gribov copies on {Lmax}$^4$ (violet: lowest F found, red: other minima)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig04_gribov_copies_3d.png"); plt.close(fig)

# ------------------------------------------------------------------ fig 5: counting function surface (largest L)
bsN = list(DATA[Lmax].keys())
Z = np.array([np.mean([r['N'] for r in DATA[Lmax][b]], axis=0) / Lmax ** 4 for b in bsN])
X, Y = np.meshgrid(GRID, np.arange(len(bsN)))
fig = plt.figure(figsize=(7, 5)); ax = fig.add_subplot(111, projection='3d'); surf(ax, X, Y, Z)
cn = cold_N(Lmax); ax.plot(GRID, np.full_like(GRID, -0.6), cn, color='k', lw=1.2, label="cold lattice (exact)")
ax.set_yticks(np.arange(len(bsN))); ax.set_yticklabels([f"{b:g}" for b in bsN]); ax.set_xlabel(r"cutoff $\Lambda$"); ax.set_ylabel(r"$\beta$"); ax.set_zlabel(r"$N(\Lambda)/V$")
ax.set_title(rf"Counting function of $\widehat H_\tau$ ($\tau=0.5$, $T=0.2$), {Lmax}$^4$ (matrix-free)", fontsize=9); ax.legend(loc='upper left')
ax.view_init(elev=24, azim=-125); fig.tight_layout(); fig.savefig(f"{OUT}/fig05_counting_3d.png"); plt.close(fig)

# ------------------------------------------------------------------ fig 6: extensivity across volumes
fig, axs = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
for a, beta in zip(axs, (3.0, 6.0, 12.0)):
    for L in DATA:
        if beta in DATA[L]:
            Nm = np.mean([r['N'] for r in DATA[L][beta]], axis=0) / L ** 4
            a.plot(GRID, Nm, color=COL[L], label=f"{L}$^4$")
    a.set_title(rf"$\beta={beta:g}$"); a.set_xlabel(r"$\Lambda$")
axs[0].set_ylabel(r"$N(\Lambda)/V$"); axs[0].legend()
fig.suptitle(r"Volume dependence of the counting function (necessary condition of Conjecture 2)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig06_extensivity_volumes.png"); plt.close(fig)

# ------------------------------------------------------------------ fig 7: Conjecture 1 -- normalised trace vs volume
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6))
for k, Lm in enumerate((2.0, 4.0)):
    for L in DATA:
        b, m, e = series(L, lambda r: trXn(r, Lm)); axs[k].errorbar(b, m, e, marker='o', color=COL[L], label=f"{L}$^4$", capsize=2)
    if 4 in DATA:
        b, m, e = series(4, lambda r: r['exact'][str(Lm)]['trX'] / r['exact'][str(Lm)]['dimp'])
        axs[k].plot(b, m, 'k--', lw=1, label="4$^4$ exact (dense)")
    axs[k].set_xscale('log'); axs[k].set_xlabel(r"$\beta$"); axs[k].set_ylabel(r"$\mathrm{Tr}\,X/\dim p$"); axs[k].set_title(rf"$\Lambda={Lm:g}$")
axs[0].legend(fontsize=7)
for L in DATA:
    b, m, e = series(L, lambda r: f1(r, 4.0)); axs[2].errorbar(b, m, e, marker='o', color=COL[L], label=f"{L}$^4$", capsize=2)
axs[2].set_xscale('log'); axs[2].set_xlabel(r"$\beta$"); axs[2].set_ylabel(r"$f_1=\|p_\Lambda v_1\|^2$"); axs[2].set_title(r"near-horizon weight, $\Lambda=4$")
fig.suptitle("Conjecture 1 test: normalised compressed inverse Tr X/dim p against beta for 4^4, 6^4, 8^4", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig07_conjecture1_volume.png"); plt.close(fig)

# ------------------------------------------------------------------ fig 8: compressed inverse (3D bars) largest L
fig = plt.figure(figsize=(10, 4.4))
LAMS = (1.0, 2.0, 4.0)
for k, (name, fn) in enumerate((("$\\lambda_1\\,\\langle \\tilde p v_1, M^{+}\\tilde p v_1\\rangle$ (lower bound of $\\lambda_1\\|X\\|$)", lam1ray),
                                ("$f_1=\\|\\tilde p v_1\\|^2$", f1))):
    ax = fig.add_subplot(1, 2, k + 1, projection='3d')
    for j, Lm in enumerate(LAMS):
        for i, b in enumerate(bsN):
            v = np.mean([fn(r, Lm) for r in DATA[Lmax][b]])
            ax.bar3d(i - 0.3, j - 0.3, 0, 0.6, 0.6, v, color=CM(1 - min(v, 1)), shade=True)
    tick_beta(ax, bsN); ax.set_yticks(range(3)); ax.set_yticklabels([f"{x:g}" for x in LAMS]); ax.set_xlabel(r"$\beta$"); ax.set_ylabel(r"$\Lambda$"); ax.set_zlim(0, 1.05)
    ax.set_title(name, fontsize=8)
fig.suptitle(f"Test of Proposition 10 on {Lmax}$^4$: the two coincide, the compressed inverse is dominated by the near-horizon mode", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig08_compressed_inverse_3d.png"); plt.close(fig)

# ------------------------------------------------------------------ fig 9: weights of the lowest modes
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
cols = [CM(0.25 + 0.7 * i / max(1, len(bsN) - 1)) for i in range(len(bsN))]
for i, b in enumerate(bsN):
    fk = np.mean([r['c1_2.0']['f'] for r in DATA[Lmax][b]], axis=0); lk = np.mean([r['lamM'] for r in DATA[Lmax][b]], axis=0)
    axs[0].plot(np.arange(1, len(fk) + 1), fk, 'o-', color=cols[i], label=rf"$\beta={b:g}$")
    axs[1].plot(lk, fk, 'o-', color=cols[i])
axs[0].set_xlabel("mode index $k$ (ordered by $\\lambda_k$)"); axs[0].set_ylabel(r"$f_k=\|p_\Lambda v_k\|^2$, $\Lambda=2$"); axs[0].legend(fontsize=7)
axs[1].set_xlabel(r"$\lambda_k$"); axs[1].set_ylabel(r"$f_k$"); axs[1].set_xscale('log')
fig.suptitle(f"How much of each near-horizon mode lies inside the ultraviolet range ({Lmax}$^4$)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig09_near_horizon_weights.png"); plt.close(fig)

# ------------------------------------------------------------------ fig 10-12: algebra of regimes (L = 4, dense)
if 4 in DATA and 'regimes' in DATA[4][list(DATA[4])[0]][0]:
    pairs = ["0.1_4.0", "0.3_4.0", "0.3_6.0", "1.0_4.0"]; pcol = ["#d6283c", "#8e2a8a", "#3b1a78", "#e88a2a"]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5))
    for p, c in zip(pairs, pcol):
        lab = r"$(\lambda_c,\Lambda)=(%s,%s)$" % tuple(p.split("_"))
        b, m, e = series(4, lambda r: r['regimes'][p]['comm']); axs[0].errorbar(b, m, e, marker='o', color=c, label=lab, capsize=2)
        b, m, e = series(4, lambda r: r['regimes'][p]['leak']); axs[1].errorbar(b, m, e, marker='o', color=c, capsize=2)
        b, m, e = series(4, lambda r: r['regimes'][p]['ngen']); axs[2].errorbar(b, m, e, marker='o', color=c, capsize=2)
    for a in axs: a.set_xscale('log'); a.set_xlabel(r"$\beta$")
    axs[0].axhline(0.5, color='gray', ls=':'); axs[0].set_ylabel(r"$\|[P_c,p_\Lambda]\|$"); axs[0].legend(fontsize=7)
    axs[1].set_ylabel(r"leakage $\mathrm{Tr}(P_c^\perp p_\Lambda)/\dim P_c^\perp$"); axs[2].set_ylabel("number of generic angles"); axs[2].set_yscale('log')
    fig.suptitle("Algebra of regimes on 4$^4$ (dense, independent reimplementation)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig10_regimes_L4.png"); plt.close(fig)
    # generic angles histogram surface
    bins = np.linspace(0, 1, 41); b4 = list(DATA[4].keys()); H = []
    for b in b4:
        h = np.zeros(len(bins) - 1)
        for r in DATA[4][b]:
            h += np.histogram(r['regimes']['0.3_4.0']['cos2'], bins=bins)[0]
        H.append(h / len(DATA[4][b]))
    X, Y = np.meshgrid(0.5 * (bins[1:] + bins[:-1]), np.arange(len(b4)))
    fig = plt.figure(figsize=(6.6, 5)); ax = fig.add_subplot(111, projection='3d'); surf(ax, X, Y, np.array(H))
    ax.set_yticks(np.arange(len(b4))); ax.set_yticklabels([f"{b:g}" for b in b4]); ax.set_xlabel(r"$\cos^2\theta_k$"); ax.set_ylabel(r"$\beta$"); ax.set_zlabel("number of angles")
    ax.set_title(r"Generic part of the regime algebra, 4$^4$, $(\lambda_c,\Lambda)=(0.3,4)$", fontsize=9); ax.view_init(elev=26, azim=-120)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig11_generic_angles_3d.png"); plt.close(fig)

# ------------------------------------------------------------------ fig 12: validation of the matrix-free estimators (L = 4)
if 4 in DATA:
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5))
    err = []; xs = []; ys = []; xe = []; ye = []
    for b in DATA[4]:
        for r in DATA[4][b]:
            err.append(np.abs(r['N'] - r['N_exact']).max() / 2048 * 100)
            for Lm, c in ((2.0, "#d6283c"), (4.0, "#3b1a78")):
                ex = r['exact'][str(Lm)]; axs[1].plot(ex['trX'] / ex['dimp'], trXn(r, Lm), 'o', color=c, ms=3)
                axs[2].plot(ex['f'][0], f1(r, Lm), 'o', color=c, ms=3)
    axs[0].hist(err, bins=20, color="#8e2a8a"); axs[0].set_xlabel("max |KPM $-$ exact| of $N(\\Lambda)$ (% of dim)"); axs[0].set_ylabel("configurations")
    for a, nm in ((axs[1], r"$\mathrm{Tr}\,X/\dim p$"), (axs[2], r"$f_1$")):
        lim = a.get_xlim(); a.plot(lim, lim, 'k:', lw=0.8); a.set_xlabel("exact (dense)"); a.set_ylabel("matrix-free"); a.set_title(nm)
    axs[1].plot([], [], 'o', color="#d6283c", label=r"$\Lambda=2$"); axs[1].plot([], [], 'o', color="#3b1a78", label=r"$\Lambda=4$"); axs[1].legend(fontsize=7)
    fig.suptitle("Validation of the matrix-free estimators against dense diagonalisation (4$^4$)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig12_validation_matrix_free.png"); plt.close(fig)

# ------------------------------------------------------------------ results table
def row(L, b):
    rr = DATA[L][b]; g = lambda fn: ms([fn(r) for r in rr])
    pl = g(lambda r: r['plaq']); l1 = g(lambda r: r['lamM'][0]); l1d = g(lambda r: r['lam1D'])
    cop = g(lambda r: float(r['ndistinct'] >= 2)); nd = g(lambda r: r['ndistinct'])
    N2 = g(lambda r: np.interp(2.0, GRID, r['N']) / L ** 4); N4 = g(lambda r: np.interp(4.0, GRID, r['N']) / L ** 4)
    t2 = g(lambda r: trXn(r, 2.0)); t4 = g(lambda r: trXn(r, 4.0)); f4 = g(lambda r: f1(r, 4.0)); q4 = g(lambda r: lam1ray(r, 4.0))
    minl1 = min(r['lamM'][0] for r in rr)
    return (f"| {L}$^4$ | {b:g} | {len(rr)} | {pl[0]:.4f}±{pl[1]:.4f} | {l1[0]:.4f}±{l1[1]:.4f} | {minl1:.4f} | {l1d[0]:.3f} | {cop[0]:.2f} | {nd[0]:.2f} | "
            f"{N2[0]:.3f} | {N4[0]:.3f} | {t2[0]:.3f}±{t2[1]:.3f} | {t4[0]:.3f}±{t4[1]:.3f} | {f4[0]:.4f} | {q4[0]:.4f} |")


lines = ["| lattice | β | n cfg | ⟨plaq⟩ | ⟨λ1(M)⟩ | min λ1 | ⟨λ1(Δ)⟩ | frac. with copies | distinct minima | N(2)/V | N(4)/V | TrX/dim p (Λ=2) | TrX/dim p (Λ=4) | f1 (Λ=4) | λ1·Ray (Λ=4) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for L in DATA:
    for b in DATA[L]:
        lines.append(row(L, b))
open("tabela_resultados.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
