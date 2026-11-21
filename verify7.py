"""Bateria de verificacao independente (cl7core + cl7uv). Cada teste compara com algo que nao
compartilha a implementacao: diferencas finitas, integral de Haar, Metropolis explicito,
espectro frio exato, diagonalizacao densa, ou valores publicados no artigo."""
import json, time, sys
import numpy as np, scipy.sparse.linalg as spla
from scipy.special import iv
from cl7core import *
import cl7core as core
import cl7uv as uv

rng = np.random.default_rng(20261120)
t00 = time.time(); res = {}
def rep(name, ok, d):
    res[name] = dict(passed=bool(ok), detail=d)
    print(("PASS " if ok else "FAIL ") + name + " :: " + d + f" [{time.time()-t00:.0f}s]", flush=True)
    json.dump(res, open('verify7_results.json', 'w'), indent=1)

def therm(lat, beta, n=300, start='hot'):
    U = hot_start(lat, rng) if start == 'hot' else cold_start(lat)
    for _ in range(n): heatbath_sweep_fast(lat, U, beta, rng)
    return U

def blocked_err(x, nb=20):
    x = np.asarray(x); m = len(x) // nb
    b = x[:m * nb].reshape(nb, m).mean(1)
    return b.mean(), b.std(ddof=1) / np.sqrt(nb)

lat4 = Lattice(4); V4 = lat4.V

# 1. Hessian: simetria, c^T M c = 0, diferencas finitas de F (entradas aleatorias)
U = therm(lat4, 6.0, 100)
M = fp_hessian_fast(lat4, U).toarray(); C = const_modes(lat4)
def F_of(om):
    H = np.einsum('xa,aij->xij', om.reshape(V4, 8), T)
    return landau_F(gauge_transform(lat4, U, expm_herm(H)))
eps = 1e-4; worst = 0
idx = rng.choice(8 * V4, size=(30, 2))
for i, j in idx:
    e = np.zeros(8 * V4)
    def f(si, sj):
        e = np.zeros(8 * V4); e[i] += si * eps; e[j] += sj * eps; return F_of(e)
    if i == j:
        fd = (F_of(np.eye(8 * V4)[i] * eps) - 2 * F_of(0 * e) + F_of(-np.eye(8 * V4)[i] * eps)) / eps ** 2
    else:
        fd = (f(1, 1) - f(1, -1) - f(-1, 1) + f(-1, -1)) / (4 * eps ** 2)
    worst = max(worst, abs(fd - M[i, j]))
rep("hessian_fd", abs(M - M.T).max() < 1e-14 and np.abs(C.T @ M @ C).max() < 1e-12 and worst < 1e-6,
    f"asym {abs(M-M.T).max():.0e}; |c^T M c| {np.abs(C.T@M@C).max():.1e}; max |FD - M| over 30 entries {worst:.1e}")

# 2. Gradiente vs diferencas finitas
g = landau_grad(lat4, U).reshape(-1); worst = 0
for i in rng.choice(8 * V4, 40):
    e = np.zeros(8 * V4); e[i] = 1e-5
    worst = max(worst, abs((F_of(e) - F_of(-e)) / 2e-5 - g[i]))
rep("gradient_fd", worst < 1e-8, f"max deviation over 40 components {worst:.1e}")

# 3. Banho termico: distribuicao de a0 (Bessel) e um elo com staple fixo vs integral de Haar
bad = 0.0
for al in (0.8, 3.0, 12.0):
    a0 = core._kp_a0(np.full(400000, al), rng)
    bad = max(bad, abs(a0.mean() - iv(2, al) / iv(1, al)) / (a0.std() / np.sqrt(a0.size)))
for al in (0.2, 0.5):
    a0 = core._kp_a0(np.full(400000, al), rng)
    bad = max(bad, abs(a0.mean() - iv(2, al) / iv(1, al)) / (a0.std() / np.sqrt(a0.size)))
rep("kp_distribution", bad < 4.0, f"<a0> vs I2/I1 at alpha=0.2..12: worst deviation {bad:.1f} sigma")

Sg = (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))) * 0.8; beta = 2.5
N = 200000
Ul = random_su3(rng, N)
for _ in range(6): Ul = hb_update_links(Ul, np.broadcast_to(Sg, (N, 3, 3)), beta, rng)
obs = lambda Uu: np.stack([np.trace(Uu @ Sg, axis1=-2, axis2=-1).real, np.abs(np.trace(Uu, axis1=-2, axis2=-1)) ** 2])
hb = obs(Ul)
H = random_su3(rng, 2_000_000); wgt = np.exp(beta / 3 * np.trace(H @ Sg, axis1=-2, axis2=-1).real); wgt /= wgt.sum()
im = np.array([np.sum(wgt * o) for o in obs(H)])
sig = hb.std(1) / np.sqrt(N)
rep("single_link_haar", np.all(np.abs(hb.mean(1) - im) < 4 * np.maximum(sig, 2e-3)),
    f"heat bath {np.round(hb.mean(1),4)} vs importance-sampled Haar {np.round(im,4)} (sigma {np.round(sig,4)})")

# 4. Metropolis explicito (independente do codigo de staples) vs banho termico em 2^4, beta = 3
lat2 = Lattice(2)
def plaq_sum_link(U, x, mu):
    s = 0.0
    c = lat2.coords[x]
    for nu in range(4):
        if nu == mu: continue
        xm = lat2.fwd[x, mu]; xn = lat2.fwd[x, nu]
        s += np.trace(U[x, mu] @ U[xm, nu] @ U[xn, mu].conj().T @ U[x, nu].conj().T).real
        xb = lat2.bwd[x, nu]; xmb = lat2.bwd[xm, nu]
        s += np.trace(U[x, mu] @ U[xmb, nu].conj().T @ U[xb, mu].conj().T @ U[xb, nu]).real
    return s
def metropolis(lat, U, beta, nsw, rng):
    out = []
    for s in range(nsw):
        for x in range(lat.V):
            for mu in range(4):
                old = U[x, mu].copy(); s0 = plaq_sum_link(U, x, mu)
                for _ in range(3):
                    A = np.eye(3, dtype=complex) + 0.5 * (rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))
                    A = A / np.linalg.det(A) ** (1 / 3) if abs(np.linalg.det(A)) > 1e-3 else np.eye(3)
                    Q, R_ = np.linalg.qr(A); Q = Q * (np.diag(R_) / abs(np.diag(R_)))
                    Q = Q / np.linalg.det(Q) ** (1 / 3)
                    Un = Q @ U[x, mu]; U[x, mu] = Un; s1 = plaq_sum_link(U, x, mu)
                    if rng.random() < np.exp(beta / 3 * (s1 - s0)): s0 = s1
                    else: U[x, mu] = old
                    old = U[x, mu].copy()
        out.append(mean_plaquette(lat, U))
    return np.array(out)
Um = hot_start(lat2, rng); pm = metropolis(lat2, Um, 3.0, 1300, rng)[300:]
Uh = hot_start(lat2, rng); ph = []
for s in range(20000):
    heatbath_sweep_fast(lat2, Uh, 3.0, rng)
    if s >= 500: ph.append(mean_plaquette(lat2, Uh))
mm, em = blocked_err(pm, 10); mh, eh = blocked_err(ph, 20)
rep("heatbath_vs_metropolis_2x4_b3", abs(mm - mh) < 3.5 * np.hypot(em, eh),
    f"heat bath {mh:.4f}+-{eh:.4f}; Metropolis {mm:.4f}+-{em:.4f}; paper (2^4, beta=3) 0.2066+-0.0003")

# 5. Termalizacao hot vs cold e comparacao com valores publicados (4^4)
rows = []
ok = True
for beta, ref in ((3.0, 0.2052), (5.0, 0.4004), (6.0, 0.5967)):
    r = []
    for st in ('hot', 'cold'):
        U = therm(lat4, beta, 500, st); pl = []
        for s in range(7000):
            heatbath_sweep_fast(lat4, U, beta, rng)
            if s % 3 == 0: pl.append(mean_plaquette(lat4, U))
        r.append(blocked_err(pl, 20))
    ok &= abs(r[0][0] - r[1][0]) < 3.5 * np.hypot(r[0][1], r[1][1]) and abs(r[0][0] - ref) < 4 * max(r[0][1], 5e-4)
    rows.append(f"beta={beta:g}: hot {r[0][0]:.4f}+-{r[0][1]:.4f}, cold {r[1][0]:.4f}+-{r[1][1]:.4f}, paper {ref}")
rep("thermalization_and_published_plaquette_4x4", ok, "; ".join(rows))

# 6. Fixacao de Landau
U = therm(lat4, 6.0, 300)
g = random_su3(rng, V4); Uf, _, inf = landau_fix_fast(lat4, gauge_transform(lat4, U, g), tol=1e-11, rng=rng)
Mf = fp_hessian_fast(lat4, Uf).toarray(); w = np.linalg.eigvalsh(Mf)
Cm = const_modes(lat4)
rep("landau_fixing_4x4", inf['converged'] and inf['grad'] < 1e-11 and w[8] > 1e-6 and abs(w[7]) < 1e-9
    and np.abs(Mf @ Cm).max() < 1e-9,
    f"|grad F|_max {inf['grad']:.1e}; lowest reduced eigenvalue {w[8]:.5f}; max|M c| {np.abs(Mf@Cm).max():.1e}; n_or {inf['n_or']}")

# 7. Invariancia de calibre: plaquette e espectro de Delta_A
Ug = gauge_transform(lat4, U, random_su3(rng, V4))
e1 = np.linalg.eigvalsh(uv.covariant_laplacian(lat4, U).toarray()); e2 = np.linalg.eigvalsh(uv.covariant_laplacian(lat4, Ug).toarray())
Dsym = uv.covariant_laplacian(lat4, U)
rep("gauge_invariance", abs(mean_plaquette(lat4, U) - mean_plaquette(lat4, Ug)) < 1e-13 and np.abs(e1 - e2).max() < 1e-12
    and abs(Dsym - Dsym.T).max() < 1e-14 and e1[0] > 0 and e1[-1] < 8,
    f"plaquette diff {abs(mean_plaquette(lat4,U)-mean_plaquette(lat4,Ug)):.0e}; Delta spectrum diff {np.abs(e1-e2).max():.1e}; range [{e1[0]:.3f},{e1[-1]:.3f}] subset [0,8]")

# 8. Fluxo de Wilson: S decresce, ordem, covariancia, unitariedade
Ufl = therm(lat4, 6.0, 200)
S = [wilson_action(lat4, wilson_flow(lat4, Ufl, t, 4), 6.0) for t in np.linspace(0, 0.3, 7)]
ref = wilson_flow(lat4, Ufl, 0.2, 256)
errs = [np.abs(wilson_flow(lat4, Ufl, 0.2, n) - ref).max() for n in (2, 4, 8)]
order = [np.log2(errs[i] / errs[i + 1]) for i in range(2)]
g = random_su3(rng, V4)
cov = np.abs(gauge_transform(lat4, wilson_flow(lat4, Ufl, 0.1, 4), g) - wilson_flow(lat4, gauge_transform(lat4, Ufl, g), 0.1, 4)).max()
unit = np.abs(np.einsum('xmij,xmkj->xmik', wilson_flow(lat4, Ufl, 0.1, 4), wilson_flow(lat4, Ufl, 0.1, 4).conj()) - np.eye(3)).max()
rep("wilson_flow", np.all(np.diff(S) < 0) and min(order) > 2.8 and cov < 1e-12 and unit < 1e-13,
    f"S monotone: {bool(np.all(np.diff(S)<0))}; observed order {order[0]:.2f}, {order[1]:.2f}; gauge covariance {cov:.1e}; unitarity {unit:.1e}")

# 9. Rede fria: Hessiana e Delta contra Sum(1-cos p) com multiplicidade
okc = True; txt = []
for L in (2, 4):
    lt = Lattice(L); Uc = cold_start(lt)
    ex = uv.cold_spectrum(L)
    if L <= 4:
        eM = np.sort(np.linalg.eigvalsh(fp_hessian_fast(lt, Uc).toarray()))
        eD = np.sort(np.linalg.eigvalsh(uv.covariant_laplacian(lt, Uc).toarray()))
    else:
        eM = np.sort(np.linalg.eigvalsh(fp_hessian_fast(lt, Uc).toarray()))
        eD = np.sort(np.linalg.eigvalsh(uv.covariant_laplacian(lt, Uc).toarray()))
    d = max(np.abs(eM - ex).max(), np.abs(eD - ex).max()); okc &= d < 1e-11; txt.append(f"L={L}: {d:.1e}")
lt = Lattice(4); nz = np.round(np.sort(np.linalg.eigvalsh(fp_hessian_fast(lt, cold_start(lt)).toarray())), 6); nz = nz[8:]
mult = [int(np.sum(np.abs(nz - k) < 1e-6)) for k in range(1, 9)]
okc &= mult == [64, 224, 448, 560, 448, 224, 64, 8]
rep("cold_lattice_spectra", okc, "M and Delta vs sum(1-cos p), " + ", ".join(txt) + f"; L=4 multiplicities {mult}")

# 10. lambda_1: esparso (Lanczos) vs denso
w1 = []
for _ in range(3):
    Uq = therm(lat4, 6.0, 150); g = random_su3(rng, V4)
    Uf, _, inf = landau_fix_fast(lat4, gauge_transform(lat4, Uq, g), tol=1e-11, rng=rng)
    Mq = fp_hessian_fast(lat4, Uf)
    lam_sp = uv.lowest_modes(Mq, V4, k=4)[0][0]; lam_d = np.linalg.eigvalsh(Mq.toarray())[8]
    w1.append(abs(lam_sp - lam_d))
rep("lambda1_sparse_vs_dense", max(w1) < 1e-9, f"max |lanczos - dense| over 3 configs {max(w1):.1e}")

# 11. Nucleo de calor de Chebyshev e Hbar_tau -> Delta_A(0)
Uq = therm(lat4, 6.0, 200)
D = uv.covariant_laplacian(lat4, Uq); lam, W = np.linalg.eigh(D.toarray()); ex = (W * np.exp(-0.5 * lam)) @ W.T
hc = uv.heat_coeffs(0.5); I = np.eye(8 * V4)[:, :64]
ap = uv.cheb_heat_apply(D, I, hc)
errc = np.abs(ap - ex[:, :64]).max()
errs = []
for Tt in (1e-4, 1e-6):
    Kb, h, Wk, l0 = uv.H_hat_exact(lat4, Uq, 0.5, Tt, n=2)
    errs.append(np.abs(h - l0).max())
rep("chebyshev_and_T0_limit", errc < 1e-13 and errs[0] < 1e-3 and 5 < errs[0] / errs[1] < 200,
    f"Chebyshev vs exact {errc:.1e} with {len(hc)} terms; |H_tau - Delta0| = {errs[0]:.1e} (T=1e-4), {errs[1]:.1e} (T=1e-6): linear in T")

# 12. KPM: contagem vs exata em 4^4, e vs espectro frio exato em 4^4, 6^4, 8^4
Lams = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
fr, wt = uv.flow_frames(lat4, Uq, 0.2, 2); Ds = [uv.covariant_laplacian(lat4, u) for u in fr]
Kb, h, Wk, l0 = uv.H_hat_exact(lat4, Uq, 0.5, 0.2, frames=(fr, wt))
Z = rng.choice([-1., 1.], size=(8 * V4, 40)); mu, filt = uv.kpm_and_filters(Ds, list(wt), 0.5, 160, Z, 40, Lams)
Nk = uv.kpm_count_H_below(mu, 8 * V4, 0.5, np.array(Lams)); Ne = np.array([np.sum(h <= L) for L in Lams])
rep("kpm_counting_vs_exact_4x4", np.abs(Nk - Ne).max() / (8 * V4) < 0.005,
    f"max error {100*np.abs(Nk-Ne).max()/(8*V4):.2f}% of the dimension (40 probes, 160 moments)")
txt = []; okk = True
for L in (4, 6, 8):
    lt = Lattice(L); n = 8 * lt.V; Dc = uv.covariant_laplacian(lt, cold_start(lt))
    Zc = rng.choice([-1., 1.], size=(n, 8))
    muc, _ = uv.kpm_and_filters([Dc], [1.0], 0.5, 120, Zc, 8, ())
    Nc = uv.kpm_count_H_below(muc, n, 0.5, np.array(Lams)); spec = uv.cold_spectrum(L)
    Nex = np.array([np.sum(spec <= L_ + 1e-12) for L_ in Lams])
    e = np.abs(Nc - Nex).max() / n; okk &= e < 0.03; txt.append(f"L={L}: {100*e:.2f}%")
rep("kpm_cold_lattice_staircase", okk, "KPM vs exact cold counting function (L=4,6,8; staircase is hard for a smooth kernel), max error " + ", ".join(txt))

# 13. Proposicao 12 (tautologia) e Proposicao 10 (f1 <= lambda1 ||X|| <= 1), configuracao por configuracao
hk = np.sort(rng.uniform(0.1, 7, 200)); Lm = 2.3
omg = np.linspace(-12, 12, 480001); dw = omg[1] - omg[0]
num = sum(np.sum(((hh + omg >= 0) & (hh + omg <= Lm)) * np.exp(-omg)) * dw for hh in hk)
rep("proposition12_tautology", abs(num / ((1 - np.exp(-Lm)) * np.sum(np.exp(hk))) - 1) < 1e-3,
    f"ratio {num/((1-np.exp(-Lm))*np.sum(np.exp(hk))):.5f}")
okp = True; tx = []
for beta in (3.0, 6.0, 12.0):
    for _ in range(2):
        Uq = therm(lat4, beta, 200); g = random_su3(rng, V4)
        Uf, _, inf = landau_fix_fast(lat4, gauge_transform(lat4, Uq, g), tol=1e-11, rng=rng)
        Mq = fp_hessian_fast(lat4, Uf).toarray(); wM, VM = np.linalg.eigh(Mq)
        Kb, h, Wk, l0 = uv.H_hat_exact(lat4, Uf, 0.5, 0.2, n=2)
        for Lm_ in (2.0, 4.0):
            m = h <= Lm_; Q = VM[:, 8:].T @ Wk[:, m]; B = (Q.T / wM[8:]) @ Q
            nX = np.linalg.eigvalsh(B)[-1]; f1 = np.sum((Wk[:, m].T @ VM[:, 8]) ** 2)
            okp &= (f1 <= wM[8] * nX * (1 + 1e-9)) and (wM[8] * nX <= 1 + 1e-9)
            tx.append(wM[8] * nX / max(f1, 1e-300))
rep("proposition10_bounds", okp, f"f1 <= lambda1||X|| <= 1 in 12 cases (3 couplings x 2 configs x 2 cutoffs); ratio lambda1||X||/f1 in [{min(tx):.4f},{max(tx):.4f}]")

# 14. Halmos: dimensao da janela e comutador <= 1/2 (4^4)
Pc = (VM[:, 8:][:, wM[8:] >= 0.3]); pL = Wk[:, h <= 4.0]
sv = np.linalg.svd(Pc.T @ pL, compute_uv=False)
nW = int(np.sum(sv > 1 - 1e-9))
Pm = Pc @ Pc.T; pm_ = pL @ pL.T; comm = np.linalg.norm(Pm @ pm_ - pm_ @ Pm, 2)
rep("halmos_two_projections", nW == Pc.shape[1] + pL.shape[1] - 8 * V4 and comm <= 0.5 + 1e-12,
    f"dim P_c={Pc.shape[1]}, dim p={pL.shape[1]}, unit singular values={nW}, ||[P,p]||={comm:.3f} <= 1/2")

# 15. Estimador da Conjectura 1 (filtro suave + Hutchinson) vs exato em 4^4
Lc = (2.0, 4.0)
fr, wt = uv.flow_frames(lat4, Uf, 0.2, 2); Ds = [uv.covariant_laplacian(lat4, u) for u in fr]
Kb, h, Wk, l0 = uv.H_hat_exact(lat4, Uf, 0.5, 0.2, frames=(fr, wt))
Mq = fp_hessian_fast(lat4, Uf); wM, VM = np.linalg.eigh(Mq.toarray()); lw, vl = uv.lowest_modes(Mq, V4, k=4)
R = 64; Z = np.concatenate([rng.choice([-1., 1.], size=(8 * V4, R)), vl[:, :1]], axis=1)
mu, filt = uv.kpm_and_filters(Ds, list(wt), 0.5, 160, Z, R, Lc)
txt = []; okx = True
for L_ in Lc:
    m = h <= L_; Q = VM[:, 8:].T @ Wk[:, m]; B = (Q.T / wM[8:]) @ Q
    exact = np.trace(B) / m.sum(); f1e = np.sum((Wk[:, m].T @ VM[:, 8]) ** 2)
    P = filt[L_]; t = uv.trace_X_hutchinson(Mq, V4, P[:, :R]); trp = np.sum(Z[:, :R] * P[:, :R]) / R
    est = t.mean() / trp; err = t.std(ddof=1) / np.sqrt(R) / trp; f1s = np.sum(P[:, R] ** 2)
    okx &= abs(est - exact) / exact < 0.06 and abs(f1s - f1e) < 0.02
    txt.append(f"Lam={L_}: Tr X/dim p exact {exact:.4f}, estimate {est:.4f}+-{err:.4f}; f1 exact {f1e:.4f}, filtered {f1s:.4f}")
rep("conjecture1_estimator_vs_exact_4x4", okx, "; ".join(txt))
print(sum(v['passed'] for v in res.values()), "of", len(res), "tests passed")
