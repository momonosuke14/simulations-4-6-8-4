r"""
cl7uv: operador ultravioleta  H_hat_tau  e teste matrix-free da Conjectura 1.

Mesmas definicoes de cl6.py (Delta_A = 1/2 D^*D, K_bar = media do nucleo de calor ao
longo do fluxo de Wilson, H_hat = -(1/tau) ln K_bar), com duas adicoes:

  * kpm_and_filters : UMA recorrencia de Chebyshev em x = 2 K_bar - 1 que entrega, ao
    mesmo tempo, (i) os momentos estocasticos (funcao de contagem N(Lambda)) e
    (ii) o projetor suave  p~_Lambda = sum_k g_k c_k(Lambda) T_k(x)  aplicado a
    vetores de teste e a autovetores do operador de Faddeev-Popov.
    O nucleo de Jackson mantem 0 <= p~ <= 1, logo  ||p~ M^+ p~|| <= ||M^+||.
  * conj1_estimators : f_k = ||p~ v_k||^2, Tr(p~ M^+ p~) (Hutchinson + CG reduzido),
    Rayleigh (cota inferior) de ||X||.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import ive

from cl7core import T, Lattice, wilson_flow, cg_reduced, _proj

LMAX = 8.0


# ---------------------------------------------------------------- Laplacian
def adjoint_links(U):
    V = U.shape[0]
    out = np.empty((V, 4, 8, 8))
    Ud = np.conj(np.swapaxes(U, -1, -2))
    for mu in range(4):
        TU = np.einsum('aij,xjk->xaik', T, U[:, mu])
        TUd = np.einsum('bkl,xli->xbki', T, Ud[:, mu])
        out[:, mu] = 2.0 * np.einsum('xajk,xbkj->xab', TU, TUd).real
    return out


def covariant_laplacian(lat, U):
    """Delta_A = 1/2 D^*D on adjoint fields (8V x 8V), sparse, real symmetric."""
    V = lat.V
    R = adjoint_links(U)
    a = np.arange(8); x = np.arange(V)
    rows, cols, vals = [], [], []
    for mu in range(4):
        y = lat.fwd[:, mu]
        rr = (8 * x[:, None, None] + a[None, :, None]) * np.ones((1, 1, 8), int)
        cc = (8 * y[:, None, None] + a[None, None, :]) * np.ones((1, 8, 1), int)
        rows += [rr.ravel(), cc.ravel()]; cols += [cc.ravel(), rr.ravel()]
        vals += [(-0.5 * R[:, mu]).ravel()] * 2
    idx = np.arange(8 * V)
    rows.append(idx); cols.append(idx); vals.append(np.full(8 * V, 4.0))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(8 * V, 8 * V))


def flow_frames(lat, U, T_total, n, steps=4):
    dt = T_total / n
    fr = [U]
    for _ in range(n):
        fr.append(wilson_flow(lat, fr[-1], dt, steps))
    w = np.full(n + 1, dt); w[0] = w[-1] = dt / 2
    return fr, w / T_total


# ---------------------------------------------------------------- dense reference
def H_hat_exact(lat, U, tau, T_total, n=2, frames=None):
    if frames is None:
        frames, w = flow_frames(lat, U, T_total, n)
    else:
        frames, w = frames
    Kb = np.zeros((8 * lat.V,) * 2); lam0 = None
    for wi, Ui in zip(w, frames):
        lam, W = np.linalg.eigh(covariant_laplacian(lat, Ui).toarray())
        Kb += wi * (W * np.exp(-tau * lam)) @ W.T
        if lam0 is None:
            lam0 = lam
    Kb = (Kb + Kb.T) / 2
    kap, W = np.linalg.eigh(Kb)
    h = -np.log(kap) / tau
    o = np.argsort(h)
    return Kb, h[o], W[:, o], lam0


# ---------------------------------------------------------------- Chebyshev
def heat_coeffs(tau, lmax=LMAX, tol=1e-16):
    a = tau * lmax / 2.0
    c = [ive(0, a)]; k = 1
    while True:
        ck = 2 * (-1) ** k * ive(k, a)
        c.append(ck)
        if abs(ck) < tol and k > 3:
            break
        k += 1
    return np.array(c)


def cheb_heat_apply(Delta, V_, coeffs, lmax=LMAX):
    xop = lambda v: (2.0 / lmax) * (Delta @ v) - v
    t0 = V_; t1 = xop(V_)
    out = coeffs[0] * t0 + coeffs[1] * t1
    for kk in range(2, len(coeffs)):
        t2 = 2 * xop(t1) - t0
        out += coeffs[kk] * t2
        t0, t1 = t1, t2
    return out


def jackson(N):
    k = np.arange(N)
    return ((N - k + 1) * np.cos(np.pi * k / (N + 1)) +
            np.sin(np.pi * k / (N + 1)) / np.tan(np.pi / (N + 1))) / (N + 1)


def step_coeffs(x0, N):
    """Chebyshev coefficients of 1[x >= x0], Jackson damped (length N)."""
    th = np.arccos(np.clip(x0, -1, 1)); k = np.arange(1, N)
    c = np.empty(N); c[0] = th / np.pi; c[1:] = (2 / np.pi) * np.sin(k * th) / k
    return c * jackson(N)


def kpm_and_filters(Deltas, weights, tau, nmom, Z, nprobe, Lams):
    """Z: (n, R) block; the first `nprobe` columns are Rademacher probes, the rest are
    arbitrary vectors that only get filtered.  Returns (mu, filt) with
    mu[k] ~ Tr T_k(2 K_bar - 1)  and filt[Lam] = p~_Lam Z  (n, R)."""
    n = Z.shape[0]
    hc = heat_coeffs(tau)
    def xK(v):
        s = np.zeros_like(v)
        for w, D in zip(weights, Deltas):
            s += w * cheb_heat_apply(D, v, hc)
        return 2 * s - v
    cs = {L: step_coeffs(2 * np.exp(-tau * L) - 1, nmom) for L in Lams}
    filt = {L: np.zeros_like(Z) for L in Lams}
    mu = np.zeros(nmom)
    t0 = Z; t1 = xK(Z)
    for k, t in ((0, t0), (1, t1)):
        mu[k] = np.sum(Z[:, :nprobe] * t[:, :nprobe]) / nprobe
        for L in Lams: filt[L] += cs[L][k] * t
    for k in range(2, nmom):
        t2 = 2 * xK(t1) - t0
        mu[k] = np.sum(Z[:, :nprobe] * t2[:, :nprobe]) / nprobe
        for L in Lams: filt[L] += cs[L][k] * t2
        t0, t1 = t1, t2
    return mu, filt


def kpm_count_H_below(mu, n, tau, Lams):
    N = len(mu); g = jackson(N); k = np.arange(1, N)
    out = []
    for L in np.atleast_1d(Lams):
        th = np.arccos(np.clip(2 * np.exp(-tau * L) - 1, -1, 1))
        val = g[0] * mu[0] * th / np.pi + (2 / np.pi) * np.sum(g[1:] * mu[1:] * np.sin(k * th) / k)
        out.append(n * val / mu[0])
    return np.array(out)


def cold_spectrum(L):
    p = 2 * np.pi * np.arange(L) / L
    g = np.meshgrid(*([1 - np.cos(p)] * 4), indexing='ij')
    return np.sort(np.repeat(sum(g).ravel(), 8))


# ---------------------------------------------------------------- Conjecture 1
def lowest_modes(M, V, k=6, tol=1e-11):
    """k lowest non-constant eigenpairs of the Faddeev-Popov operator (constants shifted to 1)."""
    C = np.zeros((8 * V, 8))
    for a in range(8):
        C[a::8, a] = 1.0 / np.sqrt(V)
    op = spla.LinearOperator(M.shape, matvec=lambda v: M @ v + C @ (C.T @ v), dtype=float,
                             matmat=lambda B: M @ B + C @ (C.T @ B))
    w, v = spla.eigsh(op, k=k, which='SA', tol=tol, ncv=max(4 * k, 60))
    o = np.argsort(w)
    return w[o], v[:, o]


def trace_X_hutchinson(M, V, Pz, cg_tol=1e-9):
    """E[(p~ z)^T M^+ (p~ z)]  for each column of Pz (already filtered probes). Returns per-probe values."""
    out = []
    for j in range(Pz.shape[1]):
        b = _proj(Pz[:, j].copy(), V)
        x, it, neg = cg_reduced(M, b, V, tol=cg_tol)
        out.append(float(b @ x))
    return np.array(out)
