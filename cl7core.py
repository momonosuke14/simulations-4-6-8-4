r"""
cl7core: camada base independente (SU(3), rede periodica L^4) com a mesma interface
que cl6.py espera de ColorLattice 4/5.

  * Lattice, T (geradores de Gell-Mann/2), hot_start, cold_start, random_su3,
    gauge_transform, mean_plaquette
  * heatbath_sweep_fast  : Cabibbo-Marinari + Kennedy-Pendleton, 8 grupos
                           (direcao x paridade do sitio), todos independentes
  * landau_fix_fast      : sobre-relaxacao SU(2) por paridade + Newton-CG com a
                           Hessiana exata (sem matriz densa)
  * fp_hessian_fast      : Hessiana de Faddeev-Popov (esparsa, csr)
  * wilson_flow          : fluxo de gradiente de Luescher, RK3 de 3a ordem

Convencoes (iguais as do artigo):
  U_mu(x) -> g(x) U_mu(x) g(x+mu)^+ ;   F[g] = - sum Re Tr U_mu(x)  (Landau)
  S = beta * sum_p (1 - Re Tr P / 3)
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------- algebra
def _gellmann():
    l = np.zeros((8, 3, 3), complex)
    l[0][0, 1] = l[0][1, 0] = 1
    l[1][0, 1] = -1j; l[1][1, 0] = 1j
    l[2][0, 0] = 1; l[2][1, 1] = -1
    l[3][0, 2] = l[3][2, 0] = 1
    l[4][0, 2] = -1j; l[4][2, 0] = 1j
    l[5][1, 2] = l[5][2, 1] = 1
    l[6][1, 2] = -1j; l[6][2, 1] = 1j
    l[7] = np.diag([1, 1, -2]) / np.sqrt(3)
    return l / 2.0


T = _gellmann()                                              # Tr(Ta Tb) = delta/2
S_ANTI = 0.5 * (np.einsum('aij,bjk->abik', T, T) + np.einsum('bij,ajk->abik', T, T))


def dag(A):
    return np.conj(np.swapaxes(A, -1, -2))


# ---------------------------------------------------------------- lattice
class Lattice:
    def __init__(self, L):
        assert L % 2 == 0
        self.L = L; self.V = V = L ** 4
        shp = (L,) * 4
        c = np.stack(np.unravel_index(np.arange(V), shp), axis=1)
        self.coords = c
        self.fwd = np.empty((V, 4), int); self.bwd = np.empty((V, 4), int)
        for mu in range(4):
            cf = c.copy(); cf[:, mu] = (cf[:, mu] + 1) % L
            cb = c.copy(); cb[:, mu] = (cb[:, mu] - 1) % L
            self.fwd[:, mu] = np.ravel_multi_index(cf.T, shp)
            self.bwd[:, mu] = np.ravel_multi_index(cb.T, shp)
        self.parity = c.sum(1) % 2
        self.sites_par = [np.where(self.parity == p)[0] for p in (0, 1)]
        self._hess_idx = None


# ---------------------------------------------------------------- group helpers
def reunitarize(U):
    r0 = U[..., 0, :]; r1 = U[..., 1, :]
    r0 = r0 / np.linalg.norm(r0, axis=-1, keepdims=True)
    r1 = r1 - np.sum(np.conj(r0) * r1, axis=-1, keepdims=True) * r0
    r1 = r1 / np.linalg.norm(r1, axis=-1, keepdims=True)
    r2 = np.conj(np.cross(r0, r1))
    return np.stack([r0, r1, r2], axis=-2)


def random_su3(rng, n):
    Z = rng.normal(size=(n, 3, 3)) + 1j * rng.normal(size=(n, 3, 3))
    Q, R = np.linalg.qr(Z)
    d = np.diagonal(R, axis1=-2, axis2=-1)
    Q = Q * (d / np.abs(d))[:, None, :]
    det = np.linalg.det(Q)
    return Q / (det ** (1 / 3))[:, None, None]


def hot_start(lat, rng):
    return random_su3(rng, lat.V * 4).reshape(lat.V, 4, 3, 3)


def cold_start(lat):
    U = np.zeros((lat.V, 4, 3, 3), complex)
    U[..., 0, 0] = U[..., 1, 1] = U[..., 2, 2] = 1
    return U


def gauge_transform(lat, U, g):
    out = np.empty_like(U)
    for mu in range(4):
        out[:, mu] = g @ U[:, mu] @ dag(g[lat.fwd[:, mu]])
    return out


def expm_herm(H):
    """exp(i H) for Hermitian 3x3 H, batched."""
    w, v = np.linalg.eigh(H)
    return (v * np.exp(1j * w)[..., None, :]) @ dag(v)


def expm_anti(X):
    """exp(X) for anti-Hermitian 3x3 X, batched."""
    return expm_herm(-1j * X)


def plaquette_field(lat, U, mu, nu):
    xm = lat.fwd[:, mu]; xn = lat.fwd[:, nu]
    return U[:, mu] @ U[xm, nu] @ dag(U[xn, mu]) @ dag(U[:, nu])


def mean_plaquette(lat, U):
    s = 0.0
    for mu in range(4):
        for nu in range(mu + 1, 4):
            s += np.trace(plaquette_field(lat, U, mu, nu), axis1=-2, axis2=-1).real.sum()
    return s / (3.0 * 6 * lat.V)


def wilson_action(lat, U, beta):
    return beta * 6 * lat.V * (1.0 - mean_plaquette(lat, U))


# ---------------------------------------------------------------- staples
def staple(lat, U, mu, sites):
    """Sigma with Tr(U_mu(x) Sigma) = sum of the 6 plaquettes containing the link."""
    S = np.zeros((len(sites), 3, 3), complex)
    xpm = lat.fwd[sites, mu]
    for nu in range(4):
        if nu == mu:
            continue
        xpn = lat.fwd[sites, nu]; xmn = lat.bwd[sites, nu]; xpmmn = lat.bwd[xpm, nu]
        S += U[xpm, nu] @ dag(U[xpn, mu]) @ dag(U[sites, nu])
        S += dag(U[xpmmn, nu]) @ dag(U[xmn, mu]) @ U[xmn, nu]
    return S


# ---------------------------------------------------------------- heat bath
def _kp_a0(alpha, rng):
    """a0 ~ sqrt(1-a0^2) exp(alpha a0) on [-1,1]: Kennedy-Pendleton (alpha>=0.7),
    plain rejection for small alpha."""
    n = alpha.size
    a0 = np.empty(n)
    big = alpha >= 0.7
    todo = np.where(big)[0]
    while todo.size:
        al = alpha[todo]; m = todo.size
        r = 1.0 - rng.random((4, m))                    # (0,1]
        x = -np.log(r[0]) / al; y = -np.log(r[1]) / al
        d = y + x * np.cos(2 * np.pi * r[2]) ** 2
        ok = r[3] ** 2 <= 1.0 - d / 2.0
        a0[todo[ok]] = 1.0 - d[ok]
        todo = todo[~ok]
    todo = np.where(~big)[0]
    while todo.size:
        al = alpha[todo]; m = todo.size
        x = rng.uniform(-1, 1, m)
        ok = rng.random(m) <= np.sqrt(1 - x * x) * np.exp(al * (x - 1.0))
        a0[todo[ok]] = x[ok]
        todo = todo[~ok]
    return a0


def _su2_matrix(a0, rng):
    n = a0.size
    r = np.sqrt(np.maximum(1 - a0 ** 2, 0))
    ct = rng.uniform(-1, 1, n); ph = rng.uniform(0, 2 * np.pi, n); st = np.sqrt(1 - ct ** 2)
    a1 = r * st * np.cos(ph); a2 = r * st * np.sin(ph); a3 = r * ct
    X = np.empty((n, 2, 2), complex)
    X[:, 0, 0] = a0 + 1j * a3; X[:, 0, 1] = a2 + 1j * a1
    X[:, 1, 0] = -a2 + 1j * a1; X[:, 1, 1] = a0 - 1j * a3
    return X


SUBGROUPS = ((0, 1), (0, 2), (1, 2))


def hb_update_links(Ul, Sg, beta, rng):
    """Cabibbo-Marinari update of N links Ul given their staples Sg (weight exp(beta/3 Re Tr(U Sg)))."""
    Ul = Ul.copy()
    n = len(Ul)
    W = Ul @ Sg
    for (i, j) in SUBGROUPS:
        w11 = W[:, i, i]; w12 = W[:, i, j]; w21 = W[:, j, i]; w22 = W[:, j, j]
        p = 0.5 * (w11 + np.conj(w22)); q = 0.5 * (w12 - np.conj(w21))
        k = np.sqrt(np.abs(p) ** 2 + np.abs(q) ** 2)
        kk = np.where(k > 1e-14, k, 1.0)
        ut = np.empty((n, 2, 2), complex)
        ut[:, 0, 0] = p / kk; ut[:, 0, 1] = q / kk
        ut[:, 1, 0] = -np.conj(q) / kk; ut[:, 1, 1] = np.conj(p) / kk
        ut[k <= 1e-14] = np.eye(2)
        alpha = np.maximum(2.0 * beta * k / 3.0, 1e-8)
        X = _su2_matrix(_kp_a0(alpha, rng), rng)
        R = X @ dag(ut)
        idx = [i, j]
        Ul[:, idx, :] = R @ Ul[:, idx, :]
        W[:, idx, :] = R @ W[:, idx, :]
    return Ul


def _heatbath_links(lat, U, mu, sites, beta, rng):
    U[sites, mu] = hb_update_links(U[sites, mu], staple(lat, U, mu, sites), beta, rng)


def heatbath_sweep_fast(lat, U, beta, rng):
    for mu in range(4):
        for par in (0, 1):
            _heatbath_links(lat, U, mu, lat.sites_par[par], beta, rng)
    U[:] = reunitarize(U)


# ---------------------------------------------------------------- Landau gauge
def landau_F(U):
    return -np.einsum('xmii->', U).real


def landau_grad(lat, U):
    """dF/d omega_x^a at omega = 0 : (V,8)."""
    tr = np.einsum('aij,xmji->xma', T, U).imag            # Im Tr(T_a U_mu(x))
    g = tr.sum(1)
    for mu in range(4):
        g = g - tr[lat.bwd[:, mu], mu]
    return g


def _blk_idx(rs, cs):
    a = np.arange(8)
    R = (8 * rs[:, None, None] + a[None, :, None]) + 0 * (8 * cs[:, None, None] + a[None, None, :])
    C = (8 * cs[:, None, None] + a[None, None, :]) + 0 * (8 * rs[:, None, None] + a[None, :, None])
    return R.ravel(), C.ravel()


def fp_hessian_fast(lat, U, dense=False):
    """Hessian of F at omega=0 (8V x 8V), sparse csr (or dense).
    per link (x->y): (x,x)+=D, (y,y)+=D, (x,y)=C, (y,x)=C^T with
    D_ab = Re Tr(1/2{Ta,Tb} U),  C_ab = -Re Tr(Ta U Tb)."""
    V = lat.V
    if lat._hess_idx is None:
        x = np.arange(V); rows, cols = [], []
        for mu in range(4):
            y = lat.fwd[:, mu]
            for rs, cs in ((x, y), (x, x), (y, y), (y, x)):
                r, c = _blk_idx(rs, cs); rows.append(r); cols.append(c)
        lat._hess_idx = (np.concatenate(rows), np.concatenate(cols))
    vals = []
    for mu in range(4):
        Ul = U[:, mu]
        D = np.einsum('abij,xji->xab', S_ANTI, Ul).real
        TU = np.einsum('aij,xjk->xaik', T, Ul)
        C = -np.einsum('xaik,bki->xab', TU, T).real
        vals += [C.ravel(), D.ravel(), D.ravel(), np.swapaxes(C, 1, 2).ravel()]
    r, c = lat._hess_idx
    M = sp.csr_matrix((np.concatenate(vals), (r, c)), shape=(8 * V, 8 * V))
    return M.toarray() if dense else M


def const_modes(lat):
    V = lat.V
    C = np.zeros((8 * V, 8))
    for a in range(8):
        C[a::8, a] = 1.0 / np.sqrt(V)
    return C


def _proj(x, V):
    """remove the 8 constant modes from x (8V, [R])."""
    X = x.reshape(V, 8, -1)
    return (X - X.mean(0, keepdims=True)).reshape(x.shape)


def cg_reduced(M, b, V, tol=1e-12, maxit=2000):
    """Solve M x = b in the complement of the constant modes (b assumed orthogonal)."""
    b = _proj(b, V)
    diag = M.diagonal(); inv = 1.0 / np.maximum(diag, 1e-12)
    x = np.zeros_like(b); r = b.copy(); z = _proj(inv * r, V); p = z.copy()
    rz = r @ z; bn = np.linalg.norm(b)
    neg = False
    for it in range(maxit):
        Ap = _proj(M @ p, V); pAp = p @ Ap
        if pAp <= 0:
            neg = True; break
        al = rz / pAp; x += al * p; r -= al * Ap
        if np.linalg.norm(r) <= tol * bn:
            break
        z = _proj(inv * r, V); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2
    return x, it, neg


def _apply_g(lat, U, g):
    return gauge_transform(lat, U, g)


def _or_sweep(lat, U, omega, rng):
    """one over-relaxation sweep (maximise sum Re Tr g_x W_x) on both parities, in place.
    Site update: U_mu(x) -> g U_mu(x),  U_mu(x-mu) -> U_mu(x-mu) g^+."""
    for par in (0, 1):
        sites = lat.sites_par[par]
        W = np.zeros((len(sites), 3, 3), complex)
        for mu in range(4):
            W += U[sites, mu] + dag(U[lat.bwd[sites, mu], mu])
        # we maximise Re Tr(g W^+)?  Re Tr(g_x U_mu(x)) and Re Tr(U_mu(x-mu) g_x^+)=Re Tr(g_x U^+)
        # => functional  Re Tr( g_x * (sum_mu [U_mu(x) + U_mu(x-mu)^+]) ) = Re Tr(g W)
        G = np.broadcast_to(np.eye(3, dtype=complex), (len(sites), 3, 3)).copy()
        Wc = W.copy()
        for (i, j) in SUBGROUPS:
            w11 = Wc[:, i, i]; w12 = Wc[:, i, j]; w21 = Wc[:, j, i]; w22 = Wc[:, j, j]
            p = 0.5 * (w11 + np.conj(w22)); q = 0.5 * (w12 - np.conj(w21))
            k = np.sqrt(np.abs(p) ** 2 + np.abs(q) ** 2)
            kk = np.where(k > 1e-14, k, 1.0)
            ut = np.empty((len(sites), 2, 2), complex)
            ut[:, 0, 0] = p / kk; ut[:, 0, 1] = q / kk
            ut[:, 1, 0] = -np.conj(q) / kk; ut[:, 1, 1] = np.conj(p) / kk
            r = dag(ut)                                  # maximises Re Tr(r w)
            # r^omega
            a0 = np.clip(r[:, 0, 0].real, -1, 1); a3 = r[:, 0, 0].imag
            a2 = r[:, 0, 1].real; a1 = r[:, 0, 1].imag
            nrm = np.sqrt(a1 ** 2 + a2 ** 2 + a3 ** 2)
            th = np.arctan2(nrm, a0); nn = np.where(nrm > 1e-14, nrm, 1.0)
            c = np.cos(omega * th); s = np.sin(omega * th) / nn
            s = np.where(nrm > 1e-14, s, 0.0)
            c = np.where(nrm > 1e-14, c, 1.0)
            rw = np.empty_like(r)
            rw[:, 0, 0] = c + 1j * s * a3; rw[:, 0, 1] = s * a2 + 1j * s * a1
            rw[:, 1, 0] = -s * a2 + 1j * s * a1; rw[:, 1, 1] = c - 1j * s * a3
            idx = [i, j]
            Wc[:, idx, :] = rw @ Wc[:, idx, :]
            G[:, idx, :] = rw @ G[:, idx, :]
        for mu in range(4):
            U[sites, mu] = G @ U[sites, mu]
            xb = lat.bwd[sites, mu]
            U[xb, mu] = U[xb, mu] @ dag(G)


def landau_fix_fast(lat, U0, tol=1e-10, omega=1.7, newton_start=2e-3, max_or=20000,
                    max_newton=60, rng=None, verbose=False):
    """Landau gauge fixing. Returns (Ufixed, g_total, info)."""
    V = lat.V
    U = U0.copy()
    gtot = np.broadcast_to(np.eye(3, dtype=complex), (V, 3, 3)).copy()
    n_or = 0; n_newton = 0; converged = False
    # --- over-relaxation until we are inside a basin
    while n_or < max_or:
        for _ in range(10):
            _or_sweep(lat, U, omega, rng); n_or += 1
        U = reunitarize(U)
        gmax = np.abs(landau_grad(lat, U)).max()
        if verbose and n_or % 100 == 0:
            print(n_or, gmax)
        if gmax < newton_start:
            # try Newton; if it detects negative curvature, continue with OR
            ok, U, n_newton_i = _newton(lat, U, tol, max_newton)
            n_newton += n_newton_i
            if ok:
                converged = True
                break
    gtot = None
    return U, gtot, dict(converged=converged, F=float(landau_F(U)), n_or=n_or, n_newton=n_newton,
                         grad=float(np.abs(landau_grad(lat, U)).max()))


def _newton(lat, U, tol, max_newton):
    V = lat.V
    for it in range(max_newton):
        g = landau_grad(lat, U)
        gm = np.abs(g).max()
        if gm < tol:
            return True, U, it
        M = fp_hessian_fast(lat, U)
        step, nit, neg = cg_reduced(M, -g.reshape(-1), V, tol=1e-11)
        if neg:
            return False, U, it
        F0 = landau_F(U)
        s = 1.0
        for _ in range(12):
            om = (s * step).reshape(V, 8)
            H = np.einsum('xa,aij->xij', om, T)
            Un = gauge_transform(lat, U, expm_herm(H))
            if landau_F(Un) <= F0 + 1e-13:
                U = reunitarize(Un); break
            s *= 0.5
        else:
            return False, U, it
    return np.abs(landau_grad(lat, U)).max() < tol, U, max_newton


# ---------------------------------------------------------------- Wilson flow
def _Z(lat, U, beta_unused=None):
    """Z = - P_ah( U Sigma ) , dU/dt = Z U  (Luescher normalisation, flow time t)."""
    Zf = np.empty_like(U)
    allsites = np.arange(lat.V)
    for mu in range(4):
        W = U[:, mu] @ staple(lat, U, mu, allsites)
        A = 0.5 * (W - dag(W))
        tr = np.trace(A, axis1=-2, axis2=-1)[:, None, None] / 3.0
        Zf[:, mu] = -(A - tr * np.eye(3))
    return Zf


def _exp_all(X):
    sh = X.shape
    return expm_anti(X.reshape(-1, 3, 3)).reshape(sh)


def wilson_flow(lat, U, t_total, steps=4):
    """Luescher third-order Runge-Kutta, `steps` steps of size t_total/steps."""
    eps = t_total / steps
    V0 = U.copy()
    for _ in range(steps):
        Z0 = eps * _Z(lat, V0)
        W1 = _exp_all(0.25 * Z0) @ V0
        Z1 = eps * _Z(lat, W1)
        W2 = _exp_all((8.0 / 9.0) * Z1 - (17.0 / 36.0) * Z0) @ W1
        Z2 = eps * _Z(lat, W2)
        V0 = _exp_all(0.75 * Z2 - (8.0 / 9.0) * Z1 + (17.0 / 36.0) * Z0) @ W2
    return reunitarize(V0)
