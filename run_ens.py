r"""
run_ens.py  --  ensembles SU(3) em L^4 (L = 4, 6, 8): camada infravermelha (Faddeev-Popov),
camada ultravioleta (Delta_A, H_hat_tau) e teste da Conjectura 1, tudo sem matriz densa.
Em L = 4 calcula-se tambem a referencia exata (densa) e os dados da algebra de regimes.

uso:  python3 run_ens.py L  betas  nchains  nper  nstarts  R  [outdir]
ex.:  python3 run_ens.py 6 1,3,5,6,8,12 2 3 8 16
"""
import sys, os, json, time
import numpy as np, scipy.sparse.linalg as spla
from cl7core import *
import cl7uv as uv

L = int(sys.argv[1]); BETAS = [float(b) for b in sys.argv[2].split(',')]
NCH, NPER, NST, R = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
OUT = sys.argv[7] if len(sys.argv) > 7 else f"res_L{L}"
os.makedirs(OUT, exist_ok=True)
NTH, SEP = 300, 40
TAU, TFLOW, NFR, NMOM, NMODES = 0.5, 0.2, 2, 120, 6
LAMS = (1.0, 2.0, 4.0)
GRID = np.round(np.linspace(0, 8, 81), 10)
REG = ((0.1, 4.0), (0.3, 4.0), (0.3, 6.0), (1.0, 4.0))
SEED = 20261120
lat = Lattice(L); V = lat.V; n = 8 * V
t0 = time.time()
json.dump(dict(L=L, betas=BETAS, nch=NCH, nper=NPER, nstarts=NST, R=R, ntherm=NTH, sep=SEP, tau=TAU,
               flow=TFLOW, nframes=NFR, nmom=NMOM, nmodes=NMODES, lams=LAMS, seed=SEED),
          open(f"{OUT}/settings.json", "w"), indent=1)


def process(U, rng):
    rec = {}
    # ---- gauge copies
    copies = []
    for s in range(NST):
        g = random_su3(rng, V)
        Uf, _, inf = landau_fix_fast(lat, gauge_transform(lat, U, g), tol=1e-10, max_or=6000, rng=rng)
        if not inf['converged']:
            continue
        M = fp_hessian_fast(lat, Uf)
        w, v = uv.lowest_modes(M, V, k=NNODE)
        copies.append((inf['F'], Uf, M, w, v))
    assert copies, "no converged gauge copy"
    copies.sort(key=lambda c: c[0])
    Fs = np.array([c[0] for c in copies]); l1s = np.array([c[3][0] for c in copies])
    rec['copies_F'] = Fs - Fs[0]; rec['copies_l1'] = l1s; rec['ncopies_conv'] = len(copies)
    ndist = 1 + int(np.sum(np.diff(Fs) > 1e-6))
    rec['ndistinct'] = ndist
    F0, Uf, M, w, v = copies[0]
    rec['F'] = F0; rec['lamM'] = w[:NMODES]
    # campos na grade (gauge-invariantes): densidade dos modos de Faddeev-Popov mais baixos e plaquete local
    rec['rho_modes'] = (v[:, :NMODES].reshape(V, 8, NMODES) ** 2).sum(1).T.reshape(NMODES, L, L, L, L).astype(np.float32)
    rec['plaq_field'] = local_plaq(Uf)
    assert w[0] > 0
    # ---- covariant Laplacian lowest eigenvalue (frame of the best copy)
    D0 = uv.covariant_laplacian(lat, Uf)
    rec['lam1D'] = float(spla.eigsh(D0, k=1, which='SA', tol=1e-10, return_eigenvectors=False)[0])
    # ---- UV: KPM + filters on the same Wilson-flow frames (no re-fixing)
    fr, wt = uv.flow_frames(lat, Uf, TFLOW, NFR)
    Ds = [D0] + [uv.covariant_laplacian(lat, u) for u in fr[1:]]
    Z = np.concatenate([rng.choice([-1.0, 1.0], size=(n, R)), v[:, :NMODES]], axis=1)
    mu, filt = uv.kpm_and_filters(Ds, list(wt), TAU, NMOM, Z, R, LAMS)
    rec['N'] = uv.kpm_count_H_below(mu, n, TAU, GRID)
    for Lm in LAMS:
        P = filt[Lm]
        tr1 = float(np.sum(Z[:, :R] * P[:, :R]) / R)              # Tr p~
        tr2 = float(np.mean(np.sum(P[:, :R] ** 2, axis=0)))      # Tr p~^2
        tx = uv.trace_X_hutchinson(M, V, P[:, :R])               # Tr(p~ M^+ p~) per probe
        fk = np.sum(P[:, R:] ** 2, axis=0)                        # weights of the lowest modes
        # Rayleigh lower bound of ||X|| with trial vector v1:  <p~v1, M^+ p~v1>
        b = P[:, R].copy(); x, _, _ = cg_reduced(M, b, V, tol=1e-11); ray = float(b @ x)
        rec[f'c1_{Lm}'] = dict(tr1=tr1, tr2=tr2, trX=float(tx.mean()), trX_se=float(tx.std(ddof=1) / np.sqrt(R)),
                               f=fk.tolist(), ray=ray)
    # ---- L = 4 only: exact (dense) reference + algebra of regimes
    if L == 4:
        Kb, h, Wk, l0 = uv.H_hat_exact(lat, Uf, TAU, TFLOW, frames=(fr, wt))
        wM, VM = np.linalg.eigh(M.toarray())
        rec['N_exact'] = np.searchsorted(np.sort(h), GRID, side='right')
        ex = {}
        for Lm in LAMS:
            m = h <= Lm
            if m.sum() == 0:
                ex[str(Lm)] = dict(dimp=0, trX=float('nan'), nX=float('nan'), f=[float('nan')] * NMODES); continue
            Q = VM[:, 8:].T @ Wk[:, m]; B = (Q.T / wM[8:]) @ Q
            ex[str(Lm)] = dict(dimp=int(m.sum()), trX=float(np.trace(B)), nX=float(np.linalg.eigvalsh(B)[-1]),
                               f=[float(np.sum((Wk[:, m].T @ VM[:, 8 + k]) ** 2)) for k in range(NMODES)])
        rec['exact'] = ex
        rg = {}
        for (lc, Lm) in REG:
            Pc = VM[:, wM >= lc]; pL = Wk[:, h <= Lm]
            sv = np.linalg.svd(Pc.T @ pL, compute_uv=False)
            gen = sv[(sv > 1e-9) & (sv < 1 - 1e-9)]
            frob = float(np.sum(sv ** 2))
            rg[f"{lc}_{Lm}"] = dict(dimPc=int(Pc.shape[1]), dimp=int(pL.shape[1]), dimW=int(np.sum(sv > 1 - 1e-9)),
                                    ngen=int(gen.size), comm=float(np.max(gen * np.sqrt(1 - gen ** 2))) if gen.size else 0.0,
                                    leak=float((pL.shape[1] - frob) / (n - Pc.shape[1])), cosmin_sep=float(np.min(np.diff(np.sort(gen))) if gen.size > 1 else np.nan),
                                    cos2=np.sort(gen ** 2).tolist())
        rec['regimes'] = rg
        rec['h_edges'] = (float(h[0]), float(h[-1]))
    return rec


NNODE = 8


def local_plaq(U):
    f = np.zeros(V)
    for mu in range(4):
        for nu in range(mu + 1, 4):
            f += np.trace(plaquette_field(lat, U, mu, nu), axis1=-2, axis2=-1).real / 18.0
    return f.reshape(L, L, L, L).astype(np.float32)


def run_beta(b):
    hist = []
    ss = np.random.SeedSequence([SEED, L, int(round(b * 1000))]).spawn(NCH); recs = []
    for ci in range(NCH):
        rng = np.random.default_rng(ss[ci]); U = hot_start(lat, rng)
        for t_ in range(NTH):
            heatbath_sweep_fast(lat, U, b, rng); hist.append((b, ci, t_, mean_plaquette(lat, U), 0))
        for k in range(NPER):
            for t_ in range(SEP):
                heatbath_sweep_fast(lat, U, b, rng); hist.append((b, ci, NTH + k * SEP + t_, mean_plaquette(lat, U), 1))
            r = process(U, rng); r['chain'] = ci; r['plaq'] = float(mean_plaquette(lat, U)); recs.append(r)
            print(f"  L={L} beta={b} chain {ci+1}/{NCH} cfg {k+1}/{NPER} lam1={r['lamM'][0]:.4f} "
                  f"distinct={r['ndistinct']} [{time.time()-t0:.0f}s]", flush=True)
    import pickle
    os.makedirs(f"{OUT}/fields", exist_ok=True)
    for i, r in enumerate(recs):
        tag = f"L{L}_beta{b:g}_chain{r['chain']}_cfg{i}"
        np.save(f"{OUT}/fields/{tag}_rho_modes.npy", r.pop('rho_modes')); np.save(f"{OUT}/fields/{tag}_plaq_field.npy", r.pop('plaq_field'))
    np.save(f"{OUT}/hist_beta_{b:g}.npy", np.array(hist))
    pickle.dump(recs, open(f"{OUT}/beta_{b}.pkl", "wb"))
    return recs


if __name__ == "__main__":
    for b in BETAS:
        if os.path.exists(f"{OUT}/beta_{b}.pkl"): continue
        recs = run_beta(b)
        print(f"L={L} beta={b:g} plaq={np.mean([r['plaq'] for r in recs]):.4f} lam1(M)={np.mean([r['lamM'][0] for r in recs]):.4f} "
              f"lam1(D)={np.mean([r['lam1D'] for r in recs]):.3f} [{time.time()-t0:.0f}s]", flush=True)
    print("done", flush=True)
