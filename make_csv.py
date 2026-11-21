"""Exporta: metricas por configuracao (CSV), copias de Gribov (CSV), funcao de contagem (CSV),
dados da algebra de regimes (CSV, L=4) e series de Monte Carlo (CSV). Campos .npy ja foram salvos por run_ens.py."""
import glob, os, pickle, csv
import numpy as np
GRID = np.round(np.linspace(0, 8, 81), 10)
os.makedirs("export/metricas", exist_ok=True); os.makedirs("export/campos", exist_ok=True)
met, cop, cnt, reg, mc = [], [], [], [], []
for L in (4, 6, 8):
    for f in sorted(glob.glob(f"res_L{L}/beta_*.pkl")):
        b = float(os.path.basename(f)[5:-4]); recs = pickle.load(open(f, "rb"))
        for i, r in enumerate(recs):
            row = dict(L=L, beta=b, chain=r['chain'], cfg=i, plaquette=r['plaq'], F_landau=r['F'], ndistinct_minima=r['ndistinct'],
                       n_converged_copies=r['ncopies_conv'], lambda1_laplacian=r['lam1D'])
            for k, x in enumerate(r['lamM']): row[f'lambdaFP_{k+1}'] = x
            for Lm in (1.0, 2.0, 4.0):
                c = r[f'c1_{Lm}']
                row.update({f'trp_{Lm}': c['tr1'], f'trp2_{Lm}': c['tr2'], f'trX_{Lm}': c['trX'], f'trX_se_{Lm}': c['trX_se'],
                            f'trX_over_dim_{Lm}': c['trX'] / c['tr2'], f'f1_{Lm}': c['f'][0], f'lambda1_x_Rayleigh_{Lm}': r['lamM'][0] * c['ray']})
                for k, x in enumerate(c['f']): row[f'f{k+1}_{Lm}'] = x
            if L == 4:
                for Lm in (1.0, 2.0, 4.0):
                    e = r['exact'][str(Lm)]; row.update({f'exact_dimp_{Lm}': e['dimp'], f'exact_trX_over_dim_{Lm}': (e['trX'] / e['dimp'] if e['dimp'] else float('nan')), f'exact_f1_{Lm}': e['f'][0], f'exact_normX_{Lm}': e['nX']})
            met.append(row)
            for dF, l1 in zip(r['copies_F'], r['copies_l1']): cop.append(dict(L=L, beta=b, chain=r['chain'], cfg=i, F_minus_Fmin=dF, lambda1=l1))
            for lam, n_ in zip(GRID, r['N']): cnt.append(dict(L=L, beta=b, chain=r['chain'], cfg=i, Lambda=lam, N_over_V=n_ / L ** 4))
            if L == 4:
                for p, d in r['regimes'].items():
                    lc, Lm = p.split('_'); reg.append(dict(beta=b, chain=r['chain'], cfg=i, lambda_c=lc, Lambda=Lm, dim_Pc=d['dimPc'], dim_p=d['dimp'], dim_W=d['dimW'],
                                                           generic_angles=d['ngen'], commutator_norm=d['comm'], leakage=d['leak']))
        h = f"res_L{L}/hist_beta_{b:g}.npy"
        if os.path.exists(h):
            for (bb, ch, sw, pl, meas) in np.load(h): mc.append(dict(L=L, beta=bb, chain=int(ch), sweep=int(sw), plaquette=pl, after_thermalization=int(meas)))
        for g in glob.glob(f"res_L{L}/fields/*.npy"):
            dst = f"export/campos/{os.path.basename(g)}"
            if not os.path.exists(dst): np.save(dst, np.load(g))


def w(name, rows):
    if not rows: return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(f"export/metricas/{name}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(rows)


w("metricas_por_configuracao", met); w("copias_de_gribov", cop); w("funcao_de_contagem_N_sobre_V", cnt)
w("algebra_de_regimes_L4", reg); w("serie_monte_carlo_plaquete", mc)
print(len(met), "configuracoes;", len(cop), "copias;", len(mc), "pontos de serie")
