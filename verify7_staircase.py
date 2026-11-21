"""Teste 12b (criterio corrigido apos ver a falha): contagem por KPM vs espectro frio exato,
avaliada nos pontos medios entre niveis distintos separados por >= 0.2 (para L=6 os niveis ficam em multiplos de 1/2), onde um nucleo suave nao perde meio degrau."""
import json, numpy as np
from cl7core import *
import cl7uv as uv
rng = np.random.default_rng(7); out = {}; ok = True
for L in (4, 6, 8):
    lt = Lattice(L); n = 8 * lt.V; D = uv.covariant_laplacian(lt, cold_start(lt))
    Z = rng.choice([-1., 1.], size=(n, 8)); mu, _ = uv.kpm_and_filters([D], [1.0], 0.5, 120, Z, 8, ())
    spec = uv.cold_spectrum(L); lev = np.unique(np.round(spec, 9)); gap = np.diff(lev)
    Lams = (lev[:-1] + gap / 2)[gap >= 0.2]
    Nc = uv.kpm_count_H_below(mu, n, 0.5, Lams)
    Nex = np.array([np.sum(spec <= x) for x in Lams]); e = float(np.abs(Nc - Nex).max() / n); out[f"L{L}"] = e; ok &= e < 0.03
    print(L, f"{100*e:.2f}% of the dimension")
json.dump(dict(passed=bool(ok), max_error_fraction=out), open("verify7_staircase_results.json", "w"))
print("PASS" if ok else "FAIL")
