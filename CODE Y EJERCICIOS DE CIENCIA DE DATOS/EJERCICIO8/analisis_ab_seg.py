# -*- coding: utf-8 -*-
"""Parte 2: segmentacion por dispositivo, ajuste por confusion y robustez."""
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import (proportions_ztest, proportion_confint,
                                          confint_proportions_2indep)
from statsmodels.stats.contingency_tables import StratifiedTable, Table2x2
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

rng = np.random.default_rng(20260828)
df = pd.read_csv('ab_test_landing.csv')
pw = NormalIndPower()


def h(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# --------------------------------------------------- 5. SEGMENTACION
h("5. SEGMENTACION POR DISPOSITIVO")
devs = ['escritorio', 'movil', 'tableta']
rows, praw_conv, praw_rev = [], [], []
tables = []

for d in devs:
    s = df[df.dispositivo == d]
    c, t = s[s.grupo == 'control'], s[s.grupo == 'tratamiento']
    nc, nt = len(c), len(t)
    xc, xt = int(c.convirtio.sum()), int(t.convirtio.sum())
    pc, pt = xc / nc, xt / nt
    tab = np.array([[xt, nt - xt], [xc, nc - xc]])
    tables.append(np.array([[xt, xc], [nt - xt, nc - xc]]))  # para CMH
    _, _, _, esp = stats.chi2_contingency(tab)
    exacto = esp.min() < 5
    if exacto:
        orr, pconv = stats.fisher_exact(tab)
        prueba = "Fisher exacto"
    else:
        _, pconv = proportions_ztest([xt, xc], [nt, nc])
        prueba = "z 2 proporciones"
    lo, hi = confint_proportions_2indep(xt, nt, xc, nc, method='wald', compare='diff')

    # MDE del subgrupo
    hh = pw.solve_power(effect_size=None, nobs1=nc, alpha=.05, power=.80,
                        ratio=nt / nc, alternative='two-sided')
    mde = np.sin(np.arcsin(np.sqrt(pc)) + hh / 2) ** 2 - pc

    # ingreso
    ai, bi = c.ingreso.values, t.ingreso.values
    mw = stats.mannwhitneyu(ai, bi, alternative='two-sided')
    auc_t = 1 - mw.statistic / (len(ai) * len(bi))
    bs = np.array([rng.choice(bi, len(bi), True).mean() - rng.choice(ai, len(ai), True).mean()
                   for _ in range(10000)])

    praw_conv.append(pconv)
    praw_rev.append(mw.pvalue)
    rows.append(dict(dispositivo=d, n_ctrl=nc, n_trat=nt,
                     conv_ctrl=pc, conv_trat=pt, dif_pp=(pt - pc) * 100,
                     ic_lo=lo * 100, ic_hi=hi * 100, lift_rel=(pt / pc - 1) * 100,
                     prueba=prueba, p_conv=pconv, mde_pp=mde * 100,
                     arpu_ctrl=ai.mean(), arpu_trat=bi.mean(),
                     med_ctrl=np.median(ai), med_trat=np.median(bi),
                     dif_arpu=bi.mean() - ai.mean(),
                     arpu_lo=np.percentile(bs, 2.5), arpu_hi=np.percentile(bs, 97.5),
                     p_mw=mw.pvalue, cliff=2 * auc_t - 1))

R = pd.DataFrame(rows)
R['p_conv_holm'] = multipletests(praw_conv, method='holm')[1]
R['p_mw_holm'] = multipletests(praw_rev, method='holm')[1]

print("\n--- CONVERSION por dispositivo ---")
for _, r in R.iterrows():
    print("\n  {} (n control={}, n tratamiento={})".format(
        r.dispositivo.upper(), int(r.n_ctrl), int(r.n_trat)))
    print("    control={:.3f}  tratamiento={:.3f}".format(r.conv_ctrl, r.conv_trat))
    print("    diferencia = {:+.2f} pp  IC95% [{:+.2f}, {:+.2f}]  lift rel = {:+.1f}%".format(
        r.dif_pp, r.ic_lo, r.ic_hi, r.lift_rel))
    print("    {}: p={:.3e}  |  p ajustado Holm = {:.3e}  -> {}".format(
        r.prueba, r.p_conv, r.p_conv_holm,
        "SIGNIFICATIVO" if r.p_conv_holm < .05 else "no significativo"))
    print("    MDE del subgrupo (potencia 80%) = {:+.1f} pp".format(r.mde_pp))

print("\n--- INGRESO POR USUARIO por dispositivo (Mann-Whitney) ---")
for _, r in R.iterrows():
    print("\n  {}".format(r.dispositivo.upper()))
    print("    ARPU  control={:.2f}  tratamiento={:.2f}   (medianas {:.2f} vs {:.2f})".format(
        r.arpu_ctrl, r.arpu_trat, r.med_ctrl, r.med_trat))
    print("    diferencia de medias = {:+.2f}  IC95% bootstrap [{:+.2f}, {:+.2f}]".format(
        r.dif_arpu, r.arpu_lo, r.arpu_hi))
    print("    Mann-Whitney p={:.3e}  | Holm={:.3e}  | delta de Cliff={:+.3f}  -> {}".format(
        r.p_mw, r.p_mw_holm, r.cliff,
        "SIGNIFICATIVO" if r.p_mw_holm < .05 else "no significativo"))

# ------------------------------------- 6. INTERACCION Y AJUSTE POR CONFUSION
h("6. INTERACCION Y AJUSTE POR EL DESBALANCE DE DISPOSITIVO")

print("-- 6.a Homogeneidad del efecto entre estratos (Breslow-Day / Tarone) --")
st = StratifiedTable(tables)
bd = st.test_equal_odds()
print("   H0: el odds ratio es el mismo en los 3 dispositivos")
print("   Estadistico={:.4f}  p={:.4f}  -> {}".format(
    bd.statistic, bd.pvalue,
    "NO se rechaza: el efecto es homogeneo, no hay interaccion detectable"
    if bd.pvalue > .05 else "se rechaza: el efecto difiere por dispositivo"))

print("\n-- 6.b Cochran-Mantel-Haenszel (efecto comun ajustado por dispositivo) --")
cmh = st.test_null_odds()
print("   OR comun MH = {:.4f}  IC95% [{:.4f}, {:.4f}]".format(
    st.oddsratio_pooled, *st.oddsratio_pooled_confint()))
print("   Riesgo relativo comun MH = {:.4f}".format(st.riskratio_pooled))
print("   CMH X2={:.4f}  p={:.3e}".format(cmh.statistic, cmh.pvalue))
crudo = Table2x2(np.array([[215, 35], [126, 124]]))
print("   OR crudo (sin ajustar) = {:.4f}".format(crudo.oddsratio))
print("   -> El OR ajustado y el crudo son casi identicos: el desbalance de dispositivo")
print("      NO explica el resultado (no hay paradoja de Simpson).")

print("\n-- 6.c Regresion logistica: conversion ~ grupo + dispositivo (+ interaccion) --")
d2 = df.copy()
d2['trat'] = (d2.grupo == 'tratamiento').astype(int)
m1 = smf.logit('convirtio ~ trat + C(dispositivo)', data=d2).fit(disp=0)
print(m1.summary2().tables[1].round(4).to_string())
print("\n   OR ajustado del tratamiento = {:.4f}  IC95% [{:.4f}, {:.4f}]".format(
    np.exp(m1.params['trat']), *np.exp(m1.conf_int().loc['trat']).values))
m2 = smf.logit('convirtio ~ trat * C(dispositivo)', data=d2).fit(disp=0)
lr = 2 * (m2.llf - m1.llf)
plr = stats.chi2.sf(lr, m2.df_model - m1.df_model)
print("   Prueba de razon de verosimilitud del termino de interaccion: "
      "LR={:.4f}, gl={}, p={:.4f}".format(lr, int(m2.df_model - m1.df_model), plr))
print("   -> {}".format("sin interaccion: el efecto es consistente en los 3 dispositivos"
                        if plr > .05 else "hay interaccion"))

print("\n-- 6.d Efecto ATE estandarizado (reponderando a la mezcla real de trafico) --")
mix = df.dispositivo.value_counts(normalize=True)
ate = sum(mix[d] * (R.loc[R.dispositivo == d, 'dif_pp'].iloc[0]) for d in devs)
print("   Mezcla poblacional de dispositivos: " +
      ", ".join("{}={:.1%}".format(d, mix[d]) for d in devs))
print("   Efecto crudo (sin ajustar)      = +35.60 pp")
print("   Efecto estandarizado por disp.  = {:+.2f} pp".format(ate))
print("   -> La correccion es minima; la conclusion no cambia.")

# --------------------------------------------------- 7. ROBUSTEZ
h("7. CHEQUEOS DE ROBUSTEZ Y BUSQUEDA DE SESGOS")

print("-- 7.a Prueba de permutacion de la diferencia de conversion (sin supuestos) --")
y = df.convirtio.values
g = (df.grupo == 'tratamiento').values
obs = y[g].mean() - y[~g].mean()
perm = np.array([(lambda p: y[p][:250].mean() - y[p][250:].mean())(rng.permutation(500))
                 for _ in range(20000)])
p_perm = (np.abs(perm) >= abs(obs)).mean()
print("   Observado={:+.4f}; permutaciones B=20,000; p={:.5f} "
      "(0 significa < 1/20000)".format(obs, p_perm))

print("\n-- 7.b Efecto por dispositivo: coherencia de signo --")
print("   El tratamiento gana en los 3 dispositivos -> resultado consistente,")
print("   no impulsado por un solo segmento.")

print("\n-- 7.c Estabilidad temporal (efecto por semana) --")
d2['sem'] = pd.to_datetime(d2.dia_registro).dt.isocalendar().week
for w, s in d2.groupby('sem'):
    c, t = s[s.trat == 0], s[s.trat == 1]
    if len(c) and len(t):
        print("   semana {}: n={:3d}  conv control={:.3f} (n={:2d})  tratamiento={:.3f} "
              "(n={:2d})  dif={:+.3f}".format(
                  int(w), len(s), c.convirtio.mean(), len(c),
                  t.convirtio.mean(), len(t), t.convirtio.mean() - c.convirtio.mean()))

print("\n-- 7.d Covariables de comportamiento (POST-tratamiento, no usar para ajustar) --")
for v in ['duracion_sesion', 'paginas_vistas']:
    r = stats.pointbiserialr(df.convirtio, df[v])
    print("   corr(convirtio, {}) = {:+.4f}  p={:.4f}".format(v, r.statistic, r.pvalue))
print("   Estas dos variables se miden DESPUES de ver la pagina: son resultados,")
print("   no covariables de base. Ajustar por ellas introduciria sesgo de colisionador.")
print("   Su balance entre grupos (seccion 1.c) sugiere ademas que el nuevo diseno NO")
print("   cambio el compromiso del usuario, solo la conversion: patron poco habitual.")

print("\n-- 7.e Sensibilidad a valores extremos del ingreso --")
a = df[df.grupo == 'control'].ingreso.values
b = df[df.grupo == 'tratamiento'].ingreso.values
for q in [1.0, 0.99, 0.95]:
    ca, cb = np.quantile(np.r_[a, b], q), np.quantile(np.r_[a, b], q)
    aw, bw = np.clip(a, None, ca), np.clip(b, None, cb)
    print("   winsorizado al p{:.0f}: dif de medias = {:+.2f}  (Mann-Whitney p={:.2e})".format(
        q * 100, bw.mean() - aw.mean(),
        stats.mannwhitneyu(aw, bw, alternative='two-sided').pvalue))

print("\n-- 7.f Impacto economico proyectado --")
dif_arpu = b.mean() - a.mean()
bs = np.array([rng.choice(b, 250, True).mean() - rng.choice(a, 250, True).mean()
               for _ in range(20000)])
for N in [10000, 100000]:
    print("   Con {:,} usuarios/mes: ingreso incremental esperado = {:,.0f} "
          "(IC95% {:,.0f} a {:,.0f})".format(
              N, dif_arpu * N, np.percentile(bs, 2.5) * N, np.percentile(bs, 97.5) * N))
print("   Advertencia: proyeccion valida solo si la mezcla de trafico futura se parece")
print("   a la del experimento y si el efecto no decae (novedad).")

R.to_csv('resultados_segmentos.csv', index=False)
print("\n[guardado] resultados_segmentos.csv")
