# -*- coding: utf-8 -*-
"""Analisis de la prueba A/B de landing page. Salida: reporte en consola."""
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.proportion import (proportions_ztest, proportion_confint,
                                          confint_proportions_2indep,
                                          proportion_effectsize)
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.contingency_tables import Table2x2

rng = np.random.default_rng(20260828)
df = pd.read_csv('ab_test_landing.csv')
C = df[df.grupo == 'control']
T = df[df.grupo == 'tratamiento']


def h(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


# ---------------------------------------------------------------- 1. INTEGRIDAD
h("1. INTEGRIDAD Y MONTAJE DEL EXPERIMENTO")
print("Filas: {} | usuarios unicos: {} | nulos: {} | filas duplicadas: {}".format(
    len(df), df.id_usuario.nunique(), int(df.isna().sum().sum()), int(df.duplicated().sum())))
print("Ventana: {} a {} ({} dias)".format(
    df.dia_registro.min(), df.dia_registro.max(), df.dia_registro.nunique()))
print("Coherencia convirtio<->ingreso: convirtio=1 con ingreso=0 -> {}; "
      "convirtio=0 con ingreso>0 -> {}".format(
          int(((df.convirtio == 1) & (df.ingreso <= 0)).sum()),
          int(((df.convirtio == 0) & (df.ingreso > 0)).sum())))

print("\n-- 1.a Sample Ratio Mismatch (SRM) del reparto de grupos --")
n_c, n_t = len(C), len(T)
srm = stats.chisquare([n_c, n_t], f_exp=[(n_c + n_t) / 2] * 2)
print("control={}  tratamiento={}  (esperado 50/50)".format(n_c, n_t))
print("Chi2 bondad de ajuste: X2={:.4f}, gl=1, p={:.4f}".format(srm.statistic, srm.pvalue))
print("-> " + ("SIN SRM: el reparto global es exactamente 50/50."
               if srm.pvalue > .01 else "SRM DETECTADO."))

print("\n-- 1.b Distribucion por dispositivo (covariable pre-tratamiento) --")
ct = pd.crosstab(df.grupo, df.dispositivo)
print(ct.to_string())
print("\nPorcentaje dentro de cada grupo:")
print((ct.T / ct.sum(1)).T.mul(100).round(1).to_string())
chi2, p_dev, gl, esp = stats.chi2_contingency(ct)
n = ct.values.sum()
cramer_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
print("\nChi2 de independencia grupo x dispositivo: X2={:.4f}, gl={}, p={:.6f}".format(
    chi2, gl, p_dev))
print("Frecuencia esperada minima: {:.1f} (>5 -> supuesto del chi2 OK)".format(esp.min()))
print("V de Cramer = {:.4f}".format(cramer_v))
print("-> " + ("DESBALANCE SIGNIFICATIVO: la aleatorizacion NO produjo grupos comparables "
               "en dispositivo. Es un confusor potencial."
               if p_dev < .05 else "Balanceado."))

print("\n-- 1.c Balance de covariables numericas --")
for v in ['duracion_sesion', 'paginas_vistas']:
    a, b = C[v].values, T[v].values
    lev = stats.levene(a, b, center='median')
    tt = stats.ttest_ind(a, b, equal_var=False)
    mw = stats.mannwhitneyu(a, b, alternative='two-sided')
    sp = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
                 / (len(a) + len(b) - 2))
    d = (b.mean() - a.mean()) / sp
    print("\n  {}: control mu={:.2f} (DE {:.2f}) | tratamiento mu={:.2f} (DE {:.2f})".format(
        v, a.mean(), a.std(ddof=1), b.mean(), b.std(ddof=1)))
    print("    Levene p={:.4f} | Welch t={:.3f} p={:.4f} | Mann-Whitney p={:.4f} | "
          "d de Cohen={:.3f}".format(lev.pvalue, tt.statistic, tt.pvalue, mw.pvalue, d))
    print("    -> " + ("balanceada" if tt.pvalue > .05 else "DESBALANCEADA"))

print("\n-- 1.d Reparto por dia (sesgo temporal) --")
byday = pd.crosstab(df.dia_registro, df.grupo)
prop_t = byday.tratamiento / byday.sum(1)
print("Proporcion diaria de tratamiento: min={:.2f} max={:.2f} media={:.2f}".format(
    prop_t.min(), prop_t.max(), prop_t.mean()))
srm_day = stats.chisquare(byday.tratamiento.values, f_exp=byday.sum(1).values / 2)
print("Chi2 de SRM diario: X2={:.3f}, gl={}, p={:.4f}".format(
    srm_day.statistic, len(byday) - 1, srm_day.pvalue))
print("-> " + ("sin evidencia de sesgo temporal en la asignacion"
               if srm_day.pvalue > .05 else "posible sesgo temporal"))

# ------------------------------------------------------- 2. POTENCIA Y MDE
h("2. POTENCIA Y EFECTO MINIMO DETECTABLE (MDE) - CONVERSION")
p_c = C.convirtio.mean()
p_t = T.convirtio.mean()
alpha, power_obj = .05, .80
pw = NormalIndPower()

hh = pw.solve_power(effect_size=None, nobs1=n_c, alpha=alpha, power=power_obj,
                    ratio=n_t / n_c, alternative='two-sided')
p_mde_up = np.sin(np.arcsin(np.sqrt(p_c)) + hh / 2) ** 2
p_mde_dn = np.sin(np.arcsin(np.sqrt(p_c)) - hh / 2) ** 2
print("Tasa base observada (control) = {:.4f}".format(p_c))
print("Con n={}+{}, alpha={} bilateral y potencia objetivo={:.0%}:".format(
    n_c, n_t, alpha, power_obj))
print("  h de Cohen minimo detectable = {:.4f}".format(hh))
print("  MDE absoluto al alza   = {:+.2f} pp  (detecta tasas >= {:.4f})".format(
    (p_mde_up - p_c) * 100, p_mde_up))
print("  MDE absoluto a la baja = {:+.2f} pp  (detecta tasas <= {:.4f})".format(
    (p_mde_dn - p_c) * 100, p_mde_dn))
print("  MDE relativo al alza   = {:+.1f}%".format((p_mde_up / p_c - 1) * 100))

h_obs = proportion_effectsize(p_t, p_c)
power_obs = pw.solve_power(effect_size=h_obs, nobs1=n_c, alpha=alpha, ratio=n_t / n_c,
                           alternative='two-sided')
print("\nPotencia post-hoc con el efecto observado (h={:.4f}): {:.4f}".format(h_obs, power_obs))
print("  Nota: la potencia post-hoc calculada con el efecto observado es una funcion")
print("  monotona del valor p y NO aporta informacion nueva. La cifra que si informa")
print("  el diseno es el MDE ex-ante de arriba, junto con el IC del efecto.")

print("\nTamano de muestra necesario por grupo (alpha=.05, potencia=.80, base {:.3f}):".format(p_c))
for target in [0.01, 0.02, 0.05, 0.10]:
    hh2 = proportion_effectsize(p_c + target, p_c)
    nn = pw.solve_power(effect_size=hh2, nobs1=None, alpha=alpha, power=.80, ratio=1.,
                        alternative='two-sided')
    print("  para detectar {:+.0f} pp -> {:,.0f} por grupo".format(target * 100, np.ceil(nn)))

# ------------------------------------------------------------ 3. CONVERSION
h("3. PRUEBA PRINCIPAL - TASA DE CONVERSION")
x_c, x_t = int(C.convirtio.sum()), int(T.convirtio.sum())
tab = np.array([[x_t, n_t - x_t], [x_c, n_c - x_c]])
_, _, _, esp2 = stats.chi2_contingency(tab)
print("Supuestos:")
print("  (a) independencia: 1 fila = 1 usuario unico, sin duplicados -> OK")
print("  (b) esperados >=5: minimo = {:.1f} -> aproximacion normal valida".format(esp2.min()))
print("  (c) asignacion aleatoria -> CUESTIONADA (ver 1.b)")
ci_c = proportion_confint(x_c, n_c, method='wilson')
ci_t = proportion_confint(x_t, n_t, method='wilson')
print("\ncontrol:     {}/{} = {:.4f}  IC95% Wilson [{:.4f}, {:.4f}]".format(
    x_c, n_c, p_c, ci_c[0], ci_c[1]))
print("tratamiento: {}/{} = {:.4f}  IC95% Wilson [{:.4f}, {:.4f}]".format(
    x_t, n_t, p_t, ci_t[0], ci_t[1]))
z, pz = proportions_ztest([x_t, x_c], [n_t, n_c])
chi2c, pc_, _, _ = stats.chi2_contingency(tab)
orr, pf = stats.fisher_exact(tab)
lo, hi = confint_proportions_2indep(x_t, n_t, x_c, n_c, method='wald', compare='diff')
print("\nz de dos proporciones : z={:.4f}  p={:.3e}".format(z, pz))
print("Chi2 (correc. Yates)  : X2={:.4f}  p={:.3e}".format(chi2c, pc_))
print("Fisher exacto         : OR={:.4f}  p={:.3e}".format(orr, pf))
t22 = Table2x2(tab)
print("\nDiferencia absoluta   : {:+.2f} pp   IC95% [{:+.2f}, {:+.2f}] pp".format(
    (p_t - p_c) * 100, lo * 100, hi * 100))
print("Lift relativo         : {:+.1f}%".format((p_t / p_c - 1) * 100))
print("Riesgo relativo (RR)  : {:.4f}  IC95% [{:.4f}, {:.4f}]".format(
    t22.riskratio, *t22.riskratio_confint()))
print("Odds ratio (OR)       : {:.4f}  IC95% [{:.4f}, {:.4f}]".format(
    t22.oddsratio, *t22.oddsratio_confint()))
print("h de Cohen            : {:.4f} (>0.8 = efecto grande)".format(h_obs))
print("NNT (usuarios expuestos por conversion extra): {:.2f}".format(1 / (p_t - p_c)))

# ------------------------------------------------- 4. INGRESO POR USUARIO
h("4. INGRESO POR USUARIO (ARPU) - PRUEBA PARA DATOS NO NORMALES")
a, b = C.ingreso.values, T.ingreso.values
print("-- Verificacion de supuestos --")
for nm, v in [('control', a), ('tratamiento', b)]:
    sh = stats.shapiro(v)
    print("  {:12s} n={} media={:.2f} mediana={:.2f} DE={:.2f} asimetria={:.3f} "
          "curtosis={:.3f} ceros={:.1f}%".format(
              nm, len(v), v.mean(), np.median(v), v.std(ddof=1),
              stats.skew(v), stats.kurtosis(v), 100 * (v == 0).mean()))
    print("               Shapiro-Wilk W={:.4f} p={:.3e} -> {}".format(
        sh.statistic, sh.pvalue, 'NO normal' if sh.pvalue < .05 else 'normal'))
lev = stats.levene(a, b, center='median')
print("  Levene (homocedasticidad): W={:.4f} p={:.3e} -> {}".format(
    lev.statistic, lev.pvalue, 'varianzas DESIGUALES' if lev.pvalue < .05 else 'iguales'))
print("  -> Se descarta el t de Student clasico. Distribucion con masa en cero y cola")
print("     derecha larga. Prueba principal: Mann-Whitney U (no parametrica).")

mw = stats.mannwhitneyu(a, b, alternative='two-sided')
U = mw.statistic
auc = U / (len(a) * len(b))
auc_t = 1 - auc
cliff = 2 * auc_t - 1
print("\nMann-Whitney U={:.1f}  p={:.3e}".format(U, mw.pvalue))
print("  Tamano de efecto: P(ingreso_tratamiento > ingreso_control) = {:.4f}".format(auc_t))
print("  Delta de Cliff = {:+.4f}  (|d|>0.474 = efecto grande)".format(cliff))
ks = stats.ks_2samp(a, b)
print("Kolmogorov-Smirnov 2 muestras: D={:.4f} p={:.3e}".format(ks.statistic, ks.pvalue))
bm = stats.brunnermunzel(a, b)
print("Brunner-Munzel (robusto a varianzas/formas desiguales): W={:.4f} p={:.3e}".format(
    bm.statistic, bm.pvalue))
print("Nota: Mann-Whitney contrasta dominancia estocastica, no medias. Como las formas")
print("      difieren, se complementa con estimadores de la media (abajo).")

diffs = (b[:, None] - a[None, :]).ravel()
hl = np.median(diffs)
print("\nEstimador Hodges-Lehmann (mediana de las diferencias) = {:+.2f}".format(hl))

B = 20000
bs = np.array([rng.choice(b, len(b), True).mean() - rng.choice(a, len(a), True).mean()
               for _ in range(B)])
lo_b, hi_b = np.percentile(bs, [2.5, 97.5])
print("Diferencia de MEDIAS (ARPU): {:+.2f}  IC95% bootstrap [{:+.2f}, {:+.2f}] (B={:,})".format(
    b.mean() - a.mean(), lo_b, hi_b, B))
print("  ARPU control={:.2f} | ARPU tratamiento={:.2f} | lift={:+.1f}%".format(
    a.mean(), b.mean(), 100 * (b.mean() / a.mean() - 1)))
wt = stats.ttest_ind(b, a, equal_var=False)
print("Welch t (solo de referencia, supuestos violados): t={:.3f} p={:.3e}".format(
    wt.statistic, wt.pvalue))

print("\n-- Descomposicion: conversion vs ticket promedio --")
ac, at = a[a > 0], b[b > 0]
mwc = stats.mannwhitneyu(ac, at, alternative='two-sided')
print("  Ticket entre CONVERSORES: control mediana={:.2f} (n={}) | "
      "tratamiento mediana={:.2f} (n={})".format(
          np.median(ac), len(ac), np.median(at), len(at)))
print("  Mann-Whitney sobre conversores: U={:.1f} p={:.4f} -> {}".format(
    mwc.statistic, mwc.pvalue, 'difiere' if mwc.pvalue < .05 else 'NO difiere'))
bs2 = np.array([rng.choice(at, len(at), True).mean() - rng.choice(ac, len(ac), True).mean()
                for _ in range(B)])
print("  Diferencia de ticket medio: {:+.2f}  IC95% [{:+.2f}, {:+.2f}]".format(
    at.mean() - ac.mean(), np.percentile(bs2, 2.5), np.percentile(bs2, 97.5)))
print("  -> El ARPU se mueve porque cambia la CONVERSION" +
      (", no el ticket promedio." if mwc.pvalue > .05 else " y tambien el ticket."))
