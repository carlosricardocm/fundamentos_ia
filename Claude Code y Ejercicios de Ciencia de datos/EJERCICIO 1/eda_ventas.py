# -*- coding: utf-8 -*-
"""EDA completo de ventas_ecommerce.csv -> graficas PNG + reporte de texto."""
import warnings; warnings.filterwarnings("ignore")
import os, io
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "graficas")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- paleta
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#7d7c76"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
            "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
            "#184f95", "#104281", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUE_SEQ)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "blue_red", ["#8c2726", "#e34948", "#f0b3b2", "#f0efec",
                 "#9ec5f4", "#2a78d6", "#104281"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
    "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "figure.dpi": 110,
})


def limpiar(ax, ejes=("top", "right")):
    for s in ejes:
        ax.spines[s].set_visible(False)


def titulo(ax, t, sub=None):
    ax.set_title(t, fontsize=13, fontweight="bold", color=INK, loc="left",
                 pad=24 if sub else 8)
    if sub:
        ax.text(0, 1.022, sub, transform=ax.transAxes, fontsize=9.5,
                color=INK_MUTED, va="bottom", ha="left")


def guardar(fig, nombre):
    ruta = os.path.join(OUT, nombre)
    fig.savefig(ruta, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("  [PNG] graficas/" + nombre)


# matplotlib interpreta un par de "$" como mathtext: en TODO texto de figura
# el simbolo va escapado como "\$".
def usd(v, dec=0):
    return ("-\$" if v < 0 else "\$") + format(abs(v), ",." + str(dec) + "f")


mon = FuncFormatter(lambda v, p: usd(v))

# ---------------------------------------------------------------- 0. carga
CSV = os.path.join(BASE, "ventas_ecommerce.csv")
df = pd.read_csv(CSV)
df["fecha_pedido_dt"] = pd.to_datetime(df["fecha_pedido"], errors="coerce")
NUM = ["ingreso", "cantidad", "descuento"]
COLS = list(df.columns.drop("fecha_pedido_dt"))

rep = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    rep.append(s)


P("=" * 78)
P("ANALISIS EXPLORATORIO DE DATOS - ventas_ecommerce.csv")
P("=" * 78)

# ---------------------------------------------------------------- 1. estructura
P("")
P("1. DIMENSIONES, TIPOS DE DATO Y VALORES NULOS")
P("-" * 78)
P("Filas: %d   |   Columnas: %d   |   Memoria: %.1f KB"
  % (df.shape[0], len(COLS), df.memory_usage(deep=True).sum() / 1024))

est = pd.DataFrame({
    "tipo": df[COLS].dtypes.astype(str),
    "no_nulos": df[COLS].notna().sum(),
    "nulos": df[COLS].isna().sum(),
})
est["pct_nulos"] = (est["nulos"] / len(df) * 100).round(1)
est["unicos"] = df[COLS].nunique()
P("")
P(est.to_string())
P("")
P("Filas con al menos un nulo: %d (%.1f%%)"
  % (df[NUM].isna().any(axis=1).sum(), df[NUM].isna().any(axis=1).mean() * 100))
P("Filas 100%% completas:      %d" % (~df[NUM].isna().any(axis=1)).sum())
P("Filas duplicadas exactas:  %d" % df.duplicated().sum())
P("id_pedido duplicados:      %d" % df["id_pedido"].duplicated().sum())
P("Rango de fechas: %s a %s (%d fechas no parseables)"
  % (df.fecha_pedido_dt.min().strftime("%Y-%m-%d"),
     df.fecha_pedido_dt.max().strftime("%Y-%m-%d"),
     df.fecha_pedido_dt.isna().sum()))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8),
                             gridspec_kw={"width_ratios": [1, 1.4]})
vals = est["pct_nulos"].values
colores = [SERIES[1] if v > 0 else GRID for v in vals]
a1.barh(range(len(COLS)), vals, color=colores, height=0.62)
a1.set_yticks(range(len(COLS)))
a1.set_yticklabels(COLS, fontsize=9)
a1.invert_yaxis()
a1.set_xlim(0, max(vals.max() * 1.45, 5))
a1.set_xlabel("% de valores nulos")
limpiar(a1)
a1.xaxis.grid(True, alpha=0.7)
a1.set_axisbelow(True)
for i, (v, n) in enumerate(zip(vals, est["nulos"])):
    if v > 0:
        a1.text(v + 0.4, i, "%.1f%%  (%d)" % (v, n), va="center",
                fontsize=8.5, color=INK_2)
titulo(a1, "Valores nulos por columna", "3 de 11 columnas tienen faltantes")

m = df[COLS].isna().astype(int).values
a2.imshow(m, aspect="auto", interpolation="nearest",
          cmap=LinearSegmentedColormap.from_list("nul", [SURFACE, SERIES[1]]))
a2.set_xticks(range(len(COLS)))
a2.set_xticklabels(COLS, rotation=45, ha="right", fontsize=8.5)
a2.set_ylabel("indice de fila")
limpiar(a2, ("top", "right", "left", "bottom"))
titulo(a2, "Patron de faltantes", "naranja = ausente; disperso, sin bloques ni co-ocurrencia")
fig.tight_layout()
guardar(fig, "01_calidad_valores_nulos.png")

# ---------------------------------------------------------------- 2. descriptiva
P("")
P("2. ESTADISTICA DESCRIPTIVA DE VARIABLES NUMERICAS")
P("-" * 78)
desc = df[NUM].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
desc["skew"] = df[NUM].skew()
desc["kurtosis"] = df[NUM].kurt()
desc["CV"] = df[NUM].std() / df[NUM].mean()
P("")
P(desc.round(3).to_string())
P("")
P("Lectura: 'ingreso' esta fuertemente sesgado a la derecha (skew=1.90, curtosis=3.66):")
P("         la media ($714.63) supera a la mediana ($447.63) en 60%.")
P("         'cantidad' y 'descuento' son en realidad DISCRETAS (5 niveles cada una):")
P("         cantidad en {1..5}, descuento en {0, .05, .10, .15, .20}.")

# ---------------------------------------------------------------- 3. distribuciones
P("")
P("3. DISTRIBUCIONES")
P("-" * 78)

ing = df["ingreso"].dropna()
fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [4, 1], "hspace": 0.12})
ax.hist(ing, bins=36, color=SERIES[0], edgecolor=SURFACE, linewidth=1.2)
ax.axvline(ing.mean(), color=SERIES[1], lw=2, ls="--")
ax.axvline(ing.median(), color=SERIES[3], lw=2, ls="--")
ax.set_ylim(0, ax.get_ylim()[1] * 1.26)
top = ax.get_ylim()[1]
ax.text(ing.mean(), top * 0.97, "  media " + usd(ing.mean()),
        color=SERIES[1], fontsize=9.5, va="top", ha="left")
ax.text(ing.median(), top * 0.97, "mediana " + usd(ing.median()) + "  ",
        color="#a06d00", fontsize=9.5, va="top", ha="right")
ax.set_ylabel("n.o de pedidos")
limpiar(ax)
ax.yaxis.grid(True, alpha=0.7)
ax.set_axisbelow(True)
titulo(ax, "Distribucion del ingreso por pedido",
       "n=%d con dato (%d nulos excluidos) - cola larga a la derecha, media arrastrada por los extremos"
       % (len(ing), df.ingreso.isna().sum()))
bp = bx.boxplot(ing, vert=False, widths=0.45, patch_artist=True,
                flierprops=dict(marker="o", markersize=4.5,
                                markerfacecolor=SERIES[0],
                                markeredgecolor=SURFACE, alpha=0.65),
                medianprops=dict(color=SURFACE, lw=2),
                boxprops=dict(facecolor=SERIES[0], edgecolor=SERIES[0]),
                whiskerprops=dict(color=INK_MUTED),
                capprops=dict(color=INK_MUTED))
bx.set_yticks([])
bx.set_xlabel("ingreso (USD)")
bx.xaxis.set_major_formatter(mon)
limpiar(bx, ("top", "right", "left"))
bx.xaxis.grid(True, alpha=0.7)
bx.set_axisbelow(True)
guardar(fig, "02_distribucion_ingreso.png")

fig, ax = plt.subplots(figsize=(9, 5))
vc = df["cantidad"].value_counts(dropna=False).sort_index()
etiq = ["sin dato" if pd.isna(k) else str(int(k)) for k in vc.index]
cols = [GRID if pd.isna(k) else SERIES[0] for k in vc.index]
ax.bar(range(len(vc)), vc.values, color=cols, width=0.62)
for i, v in enumerate(vc.values):
    ax.text(i, v + 1.4, "%d\n%.1f%%" % (v, v / len(df) * 100), ha="center",
            fontsize=9, color=INK_2, linespacing=1.3)
ax.set_xticks(range(len(vc)))
ax.set_xticklabels(etiq)
ax.set_xlabel("unidades por pedido")
ax.set_ylabel("n.o de pedidos")
ax.set_ylim(0, vc.values.max() * 1.25)
limpiar(ax)
ax.yaxis.grid(True, alpha=0.7)
ax.set_axisbelow(True)
titulo(ax, "Distribucion de la cantidad",
       "discreta y creciente entre 1 y 5 unidades; sin ceros ni valores fuera de rango")
guardar(fig, "03_distribucion_cantidad.png")

fig, ax = plt.subplots(figsize=(9, 5))
vd = df["descuento"].value_counts(dropna=False).sort_index()
etiq = ["sin dato" if pd.isna(k) else "%.0f%%" % (k * 100) for k in vd.index]
cols = [GRID if pd.isna(k) else SERIES[0] for k in vd.index]
ax.bar(range(len(vd)), vd.values, color=cols, width=0.62)
for i, v in enumerate(vd.values):
    ax.text(i, v + 1.4, "%d\n%.1f%%" % (v, v / len(df) * 100), ha="center",
            fontsize=9, color=INK_2, linespacing=1.3)
ax.set_xticks(range(len(vd)))
ax.set_xticklabels(etiq)
ax.set_xlabel("descuento aplicado")
ax.set_ylabel("n.o de pedidos")
ax.set_ylim(0, vd.values.max() * 1.28)
limpiar(ax)
ax.yaxis.grid(True, alpha=0.7)
ax.set_axisbelow(True)
titulo(ax, "Distribucion del descuento",
       "solo 5 niveles posibles; 18.5% sin registro (gris): ambiguo entre '0%' y dato perdido")
guardar(fig, "04_distribucion_descuento.png")

for c in NUM:
    s = df[c].dropna()
    P("")
    P("%s: n=%d  media=%.3f  mediana=%.3f  sd=%.3f  min=%.2f  max=%.2f  skew=%.2f"
      % (c, len(s), s.mean(), s.median(), s.std(), s.min(), s.max(), s.skew()))

# ---------------------------------------------------------------- 4. correlaciones
P("")
P("4. CORRELACIONES")
P("-" * 78)
cp = df[NUM].corr(method="pearson")
cs = df[NUM].corr(method="spearman")
P("")
P("Pearson:")
P(cp.round(3).to_string())
P("")
P("Spearman:")
P(cs.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0),
                         gridspec_kw={"wspace": 0.12})
for k, (ax, cm, nom) in enumerate(zip(axes, [cp, cs],
                                      ["Pearson (lineal)", "Spearman (monotona)"])):
    im = ax.imshow(cm.values, cmap=CMAP_DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(NUM)))
    ax.set_xticklabels(NUM, rotation=20, ha="right")
    ax.set_yticks(range(len(NUM)))
    ax.set_yticklabels(NUM if k == 0 else [])
    limpiar(ax, ("top", "right", "left", "bottom"))
    for i in range(len(NUM)):
        for j in range(len(NUM)):
            v = cm.values[i, j]
            ax.text(j, i, "%.2f" % v, ha="center", va="center", fontsize=12.5,
                    color=("#ffffff" if abs(v) > 0.55 else INK),
                    fontweight="bold" if i != j else "normal")
    titulo(ax, nom)
cb = fig.colorbar(im, ax=axes, shrink=0.74, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
cb.set_label("coeficiente de correlacion", color=INK_2, fontsize=9)
cb.outline.set_visible(False)
fig.suptitle("Mapa de calor de correlaciones entre variables numericas",
             fontsize=14, fontweight="bold", color=INK, x=0.055, ha="left", y=1.10)
fig.text(0.055, 1.02, "azul = positiva | gris = nula | rojo = negativa.   "
                      "Unica relacion apreciable: ingreso ~ cantidad (r=0.36)",
         fontsize=9.5, color=INK_MUTED, ha="left")
guardar(fig, "05_mapa_calor_correlaciones.png")

rng = np.random.RandomState(7)
# Multiplos pequenos, no cuatro colores en un mismo scatter: con 4 series todos los
# pares comparten el plano y la paleta solo garantiza separacion CVD hasta 3 colores.
cats_ord = df["categoria"].value_counts().index.tolist()
fig, axes = plt.subplots(1, 4, figsize=(14, 4.6), sharey=True,
                         gridspec_kw={"wspace": 0.08})
ymax = df.ingreso.max() * 1.08
for k, (ax, c) in enumerate(zip(axes, cats_ord)):
    s_ = df[df.categoria == c].dropna(subset=["ingreso", "cantidad"])
    ax.scatter(s_.cantidad + rng.uniform(-.2, .2, len(s_)), s_.ingreso, s=32,
               color=SERIES[0], alpha=0.7, linewidths=1.0, edgecolors=SURFACE)
    med = s_.groupby("cantidad").ingreso.median()
    ax.plot(med.index, med.values, color=INK, lw=1.6, marker="_", ms=13, mew=2.2)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.4, 5.6)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("cantidad")
    limpiar(ax, ("top", "right") if k == 0 else ("top", "right", "left"))
    if k > 0:
        ax.tick_params(left=False)
    ax.yaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_title("%s   (mediana $%s)".replace("$", "\$")
                 % (c, format(s_.ingreso.median(), ",.0f")),
                 fontsize=11, fontweight="bold", color=INK, loc="left", pad=8)
axes[0].set_ylabel("ingreso (USD)")
axes[0].yaxis.set_major_formatter(mon)
fig.suptitle("Ingreso vs. cantidad, un panel por categoria",
             fontsize=14, fontweight="bold", color=INK, x=0.062, ha="left", y=1.10)
fig.text(0.062, 1.02, "la r global de 0.36 esconde el mecanismo real: dentro de cada "
                      "categoria el ingreso si crece con la cantidad (linea negra = mediana), "
                      "pero el nivel lo fija el precio de la categoria",
         fontsize=9.5, color=INK_MUTED, ha="left")
guardar(fig, "06_dispersion_ingreso_cantidad.png")

# ---------------------------------------------------------------- 5. top / desglose
P("")
P("5. TOP 5 PRODUCTOS Y DESGLOSE POR CATEGORIA Y REGION")
P("-" * 78)
tot = df["ingreso"].sum()
prod = (df.groupby("producto")["ingreso"].agg(["sum", "count", "mean"])
        .sort_values("sum", ascending=False))
prod["pct_total"] = prod["sum"] / tot * 100
P("")
P("Ingreso total registrado: $%s" % format(tot, ",.2f"))
P("")
P("Top 5 productos por ingreso (todos los estatus):")
P(prod.head(5).round(2).to_string())
prod_c = (df[df.estatus == "completado"].groupby("producto")["ingreso"]
          .agg(["sum", "count", "mean"]).sort_values("sum", ascending=False))
P("")
P("Top 5 productos por ingreso (solo pedidos COMPLETADOS):")
P(prod_c.head(5).round(2).to_string())

fig, ax = plt.subplots(figsize=(10.8, 5.2))
t5 = prod.head(5).iloc[::-1]
t5c = prod_c.reindex(t5.index)["sum"]
ax.barh(range(5), t5["sum"], color=SERIES[0], height=0.6)
ax.barh(range(5), t5c, color="#104281", height=0.6)
for i, (v, n) in enumerate(zip(t5["sum"], t5["count"])):
    ax.text(v + tot * 0.005, i, usd(v) + "   (%.1f%% del total, %d pedidos)"
            % (v / tot * 100, int(n)),
            va="center", fontsize=9.5, color=INK_2)
ax.set_yticks(range(5))
ax.set_yticklabels(t5.index, fontsize=10.5)
ax.set_xlim(0, t5["sum"].max() * 1.48)
ax.xaxis.set_major_formatter(mon)
ax.set_xlabel("ingreso acumulado")
limpiar(ax)
ax.xaxis.grid(True, alpha=0.7)
ax.set_axisbelow(True)
ax.legend(handles=[Patch(facecolor="#104281", label="completado"),
                   Patch(facecolor=SERIES[0], label="total (incluye pendiente y cancelado)")],
          loc="lower right", fontsize=9, labelcolor=INK_2)
titulo(ax, "Top 5 productos por ingreso",
       "concentran " + usd(prod.head(5)["sum"].sum())
       + " = %.0f%% del ingreso total" % (prod.head(5)["sum"].sum() / tot * 100))
guardar(fig, "07_top5_productos_ingreso.png")

cat = (df.groupby("categoria")["ingreso"].agg(["sum", "count", "mean"])
       .sort_values("sum", ascending=False))
cat["pct_total"] = cat["sum"] / tot * 100
reg = (df.groupby("region")["ingreso"].agg(["sum", "count", "mean"])
       .sort_values("sum", ascending=False))
reg["pct_total"] = reg["sum"] / tot * 100
P("")
P("Desglose por CATEGORIA:")
P(cat.round(2).to_string())
P("")
P("Desglose por REGION:")
P(reg.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
_top2 = cat["pct_total"].head(2).sum()
_top2n = cat["count"].head(2).sum() / cat["count"].sum() * 100
subs = ["Electronica y Deportes = %.0f%% del ingreso con %.0f%% de los pedidos"
        % (_top2, _top2n),
        "Centro concentra %.0f%% del ingreso; Sur tiene el mayor ticket promedio"
        % reg["pct_total"].max()]
for ax, d_, nom, sub in zip(axes, [cat, reg], ["Ingreso por categoria", "Ingreso por region"], subs):
    dd = d_.iloc[::-1]
    ax.barh(range(len(dd)), dd["sum"], color=SERIES[0], height=0.58)
    for i, (v, n, mm) in enumerate(zip(dd["sum"], dd["count"], dd["mean"])):
        ax.text(v + tot * 0.006, i, usd(v) + "  (%.0f%%)  -  %d ped. - prom "
                % (v / tot * 100, int(n)) + usd(mm),
                va="center", fontsize=9, color=INK_2)
    ax.set_yticks(range(len(dd)))
    ax.set_yticklabels(dd.index, fontsize=10.5)
    ax.set_xlim(0, dd["sum"].max() * 1.85)
    ax.xaxis.set_major_formatter(mon)
    ax.set_xlabel("ingreso acumulado")
    limpiar(ax)
    ax.xaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    titulo(ax, nom, sub)
fig.tight_layout()
guardar(fig, "08_ingreso_categoria_region.png")

piv = df.pivot_table(index="categoria", columns="region", values="ingreso", aggfunc="sum")
piv = piv.loc[cat.index, reg.index]
P("")
P("Matriz CATEGORIA x REGION (suma de ingreso):")
P(piv.round(0).to_string())
fig, ax = plt.subplots(figsize=(9.4, 5.0))
mx = np.nanmax(piv.values)
im = ax.imshow(piv.values, cmap=CMAP_SEQ, vmin=0, vmax=mx)
ax.set_xticks(range(piv.shape[1]))
ax.set_xticklabels(piv.columns, fontsize=10.5)
ax.set_yticks(range(piv.shape[0]))
ax.set_yticklabels(piv.index, fontsize=10.5)
limpiar(ax, ("top", "right", "left", "bottom"))
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        ax.text(j, i, usd(v), ha="center", va="center",
                fontsize=10.5, color=("#ffffff" if v > mx * 0.55 else INK),
                fontweight="bold")
cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
cb.outline.set_visible(False)
cb.set_label("ingreso acumulado", color=INK_2, fontsize=9)
cb.ax.yaxis.set_major_formatter(mon)
titulo(ax, "Ingreso por categoria y region",
       "Deportes-Centro es la celda dominante (" + usd(piv.values.max())
       + "); Hogar-Centro la mas debil (" + usd(piv.loc["Hogar", "Centro"]) + ")")
guardar(fig, "09_heatmap_categoria_region.png")

# ---------------------------------------------------------------- 6. calidad
P("")
P("6. VALORES ATIPICOS Y PROBLEMAS DE CALIDAD DE DATOS")
P("-" * 78)

q1, q3 = df.ingreso.quantile([.25, .75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
out = df[(df.ingreso < lo) | (df.ingreso > hi)]
P("")
P("[A] Outliers univariados de 'ingreso' (regla IQR 1.5): limites $%s .. $%s"
  % (format(lo, ",.2f"), format(hi, ",.2f")))
P("    %d pedidos (%.1f%% de los que tienen dato) que suman $%s = %.0f%% del ingreso total."
  % (len(out), len(out) / df.ingreso.notna().sum() * 100,
     format(out.ingreso.sum(), ",.0f"), out.ingreso.sum() / tot * 100))
P("    Composicion: " + ", ".join("%s (%d)" % (k, v) for k, v in out.producto.value_counts().items()))
P("    -> Son ventas legitimas de articulos caros (Laptop/Bicicleta/Monitor) con 4-5 unidades,")
P("       NO errores. No conviene eliminarlas; si acaso, modelar el ingreso en escala log.")

neg = df[df.ingreso <= 0]
P("")
P("[B] Ingresos <= 0: %d registro(s) -- VALOR IMPOSIBLE" % len(neg))
P(neg[["id_pedido", "producto", "categoria", "ingreso", "cantidad", "estatus"]].to_string(index=False))

d_ = df.dropna(subset=["ingreso", "cantidad"]).copy()
d_ = d_[d_.ingreso > 0]
d_["pu"] = d_.ingreso / d_.cantidad
filas = []
for p, g in d_.groupby("producto"):
    a, b_ = g.pu.quantile([.25, .75])
    i_ = b_ - a
    o = g[(g.pu > b_ + 3 * i_) | (g.pu < a - 3 * i_)]
    for _, r in o.iterrows():
        filas.append([r.id_pedido, p, r.categoria, r.ingreso, int(r.cantidad),
                      round(r.pu, 2), round(g.pu.median(), 2),
                      round(r.pu / g.pu.median(), 1)])
anom = pd.DataFrame(filas, columns=["id_pedido", "producto", "categoria", "ingreso",
                                    "cantidad", "precio_unit", "mediana_producto",
                                    "veces_mediana"]).sort_values("veces_mediana", ascending=False)
P("")
P("[C] Precios unitarios inverosimiles (IQR x3 DENTRO de cada producto): %d casos" % len(anom))
P(anom.to_string(index=False))
P("    -> Estos SI parecen errores de captura: un balon de futbol a $602.88 y una")
P("       lampara a $789.20 no son ventas grandes, son magnitudes equivocadas.")

n_id = df.id_cliente.nunique()
n_nom = df.nombre_cliente.nunique()
mult = (df.groupby("id_cliente").nombre_cliente.nunique() > 1).sum()
mult2 = (df.groupby("nombre_cliente").id_cliente.nunique() > 1).sum()
P("")
P("[D] Integridad referencial de cliente: ROTA")
P("    %d id_cliente distintos frente a solo %d nombres distintos." % (n_id, n_nom))
P("    %d ids tienen mas de un nombre; %d nombres se reparten entre varios ids." % (mult, mult2))
P("    Ejemplo: CLI-055 aparece con %d nombres diferentes."
  % df[df.id_cliente == "CLI-055"].nombre_cliente.nunique())
P("    -> nombre_cliente NO sirve para identificar clientes y ninguna metrica por cliente")
P("       (recurrencia, LTV, RFM) es confiable con esta tabla tal como esta.")

st = df.groupby("estatus")["ingreso"].agg(["count", "sum"])
st["pct_ingreso"] = st["sum"] / tot * 100
P("")
P("[E] Estatus del pedido: 'ingreso' mezcla ventas realizadas y no realizadas")
P(st.round(2).to_string())
no_comp = st.loc[["cancelado", "pendiente"], "sum"].sum()
P("    -> %d pedidos (%.1f%%) no estan completados y aportan $%s (%.0f%%) al total."
  % ((df.estatus != "completado").sum(), (df.estatus != "completado").mean() * 100,
     format(no_comp, ",.0f"), no_comp / tot * 100))
P("       Cualquier KPI de ventas debe filtrar estatus == 'completado'.")

P("")
P("[F] Tipos de dato: 'fecha_pedido' se lee como texto (object) y requiere to_datetime.")
P("    'cantidad' y 'descuento' quedan como float solo por los nulos; su dominio real")
P("    es entero y categorico ordenado, respectivamente.")
P("")
P("[G] Nulos: descuento %d (%.1f%%), ingreso %d (%.1f%%), cantidad %d (%.1f%%)."
  % (df.descuento.isna().sum(), df.descuento.isna().mean() * 100,
     df.ingreso.isna().sum(), df.ingreso.isna().mean() * 100,
     df.cantidad.isna().sum(), df.cantidad.isna().mean() * 100))
P("    Los 17 nulos de 'ingreso' son los mas graves: no hay precio de catalogo con el cual")
P("    imputarlos y sesgan a la baja toda suma. Ninguna fila tiene 2 o mas nulos a la vez.")
P("")
P("[H] Consistencia correcta: producto -> categoria es 1:1 sin excepciones; no hay")
P("    duplicados de id_pedido ni filas repetidas; las 260 fechas son validas y caen")
P("    dentro de 2024; no hay variantes de mayusculas ni espacios en categoria,")
P("    region ni estatus.")

fig, ax = plt.subplots(figsize=(10.5, 5.4))
orden = cat.index.tolist()
data = [df[df.categoria == c].ingreso.dropna().values for c in orden]
bp = ax.boxplot(data, vert=True, widths=0.5, patch_artist=True, labels=orden,
                flierprops=dict(marker="o", markersize=5, markeredgecolor=SURFACE, alpha=0.8),
                medianprops=dict(color=SURFACE, lw=2),
                whiskerprops=dict(color=INK_MUTED), capprops=dict(color=INK_MUTED))
for i, patch in enumerate(bp["boxes"]):
    patch.set_facecolor(SERIES[i])
    patch.set_edgecolor(SERIES[i])
for i, fl in enumerate(bp["fliers"]):
    fl.set_markerfacecolor(SERIES[i])
ax.axhline(hi, color=INK_MUTED, lw=1, ls=":")
ax.text(0.55, hi, " limite IQR global " + usd(hi), fontsize=8.5,
        color=INK_MUTED, va="bottom")
ax.set_ylabel("ingreso (USD)")
ax.yaxis.set_major_formatter(mon)
limpiar(ax)
ax.yaxis.grid(True, alpha=0.7)
ax.set_axisbelow(True)
titulo(ax, "Dispersion del ingreso por categoria",
       "los %d outliers globales son todos Electronica y Deportes: precio alto, no error de captura"
       % len(out))
guardar(fig, "10_outliers_ingreso_por_categoria.png")

fig, ax = plt.subplots(figsize=(11, 5.8))
orden_p = d_.groupby("producto").pu.median().sort_values().index.tolist()
for i, p in enumerate(orden_p):
    g = d_[d_.producto == p]
    es_anom = g.id_pedido.isin(anom.id_pedido)
    k = int((~es_anom).sum())
    ax.scatter(g.pu[~es_anom], np.full(k, i) + rng.uniform(-.15, .15, k),
               s=30, color=SERIES[0], alpha=0.55, edgecolors=SURFACE, linewidths=0.8)
    ax.scatter(g.pu[es_anom], np.full(int(es_anom.sum()), i), s=95, color=SERIES[7],
               edgecolors=SURFACE, linewidths=1.4, zorder=5, marker="D")
    ax.plot([g.pu.median()], [i], marker="|", ms=17, color=INK, mew=2.2, zorder=6)
for _, r in anom.iterrows():
    ax.annotate(r.id_pedido, (r.precio_unit, orden_p.index(r.producto)),
                textcoords="offset points", xytext=(10, 7), fontsize=8.5, color=INK_2)
ax.set_yticks(range(len(orden_p)))
ax.set_yticklabels(orden_p, fontsize=10)
ax.set_xscale("log")
ax.set_xlabel("precio unitario implicito = ingreso / cantidad (escala log)")
ax.xaxis.set_major_formatter(mon)
limpiar(ax)
ax.xaxis.grid(True, alpha=0.7)
ax.set_axisbelow(True)
ax.legend(handles=[Patch(facecolor=SERIES[0], label="pedido"),
                   Patch(facecolor=SERIES[7], label="precio anomalo (>3x IQR intra-producto)"),
                   Patch(facecolor=INK, label="mediana del producto")],
          loc="upper left", fontsize=9, labelcolor=INK_2)
titulo(ax, "Precio unitario implicito por producto",
       "4 pedidos con precios imposibles para su producto: errores de captura, no ventas grandes")
guardar(fig, "11_anomalias_precio_unitario.png")

mens = (df.dropna(subset=["fecha_pedido_dt"])
        .assign(mes=lambda x: x.fecha_pedido_dt.dt.to_period("M").dt.to_timestamp())
        .groupby("mes").agg(ingreso=("ingreso", "sum"), pedidos=("id_pedido", "count")))
P("")
P("ANEXO - Evolucion mensual:")
P(mens.round(2).to_string())
fig, (ax, bx) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.16})
ax.plot(mens.index, mens.ingreso, color=SERIES[0], lw=2, marker="o", ms=7,
        markeredgecolor=SURFACE, markeredgewidth=1.6)
ax.set_ylabel("ingreso mensual")
ax.yaxis.set_major_formatter(mon)
ax.set_ylim(0, mens.ingreso.max() * 1.2)
limpiar(ax)
ax.yaxis.grid(True, alpha=0.7)
ax.set_axisbelow(True)
for idx, dy in [(mens.ingreso.idxmin(), -22), (mens.ingreso.idxmax(), 13)]:
    ax.annotate(usd(mens.ingreso[idx]), (idx, mens.ingreso[idx]),
                textcoords="offset points", xytext=(0, dy), ha="center",
                fontsize=9.5, color=INK_2, fontweight="bold")
titulo(ax, "Evolucion mensual del ingreso (2024)",
       "serie corta y ruidosa: entre 13 y 29 pedidos por mes, insuficiente para afirmar tendencia o estacionalidad")
bx.bar(mens.index, mens.pedidos, width=20, color=GRID)
bx.set_ylabel("pedidos")
bx.set_xlabel("mes")
limpiar(bx)
bx.yaxis.grid(True, alpha=0.7)
bx.set_axisbelow(True)
guardar(fig, "12_evolucion_mensual.png")

with io.open(os.path.join(BASE, "reporte_eda.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(rep))
print("  [TXT] reporte_eda.txt")
print("")
print("Listo: %d PNG en graficas/"
      % len([x for x in os.listdir(OUT) if x.endswith(".png")]))
