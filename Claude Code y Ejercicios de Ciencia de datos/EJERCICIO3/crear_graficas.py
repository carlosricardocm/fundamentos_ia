# -*- coding: utf-8 -*-
"""
Genera las 4 visualizaciones de ingresos 2024 como PNG listos para presentación.

Paleta categórica validada con el validador de accesibilidad del sistema de dataviz:
separación CVD ΔE 9.1 (>= 8) y piso de visión normal ΔE 22.9 (>= 15) sobre la
superficie #fcfcfb. Los tonos con contraste < 3:1 llevan etiqueta directa visible.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------- paleta
SURFACE   = "#fcfcfb"   # superficie del gráfico
INK       = "#0b0b0b"   # tinta primaria
INK_2     = "#52514e"   # tinta secundaria
INK_MUTED = "#898781"   # ejes y etiquetas
GRID      = "#e1e0d9"   # rejilla hairline
AXIS      = "#c3c2b7"   # línea base
PLANE     = "#f9f9f7"   # plano de página (tarjetas KPI)

# Orden fijo: el color sigue a la entidad (región), nunca a su posición en el ranking.
REGIONES = ["Norte", "Centro", "Sur", "Occidente"]
SERIES = {
    "Norte":     "#2a78d6",   # azul
    "Centro":    "#eb6834",   # naranja
    "Sur":       "#1baf7a",   # aqua
    "Occidente": "#eda100",   # amarillo
}

# Rampa secuencial de un solo tono (azul 100 -> 700) para el mapa de calor.
AZULES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
          "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("azul_seq", AZULES)

MESES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
         "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
ABREV = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
         "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "text.parse_math": False,          # "$" es moneda, no matemáticas
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "axes.grid": False,
    "figure.dpi": 110,
})

DPI = 200
FIGSIZE = (12.4, 6.6)


def money(v, decimals=0):
    return "$" + format(round(v, decimals), ",." + str(decimals) + "f")


def miles(v, _=None):
    return "$" + format(v / 1000, ",.0f") + "k"


def limpiar(ax, y_grid=True):
    """Cromo recesivo: sin marco, rejilla hairline solo sobre el eje de valor."""
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(length=0, pad=7)
    if y_grid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=1.0)


def titulo(fig, texto, subtitulo):
    fig.text(0.035, 0.945, texto, fontsize=21, fontweight="600", color=INK, va="top")
    fig.text(0.035, 0.878, subtitulo, fontsize=12, color=INK_2, va="top")


def pie(fig, texto):
    fig.text(0.035, 0.028, texto, fontsize=9.5, color=INK_MUTED, va="bottom")


def radio_px(ax, px=5.0):
    """Convierte un radio en píxeles a unidades de dato en cada eje por separado."""
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform((0, 0)), inv.transform((px, px))
    return abs(x1 - x0), abs(y1 - y0)


def barra_redondeada(ax, x, alto, ancho, color, rx, ry):
    """Barra vertical con extremo de dato redondeado y base anclada a la línea cero."""
    rx, ry = min(rx, ancho / 2), min(ry, abs(alto) / 2)
    x0, x1, y0, y1 = x - ancho / 2, x + ancho / 2, 0.0, alto
    verts = [(x0, y0), (x0, y1 - ry), (x0, y1), (x0 + rx, y1),
             (x1 - rx, y1), (x1, y1), (x1, y1 - ry), (x1, y0), (x0, y0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color,
                           edgecolor=SURFACE, linewidth=1.6, zorder=3))


# ---------------------------------------------------------------- datos
df = pd.read_csv("ingresos_mensuales.csv")

por_mes = df.groupby("mes", as_index=False)["ingreso"].sum().sort_values("mes")
por_region = df.groupby("region")["ingreso"].sum().reindex(REGIONES)
tabla = (df.pivot_table(index="nombre_mes", columns="region", values="ingreso")
           .reindex(index=MESES, columns=REGIONES))

total = df["ingreso"].sum()
ing = por_mes["ingreso"].values
crecimiento = (ing[-1] / ing[0] - 1) * 100
h1, h2 = ing[:6].sum(), ing[6:].sum()

# ================================================================ 1. LÍNEAS
fig, ax = plt.subplots(figsize=FIGSIZE)
fig.subplots_adjust(left=0.085, right=0.965, top=0.775, bottom=0.135)

x = np.arange(12)
ax.plot(x, ing, color=SERIES["Norte"], linewidth=2.0, zorder=3,
        marker="o", markersize=8, markerfacecolor=SERIES["Norte"],
        markeredgecolor=SURFACE, markeredgewidth=2)

# Etiquetas directas selectivas: inicio, mínimo y máximo (nunca un número por punto).
i_max, i_min = int(np.argmax(ing)), int(np.argmin(ing))
for i, dy, va, ha in ((0, 22, "bottom", "left"), (i_min, -26, "top", "center"),
                      (i_max, 22, "bottom", "right")):
    ax.annotate(money(ing[i]), (x[i], ing[i]), textcoords="offset points",
                xytext=(0, dy), ha=ha, va=va, fontsize=11,
                fontweight="600", color=INK)

ax.set_xticks(x)
ax.set_xticklabels(ABREV)
ax.set_xlim(-0.55, 11.55)
ax.set_ylim(ing.min() * 0.86, ing.max() * 1.10)
ax.yaxis.set_major_formatter(FuncFormatter(miles))
ax.set_xlabel("Mes (2024)", labelpad=10)
ax.set_ylabel("Ingreso mensual (USD)", labelpad=10)
limpiar(ax)
titulo(fig, "El ingreso mensual crece de forma sostenida durante 2024",
       "Ingreso total consolidado de las cuatro regiones, en dólares")
pie(fig, "Fuente: ingresos_mensuales.csv · 48 registros (12 meses × 4 regiones)")
fig.savefig("01_ingreso_por_mes.png", dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)

# ================================================================ 2. BARRAS
fig, ax = plt.subplots(figsize=FIGSIZE)
fig.subplots_adjust(left=0.085, right=0.965, top=0.775, bottom=0.135)

orden = por_region.sort_values(ascending=False)
tope = orden.max()
ax.set_xlim(-0.62, len(orden) - 0.38)
ax.set_ylim(0, tope * 1.24)          # límites antes de dibujar: el radio se mide en px
rx, ry = radio_px(ax, 5.0)

for i, (reg, val) in enumerate(orden.items()):
    barra_redondeada(ax, i, val, 0.56, SERIES[reg], rx, ry)
    ax.text(i, val + tope * 0.020, format(val / total * 100, ".1f") + "% del total",
            ha="center", va="bottom", fontsize=10.5, color=INK_2)
    ax.text(i, val + tope * 0.078, money(val), ha="center", va="bottom",
            fontsize=12.5, fontweight="600", color=INK)

ax.set_xticks(np.arange(len(orden)))
ax.set_xticklabels(list(orden.index), fontsize=12, color=INK_2)
ax.yaxis.set_major_formatter(FuncFormatter(miles))
ax.set_ylabel("Ingreso acumulado del año (USD)", labelpad=10)
ax.set_xlabel("Región", labelpad=10)
limpiar(ax)
titulo(fig, "Centro concentra más de un tercio del ingreso anual",
       "Ingreso total acumulado por región, de enero a diciembre de 2024")
pie(fig, "Fuente: ingresos_mensuales.csv · total anual " + money(total))
fig.savefig("02_ingreso_por_region.png", dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)

# ================================================================ 3. MAPA DE CALOR
fig, ax = plt.subplots(figsize=(11.6, 7.8))
fig.subplots_adjust(left=0.135, right=0.87, top=0.775, bottom=0.055)

M = tabla.values
vmin, vmax = M.min() * 0.92, M.max()
im = ax.imshow(M, cmap=CMAP, aspect="auto", vmin=vmin, vmax=vmax)

# Separador del color de la superficie entre celdas (2 px).
ax.set_xticks(np.arange(-0.5, len(REGIONES), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(MESES), 1), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2.4)
ax.tick_params(which="minor", length=0)

umbral = vmin + (vmax - vmin) * 0.58
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, format(M[i, j] / 1000, ",.0f") + "k", ha="center", va="center",
                fontsize=10.5, fontweight="600",
                color="#ffffff" if M[i, j] > umbral else INK)

ax.set_xticks(range(len(REGIONES)))
ax.set_xticklabels(REGIONES, fontsize=12, color=INK_2)
ax.set_yticks(range(len(MESES)))
ax.set_yticklabels(MESES, fontsize=11, color=INK_2)
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")
ax.tick_params(length=0, pad=10)
for lado in ax.spines:
    ax.spines[lado].set_visible(False)

cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.035)
cb.set_label("Ingreso mensual (USD)", color=INK_2, fontsize=11, labelpad=12)
cb.ax.tick_params(length=0, colors=INK_MUTED, labelsize=10)
cb.outline.set_visible(False)
cb.ax.yaxis.set_major_formatter(FuncFormatter(miles))

titulo(fig, "El calor se desplaza al último trimestre en las cuatro regiones",
       "Ingreso mensual por región: el tono más oscuro indica mayor ingreso")
pie(fig, "Fuente: ingresos_mensuales.csv · valores en miles de dólares")
fig.savefig("03_mapa_calor_mes_region.png", dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)

# ================================================================ 4. RESUMEN
fig = plt.figure(figsize=(12.4, 6.9))
fig.patch.set_facecolor(SURFACE)

kpis = [
    ("Ingreso total 2024", money(total), "Suma de las cuatro regiones"),
    ("Crecimiento Dic vs Ene", "+" + format(crecimiento, ".1f") + "%",
     "De " + miles(ing[0]) + " a " + miles(ing[-1])),
    ("Región líder", "Centro",
     format(por_region["Centro"] / total * 100, ".1f") + "% del ingreso anual"),
    ("Ticket promedio", money(total / df.unidades_vendidas.sum(), 2),
     "Estable en las cuatro regiones"),
    ("Unidades vendidas", format(int(df.unidades_vendidas.sum()), ","),
     "Promedio " + format(df.unidades_vendidas.sum() / 12, ",.0f") + " por mes"),
    ("Clientes nuevos", format(int(df.clientes_nuevos.sum()), ","),
     "Promedio " + format(df.clientes_nuevos.sum() / 12, ",.0f") + " por mes"),
]
acentos = [SERIES["Norte"], SERIES["Sur"], SERIES["Centro"],
           SERIES["Occidente"], SERIES["Norte"], SERIES["Sur"]]

izq, der, arriba = 0.045, 0.955, 0.775
ancho_tot, alto_t, gap = der - izq, 0.145, 0.028
ancho_t = (ancho_tot - 2 * gap) / 3

for k, ((rot, val, sub), color) in enumerate(zip(kpis, acentos)):
    fila, col = divmod(k, 3)
    x0 = izq + col * (ancho_t + gap)
    y0 = arriba - fila * (alto_t + gap) - alto_t
    fig.patches.append(plt.Rectangle((x0, y0), ancho_t, alto_t, transform=fig.transFigure,
                                     facecolor=PLANE, edgecolor=GRID, linewidth=1.0, zorder=1))
    fig.patches.append(plt.Rectangle((x0, y0), 0.0045, alto_t, transform=fig.transFigure,
                                     facecolor=color, edgecolor="none", zorder=2))
    fig.text(x0 + 0.020, y0 + alto_t - 0.030, rot.upper(), fontsize=9.5,
             color=INK_MUTED, va="top", fontweight="600")
    fig.text(x0 + 0.020, y0 + alto_t - 0.062, val, fontsize=25, color=INK,
             va="top", fontweight="600")
    fig.text(x0 + 0.020, y0 + 0.020, sub, fontsize=10, color=INK_2, va="bottom")

ax = fig.add_axes([izq, 0.100, ancho_tot, 0.250])
ax.plot(np.arange(12), ing, color=SERIES["Norte"], linewidth=2.0, zorder=3,
        marker="o", markersize=6, markerfacecolor=SERIES["Norte"],
        markeredgecolor=SURFACE, markeredgewidth=1.8)
ax.set_xticks(np.arange(12))
ax.set_xticklabels(ABREV)
ax.set_xlim(-0.5, 11.5)
ax.set_ylim(ing.min() * 0.90, ing.max() * 1.10)
ax.yaxis.set_major_formatter(FuncFormatter(miles))
ax.set_ylabel("Ingreso mensual", labelpad=8, fontsize=10.5)
limpiar(ax)
ax.annotate(money(ing[-1]), (11, ing[-1]), textcoords="offset points", xytext=(0, 11),
            ha="right", va="bottom", fontsize=10.5, fontweight="600", color=INK)
fig.text(izq, 0.395, "Tendencia del ingreso mensual consolidado",
         fontsize=11.5, color=INK_2, va="bottom", fontweight="600")

titulo(fig, "Desempeño de ingresos 2024: indicadores principales",
       "Resumen consolidado de las cuatro regiones, de enero a diciembre")
pie(fig, "Fuente: ingresos_mensuales.csv · 48 registros, sin valores faltantes")
fig.savefig("04_resumen_indicadores.png", dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
plt.close(fig)

print("Graficas generadas.")
print("Total:", money(total), "| Crecimiento Dic vs Ene:", round(crecimiento, 1), "%")
print("H1:", money(h1), "H2:", money(h2), "delta:", round((h2 / h1 - 1) * 100, 1), "%")
