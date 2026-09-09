"""
espacio.py
===========================================================================
El problema de las CASAS y los HOSPITALES.

Tenemos una cuadrícula con varias casas. Queremos construir un número fijo
de hospitales de modo que la suma de las distancias de cada casa a su
hospital más cercano sea lo más pequeña posible.

Este archivo contiene:

  1. La definición del problema  -> distancia, costo
  2. Cómo generar estados        -> configuracion_aleatoria, vecinos
  3. Cómo dibujar un estado      -> dibujar

Los algoritmos (ascenso de colina, reinicios aleatorios, recocido simulado)
viven en archivos aparte e importan lo que necesitan de aquí.

Convención de coordenadas
-------------------------
Una casilla es una tupla (fila, columna) empezando en 1:
la fila 1 es la de ARRIBA y la columna 1 es la de la IZQUIERDA.
===========================================================================
"""

import os
import random

import matplotlib
matplotlib.use("Agg")            # para guardar imágenes sin abrir ventanas
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch


# ===========================================================================
#  1. DEFINICIÓN DEL PROBLEMA
# ===========================================================================

FILAS = 5
COLUMNAS = 10

# Las cuatro casas del ejemplo de la clase (fila, columna)
CASAS = [(2, 3), (1, 9), (4, 2), (5, 7)]

N_HOSPITALES = 2


def distancia(a, b):
    """Distancia de Manhattan: pasos en vertical + pasos en horizontal."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def costo(hospitales):
    """
    Función de costo del problema.

    Para cada casa buscamos su hospital más cercano y sumamos esas
    distancias. Cuanto MÁS PEQUEÑO, mejor.
    """
    return sum(min(distancia(casa, h) for h in hospitales) for casa in CASAS)


# ===========================================================================
#  2. ESTADOS Y VECINOS
# ===========================================================================

def casillas_libres(ocupadas=()):
    """Todas las casillas de la cuadrícula que no tienen casa ni hospital."""
    return [(f, c)
            for f in range(1, FILAS + 1)
            for c in range(1, COLUMNAS + 1)
            if (f, c) not in CASAS and (f, c) not in ocupadas]


def configuracion_aleatoria():
    """Coloca los hospitales al azar en casillas libres. Es el punto de partida."""
    return tuple(random.sample(casillas_libres(), N_HOSPITALES))


def vecinos(hospitales):
    """
    Los VECINOS de un estado.

    Un vecino es lo que obtenemos al mover UN solo hospital UNA sola casilla
    (arriba, abajo, izquierda o derecha), sin salirnos de la cuadrícula y sin
    caer encima de una casa o del otro hospital.
    """
    resultado = []
    for i, (fila, columna) in enumerate(hospitales):
        for df, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nueva = (fila + df, columna + dc)

            # ¿se sale de la cuadrícula?
            if not (1 <= nueva[0] <= FILAS and 1 <= nueva[1] <= COLUMNAS):
                continue
            # ¿cae sobre una casa o sobre el otro hospital?
            if nueva in CASAS or nueva in hospitales:
                continue

            vecino = list(hospitales)
            vecino[i] = nueva
            resultado.append(tuple(vecino))
    return resultado


def optimo_exhaustivo():
    """
    Prueba TODAS las configuraciones posibles y devuelve la mejor.

    Sólo sirve porque la cuadrícula es diminuta: nos da la respuesta correcta
    contra la cual comparar lo que encuentran los algoritmos de búsqueda local.
    """
    from itertools import combinations
    return min(combinations(casillas_libres(), N_HOSPITALES), key=costo)


# ===========================================================================
#  3. DIBUJO
# ===========================================================================

# Colores del tema de la presentación
AZUL     = "#1E2878"   # casas, textos
NARANJA  = "#F08228"   # hospitales
REJILLA  = "#B0B4D1"   # líneas de la cuadrícula
BORDE    = "#6169A1"   # marco exterior
RUTA     = "#1189B5"   # líneas casa -> hospital


def _xy(casilla):
    """Convierte (fila, columna) al centro de la casilla en coordenadas del dibujo."""
    fila, columna = casilla
    return (columna - 0.5, FILAS - fila + 0.5)


def _casa(ax, casilla):
    """Dibuja el icono de una casa."""
    x, y = _xy(casilla)
    contorno = [(-0.40, 0.04), (0, 0.38), (0.40, 0.04), (0.29, 0.04),
                (0.29, -0.30), (-0.29, -0.30), (-0.29, 0.04)]
    ax.add_patch(Polygon([(x + dx, y + dy) for dx, dy in contorno],
                         closed=True, facecolor=AZUL, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x + 0.20, y + 0.16), 0.08, 0.18,
                           facecolor=AZUL, edgecolor="none", zorder=3))   # chimenea
    ax.add_patch(Rectangle((x - 0.085, y - 0.30), 0.17, 0.28,
                           facecolor="white", edgecolor="none", zorder=4))  # puerta


def _hospital(ax, casilla):
    """Dibuja el icono de un hospital."""
    x, y = _xy(casilla)
    ax.add_patch(FancyBboxPatch((x - 0.28, y - 0.31), 0.56, 0.62,
                                boxstyle="round,pad=0.05,rounding_size=0.04",
                                facecolor=NARANJA, edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x - 0.06, y - 0.36), 0.12, 0.24,
                           facecolor="white", edgecolor="none", zorder=4))  # entrada
    ax.add_patch(Rectangle((x - 0.19, y + 0.03), 0.38, 0.12,
                           facecolor="white", edgecolor="none", zorder=4))  # cruz —
    ax.add_patch(Rectangle((x - 0.06, y - 0.10), 0.12, 0.38,
                           facecolor="white", edgecolor="none", zorder=4))  # cruz |


def _ruta(ax, casa, hospital):
    """
    Dibuja el camino de una casa a su hospital: primero en vertical y
    después en horizontal (por eso la línea forma una 'L').
    """
    x0, y0 = _xy(casa)
    x1, y1 = _xy(hospital)
    ax.plot([x0, x0, x1], [y0, y1, y1],
            color=RUTA, linewidth=3, solid_capstyle="round",
            solid_joinstyle="round", zorder=2)


def dibujar(hospitales, archivo, etiqueta=None):
    """
    Guarda una imagen PNG del estado `hospitales`.

    Dibuja la cuadrícula, las casas, los hospitales, la línea que une cada
    casa con su hospital más cercano y el costo total arriba a la derecha.

    `etiqueta` es un texto opcional que aparece arriba a la izquierda
    (por ejemplo "paso 3" o "reinicio 7").
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.1), dpi=150)

    # --- cuadrícula ---
    for c in range(COLUMNAS + 1):
        ax.plot([c, c], [0, FILAS], color=REJILLA, linewidth=0.9, zorder=1)
    for f in range(FILAS + 1):
        ax.plot([0, COLUMNAS], [f, f], color=REJILLA, linewidth=0.9, zorder=1)
    ax.add_patch(Rectangle((0, 0), COLUMNAS, FILAS, fill=False,
                           edgecolor=BORDE, linewidth=1.8, zorder=1))

    # --- una línea de cada casa a su hospital más cercano ---
    for casa in CASAS:
        # Si hay empate en distancia de Manhattan, dibujamos la línea hacia el
        # hospital que está más cerca en línea recta: se ve más limpio.
        cercano = min(hospitales,
                      key=lambda h: (distancia(casa, h),
                                     (casa[0] - h[0])**2 + (casa[1] - h[1])**2))
        _ruta(ax, casa, cercano)

    # --- iconos (encima de las líneas) ---
    for casa in CASAS:
        _casa(ax, casa)
    for h in hospitales:
        _hospital(ax, h)

    # --- textos ---
    ax.text(COLUMNAS, FILAS + 0.18, f"Costo: {costo(hospitales)}",
            ha="right", va="bottom", color=AZUL, fontsize=15, fontweight="bold")
    if etiqueta:
        ax.text(0, FILAS + 0.18, etiqueta,
                ha="left", va="bottom", color=NARANJA, fontsize=12)

    ax.set_xlim(-0.15, COLUMNAS + 0.15)
    ax.set_ylim(-0.15, FILAS + 0.85)
    ax.set_aspect("equal")
    ax.axis("off")

    carpeta = os.path.dirname(archivo)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)
    fig.savefig(archivo, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return archivo
