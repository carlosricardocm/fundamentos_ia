"""
programacion_lineal.py
===========================================================================
PROGRAMACIÓN LINEAL con scipy.optimize.linprog

No implementamos el símplex ni el punto interior: nuestro trabajo es sólo
FORMULAR el problema y pasárselo a la biblioteca. Ella devuelve el óptimo
GLOBAL, no uno local.

El ejemplo de la clase (las dos máquinas)
-----------------------------------------
    x1 = horas que opera la máquina X1
    x2 = horas que opera la máquina X2

    Minimizar     50 x1 + 80 x2        costo por hora de cada máquina
    sujeto a       5 x1 +  2 x2 <= 20  sólo hay 20 unidades de trabajo
                  10 x1 + 12 x2 >= 90  hacen falta 90 unidades producidas
                     x1, x2 >= 0       no se puede operar horas negativas

Para ejecutarlo:  python programacion_lineal.py
Necesitas:        pip install scipy matplotlib
===========================================================================
"""

from scipy.optimize import linprog


# ===========================================================================
#  1. TRADUCIR EL PROBLEMA AL FORMATO QUE PIDE linprog
# ===========================================================================
#
#  linprog resuelve SIEMPRE este molde:
#
#         minimizar   c @ x
#         sujeto a    A_ub @ x <= b_ub
#                     A_eq @ x == b_eq
#                     cotas inferior/superior de cada variable
#
#  Dos cosas hay que recordar:
#    * linprog sólo MINIMIZA. Para maximizar algo, se minimiza su negativo.
#    * linprog sólo acepta "<=". Una restricción ">=" se voltea
#      multiplicando toda la fila por -1.

# Función de costo:  50*x1 + 80*x2
c = [50, 80]

# Restricciones de la forma  A_ub @ x <= b_ub
A_ub = [
    [  5,   2],     #    5*x1 +  2*x2 <=  20      (trabajo disponible)
    [-10, -12],     #  -10*x1 - 12*x2 <= -90      (era 10*x1 + 12*x2 >= 90)
]
b_ub = [20, -90]

# Cotas de cada variable: (mínimo, máximo). None = sin límite.
cotas = [(0, None),     # x1 >= 0
         (0, None)]     # x2 >= 0


# ===========================================================================
#  2. RESOLVER
# ===========================================================================

resultado = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=cotas, method="highs")


# ===========================================================================
#  3. LEER E INTERPRETAR LA RESPUESTA
# ===========================================================================

print("=" * 60)
print("PROGRAMACIÓN LINEAL — ejemplo de las dos máquinas")
print("=" * 60)

if not resultado.success:
    # Pasa, por ejemplo, si las restricciones se contradicen entre sí.
    print("No hay solución:", resultado.message)
else:
    x1, x2 = resultado.x

    print(f"\nMáquina X1: {x1:.2f} horas")
    print(f"Máquina X2: {x2:.2f} horas")
    print(f"\nCosto mínimo: ${resultado.fun:.2f}")

    # Comprobamos a mano que la solución respeta las restricciones.
    print("\nComprobación de las restricciones:")
    print(f"  trabajo usado : {5*x1 + 2*x2:6.2f}  de  20  disponibles")
    print(f"  producción    : {10*x1 + 12*x2:6.2f}  de  90  necesarias")

print("=" * 60)


# ===========================================================================
#  4. EXTRA: dibujar la región factible
# ===========================================================================
#  Con sólo dos variables el problema se puede ver en el plano. Cada
#  restricción es una recta que parte el plano en dos; la zona que cumple
#  TODAS a la vez es la región factible, y el óptimo siempre cae en una de
#  sus esquinas. Si no te interesa la gráfica, borra de aquí para abajo.
# ===========================================================================

def dibujar_region(archivo="region_factible.png"):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    AZUL, NARANJA, MORADO = "#1E2878", "#F08228", "#78148C"

    x = np.linspace(0, 3, 400)
    techo = (20 - 5 * x) / 2         #  5*x1 + 2*x2 = 20   ->  x2 = (20-5x1)/2
    piso  = (90 - 10 * x) / 12       # 10*x1 + 12*x2 = 90  ->  x2 = (90-10x1)/12

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)

    # La región factible está entre las dos rectas (y arriba de x2 = 0).
    factible = techo >= np.maximum(piso, 0)
    ax.fill_between(x, np.maximum(piso, 0), techo, where=factible,
                    color=AZUL, alpha=0.18, label="región factible")

    ax.plot(x, techo, color=AZUL, linewidth=2, label=r"$5x_1 + 2x_2 = 20$")
    ax.plot(x, piso, color=NARANJA, linewidth=2, label=r"$10x_1 + 12x_2 = 90$")

    # Las esquinas de la región. La idea central de la programación lineal es
    # que el óptimo SIEMPRE está en una de ellas, así que basta con compararlas.
    for ex, ey in [(0, 10), (0, 7.5)]:
        ax.plot(ex, ey, "o", color=AZUL, markersize=7, zorder=5)
        ax.annotate(f"  ${50*ex + 80*ey:.0f}", (ex, ey),
                    color=AZUL, fontsize=9, va="center")

    # La solución que encontró linprog: la esquina más barata.
    if resultado.success:
        ax.plot(*resultado.x, "o", color=MORADO, markersize=11, zorder=6)
        ax.annotate(f"óptimo  ({resultado.x[0]:.2f}, {resultado.x[1]:.2f})\n"
                    f"costo  ${resultado.fun:.0f}",
                    xy=resultado.x, xytext=(1.15, 3.2),
                    color=MORADO, fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=MORADO, linewidth=1.5))

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 11)
    ax.set_xlabel("$x_1$  (horas de la máquina X1)", color=AZUL)
    ax.set_ylabel("$x_2$  (horas de la máquina X2)", color=AZUL)
    ax.set_title("Región factible y solución óptima",
                 color=AZUL, fontweight="bold", pad=12)
    ax.grid(color=AZUL, alpha=0.12)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    for lado in ax.spines.values():
        lado.set_color(AZUL)
    ax.tick_params(colors=AZUL)

    fig.savefig(archivo, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Gráfica guardada en: {archivo}")


if __name__ == "__main__":
    dibujar_region()
