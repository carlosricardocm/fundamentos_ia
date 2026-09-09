"""
recocido_simulado.py
===========================================================================
RECOCIDO SIMULADO  (simulated annealing).

El ascenso de colina nunca acepta empeorar, y por eso se queda atrapado.
El recocido simulado sí acepta empeorar... pero cada vez menos:

    Al principio la "temperatura" es alta  -> aceptamos vecinos peores
                                              con facilidad (exploramos).
    Al final la "temperatura" es baja      -> casi sólo aceptamos mejoras
                                              (afinamos la solución).

En cada iteración elegimos UN vecino al azar (no los miramos todos) y:

    si el vecino es mejor            -> nos movemos siempre
    si el vecino es peor             -> nos movemos con probabilidad e^(ΔE/T)

donde ΔE es qué tanto mejor es el vecino (negativo si es peor) y T es la
temperatura del momento.

Para ejecutarlo:      python recocido_simulado.py
Genera las imágenes:  imagenes_recocido/mejora_1_costo_15.png, etc.
                      (una imagen cada vez que se rompe el récord)
===========================================================================
"""

import math
import random

from espacio import (costo, vecinos, dibujar,
                     configuracion_aleatoria, optimo_exhaustivo)


def temperatura(iteracion, total):
    """
    Programa de enfriamiento: empieza en 1.0 y baja hasta casi 0.

    Es la versión más simple posible (lineal). Nunca devolvemos 0 exacto
    para no dividir entre cero.
    """
    return max(0.01, 1.0 - iteracion / total)


def recocido_simulado(inicio, iteraciones=1000, carpeta=None):
    """
    Aplica recocido simulado partiendo del estado `inicio`.

    Devuelve el mejor estado visto durante toda la búsqueda.
    Guarda una imagen cada vez que se rompe el récord.
    """
    actual = inicio
    mejor = inicio            # el récord: el mejor estado visto hasta ahora
    n_mejoras = 0

    if carpeta:
        dibujar(actual,
                f"{carpeta}/mejora_0_costo_{costo(actual)}.png",
                etiqueta="estado inicial")
    print(f"inicio: costo = {costo(inicio)}")

    for t in range(1, iteraciones + 1):

        T = temperatura(t, iteraciones)

        # 1. Un vecino al azar (aquí NO miramos a todos, sólo a uno).
        vecino = random.choice(vecinos(actual))

        # 2. ¿Qué tanto MEJOR es el vecino? Como minimizamos el costo,
        #    delta > 0 significa que el vecino es mejor.
        delta = costo(actual) - costo(vecino)

        # 3. Si es mejor nos movemos siempre; si es peor, con probabilidad
        #    e^(delta/T), que es pequeña cuando la temperatura ya bajó.
        if delta > 0 or random.random() < math.exp(delta / T):
            actual = vecino

        # 4. Guardamos el récord (y su imagen) si acabamos de superarlo.
        if costo(actual) < costo(mejor):
            mejor = actual
            n_mejoras += 1
            print(f"  >>> iteración {t:4d} (T={T:.2f}): "
                  f"NUEVO RÉCORD, costo {costo(mejor)}")

            if carpeta:
                dibujar(mejor,
                        f"{carpeta}/mejora_{n_mejoras}_costo_{costo(mejor)}.png",
                        etiqueta=f"iteración {t}  ·  T = {T:.2f}")

    return mejor


# ===========================================================================
if __name__ == "__main__":

    random.seed(32)         # quítalo para obtener resultados distintos cada vez

    solucion = recocido_simulado(configuracion_aleatoria(),
                                 iteraciones=1000,
                                 carpeta="imagenes_recocido")

    print()
    print(f"Solución encontrada : {solucion}   costo {costo(solucion)}")
    print(f"Mejor costo posible : {costo(optimo_exhaustivo())}")
