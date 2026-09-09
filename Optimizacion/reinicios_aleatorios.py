"""
reinicios_aleatorios.py
===========================================================================
REINICIOS ALEATORIOS  (random-restart hill climbing).

El ascenso de colina se queda atrapado en el primer óptimo local que
encuentra. El truco más barato para arreglarlo es también el más obvio:

    Repetir el ascenso de colina muchas veces, cada una desde un
    punto de partida distinto elegido al azar, y quedarnos con
    la mejor solución de todas.

Aquí reutilizamos tal cual la función `ascenso_de_colina` del otro archivo.

Para ejecutarlo:      python reinicios_aleatorios.py
Genera las imágenes:  imagenes_reinicios/mejora_1_costo_13.png, etc.
                      (una imagen sólo cuando el récord global mejora)
===========================================================================
"""

import random

from espacio import costo, dibujar, configuracion_aleatoria, optimo_exhaustivo
from ascenso_colina import ascenso_de_colina


def reinicios_aleatorios(n_reinicios, carpeta=None):
    """
    Ejecuta el ascenso de colina `n_reinicios` veces desde puntos al azar.

    Devuelve el mejor estado encontrado en todos los intentos.
    Guarda una imagen cada vez que se rompe el récord.
    """
    mejor = None
    n_mejoras = 0

    for intento in range(1, n_reinicios + 1):

        # Un ascenso de colina completo, sin imágenes intermedias.
        candidato = ascenso_de_colina(configuracion_aleatoria(), mostrar=False)

        # ¿Es mejor que todo lo que habíamos visto hasta ahora?
        if mejor is None or costo(candidato) < costo(mejor):
            mejor = candidato
            n_mejoras += 1
            print(f"  >>> intento {intento}: NUEVO RÉCORD, costo {costo(mejor)}")

            if carpeta:
                dibujar(mejor,
                        f"{carpeta}/mejora_{n_mejoras}_costo_{costo(mejor)}.png",
                        etiqueta=f"reinicio {intento}  ·  mejora {n_mejoras}")
        else:
            print(f"      intento {intento}: costo {costo(candidato)} "
                  f"(no mejora el récord de {costo(mejor)})")

    return mejor


# ===========================================================================
if __name__ == "__main__":

    random.seed(32)         # quítalo para obtener resultados distintos cada vez

    solucion = reinicios_aleatorios(10, carpeta="imagenes_reinicios")

    print()
    print(f"Solución encontrada : {solucion}   costo {costo(solucion)}")
    print(f"Mejor costo posible : {costo(optimo_exhaustivo())}")
