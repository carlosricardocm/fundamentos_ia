"""
ascenso_colina.py
===========================================================================
ASCENSO DE COLINA  (hill climbing), variante de "ascenso más pronunciado".

La idea es la más simple que existe:

    Estoy en un estado. Miro a todos mis vecinos.
    Si alguno es mejor, me muevo al mejor de ellos y repito.
    Si ninguno es mejor, me detengo.

Como en este problema queremos MINIMIZAR el costo, "mejor" significa
"de costo más bajo".

Para ejecutarlo:      python ascenso_colina.py
Genera las imágenes:  imagenes_ascenso/paso_00_costo_17.png, etc.
===========================================================================
"""

import random

from espacio import (costo, vecinos, dibujar,
                     configuracion_aleatoria, optimo_exhaustivo)


def ascenso_de_colina(inicio, carpeta=None, mostrar=True):
    """
    Aplica ascenso de colina partiendo del estado `inicio`.

    Devuelve el mejor estado encontrado.

    carpeta : si le das un nombre de carpeta, guarda una imagen del estado
              inicial y otra cada vez que encontramos una solución mejor.
    mostrar : si es False no imprime nada (útil al llamarla muchas veces).
    """
    actual = inicio
    paso = 0

    if carpeta:
        dibujar(actual,
                f"{carpeta}/paso_{paso:02d}_costo_{costo(actual)}.png",
                etiqueta=f"paso {paso}  (estado inicial)")
    if mostrar:
        print(f"paso {paso}: costo = {costo(actual):3d}   hospitales en {actual}")

    while True:
        # 1. Miramos a TODOS los vecinos y nos quedamos con el de menor costo.
        mejor_vecino = min(vecinos(actual), key=costo)

        # 2. Si ni siquiera el mejor vecino mejora, ya no hay a dónde ir.
        if costo(mejor_vecino) >= costo(actual):
            if mostrar:
                print(f"Ningún vecino mejora el costo {costo(actual)}: nos detenemos.")
            return actual

        # 3. Nos movemos al vecino y guardamos la imagen de la nueva solución.
        actual = mejor_vecino
        paso += 1

        if carpeta:
            dibujar(actual,
                    f"{carpeta}/paso_{paso:02d}_costo_{costo(actual)}.png",
                    etiqueta=f"paso {paso}")
        if mostrar:
            print(f"paso {paso}: costo = {costo(actual):3d}   hospitales en {actual}")


# ===========================================================================
if __name__ == "__main__":

    random.seed(32)         # quítalo para obtener un inicio distinto cada vez

    inicio = configuracion_aleatoria()
    solucion = ascenso_de_colina(inicio, carpeta="imagenes_ascenso")

    print()
    print(f"Solución encontrada : {solucion}   costo {costo(solucion)}")

    # Como la cuadrícula es pequeña podemos calcular la respuesta perfecta
    # y comprobar si el ascenso de colina se quedó atrapado en un óptimo local.
    mejor_posible = optimo_exhaustivo()
    print(f"Mejor costo posible : {costo(mejor_posible)}")
    if costo(solucion) > costo(mejor_posible):
        print("--> El ascenso de colina se quedó atrapado en un ÓPTIMO LOCAL.")
    else:
        print("--> Esta vez sí llegó al óptimo global.")
