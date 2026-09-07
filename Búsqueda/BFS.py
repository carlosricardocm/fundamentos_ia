# ==========================================================
# Búsqueda en Anchura (Breadth First Search) 
# ==========================================================


# ----------------------------------------------------------
# Clase Nodo
# ----------------------------------------------------------
class Nodo:
    def __init__(self, estado, padre=None, accion=None):
        self.estado = estado      # (fila, columna)
        self.padre = padre        # Nodo padre
        self.accion = accion      # Acción que llevó a este estado


# ----------------------------------------------------------
# BFS
# ----------------------------------------------------------
def bfs_laberinto(laberinto, inicio, meta):

    acciones = {
        "ARRIBA": (-1, 0),
        "ABAJO": (1, 0),
        "IZQUIERDA": (0, -1),
        "DERECHA": (0, 1)
    }

    # Frontera como cola FIFO
    frontera = [Nodo(inicio)]
    explorado = set()

    while frontera:
        # BFS: sacar el primer elemento es decir se hace una extracción FIFO
        nodo_actual = frontera.pop(0)

        # Prueba de objetivo
        if nodo_actual.estado == meta:
            return reconstruir_camino(nodo_actual)

        explorado.add(nodo_actual.estado)
        fila, col = nodo_actual.estado

        # Expandir sucesores
        for accion, mov in acciones.items():
            nueva_fila = fila + mov[0]
            nueva_col = col + mov[1]
            nuevo_estado = (nueva_fila, nueva_col)

            if (0 <= nueva_fila < len(laberinto) and
                0 <= nueva_col < len(laberinto[0]) and
                laberinto[nueva_fila][nueva_col] != 1 and
                nuevo_estado not in explorado and
                not estado_en_frontera(nuevo_estado, frontera)):

                frontera.append(
                    Nodo(
                        estado=nuevo_estado,
                        padre=nodo_actual,
                        accion=accion
                    )
                )

    return None


# ----------------------------------------------------------
# Verificar estado en frontera
# ----------------------------------------------------------
def estado_en_frontera(estado, frontera):
    return any(nodo.estado == estado for nodo in frontera)


# ----------------------------------------------------------
# Reconstrucción del camino
# ----------------------------------------------------------
def reconstruir_camino(nodo):
    estados = []
    acciones = []

    while nodo.padre is not None:
        estados.append(nodo.estado)
        acciones.append(nodo.accion)
        nodo = nodo.padre

    estados.append(nodo.estado)

    estados.reverse()
    acciones.reverse()

    return estados, acciones


# ----------------------------------------------------------
# Visualización ASCII
# ----------------------------------------------------------
def mostrar_laberinto(laberinto, camino=None, inicio=None, meta=None):

    camino = set(camino) if camino else set()

    for i in range(len(laberinto)):
        fila_ascii = ""
        for j in range(len(laberinto[0])):
            pos = (i, j)

            if pos == inicio:
                fila_ascii += " S "
            elif pos == meta:
                fila_ascii += " G "
            elif laberinto[i][j] == 1:
                fila_ascii += " █ "
            elif pos in camino:
                fila_ascii += " · "
            else:
                fila_ascii += "   "
        print(fila_ascii)
    print()


# ----------------------------------------------------------
# Ejecución principal
# ----------------------------------------------------------
if __name__ == "__main__":

    laberinto = [ 
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0], 
        [0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0]
    ]

    inicio = (0, 0)
    meta = (4, 4)

    print("LABERINTO ORIGINAL:\n")
    mostrar_laberinto(laberinto, inicio=inicio, meta=meta)

    resultado = bfs_laberinto(laberinto, inicio, meta)

    if resultado:
        estados, acciones = resultado

        print("CAMINO ENCONTRADO (BFS):\n")
        mostrar_laberinto(laberinto, camino=estados, inicio=inicio, meta=meta)

        print("Estados recorridos:")
        for e in estados:
            print(e)

        print("\nAcciones:")
        for a in acciones:
            print(a)
    else:
        print("❌ No se encontró solución.")
