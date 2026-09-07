# ==========================================================
# Búsqueda en Profundidad (Depth First Search) en un laberinto
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
# DFS
# ----------------------------------------------------------
def dfs_laberinto(laberinto, inicio, meta):

    acciones = {
        "ARRIBA": (-1, 0),
        "ABAJO": (1, 0),
        "IZQUIERDA": (0, -1),
        "DERECHA": (0, 1)
    }

    frontera = [Nodo(inicio)]
    explorado = set() #utilizamos un set porque no queremos elementos duplicados

    while frontera:
        # Extraer el último nodo añadido (LIFO) lo cual es lo principal en DFS
        nodo_actual = frontera.pop()

        # Verificar si es el estado meta, solución, objetivo
        if nodo_actual.estado == meta:
            return reconstruir_camino(nodo_actual)

        explorado.add(nodo_actual.estado)
        fila, col = nodo_actual.estado

        # En estas líneas de código esta el corázon del algoritmo es decir expandir el nodo actual con los posibles movimientos válidos que podemos hacer dentro del laberinto y guardarlos en la frontera
        for accion, mov in acciones.items():
            nueva_fila = fila + mov[0]
            nueva_col = col + mov[1]
            nuevo_estado = (nueva_fila, nueva_col)

            if (0 <= nueva_fila < len(laberinto) and #Verifica que no nos salgamos del arreglo o que tengamos errores de índice
                0 <= nueva_col < len(laberinto[0]) and
                laberinto[nueva_fila][nueva_col] != 1 and #Verifica que no sea una pared es decir que el valor sea distinto a 1
                nuevo_estado not in explorado and #Verifica que no esté en explorados, esto evita ciclos como A -> B -> A -> B garantizando que cada estado se expanda una sola vez
                not estado_en_frontera(nuevo_estado, frontera)): #Verifica que no esté en la frontera para evitar duplicados

                frontera.append(
                    Nodo(
                        estado=nuevo_estado,
                        padre=nodo_actual,
                        accion=accion
                    )
                ) 

    return None


# ----------------------------------------------------------
# Verificar estado en frontera, es decir si ya está en la lista. Ejemplo:
# frontera = [Nodo((0,0)), Nodo((1,0)), Nodo((2,0))]
# estado = (1,0)
# any([False, True, False])   # True
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
# Visualización ASCII del laberinto
# ----------------------------------------------------------
def mostrar_laberinto(laberinto, camino=None, inicio=None, meta=None):
    """
    Representación ASCII:
    █ -> pared
      -> espacio libre
    S -> inicio
    G -> meta
    · -> camino solución
    """

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

    resultado = dfs_laberinto(laberinto, inicio, meta)

    if resultado:
        estados, acciones = resultado

        print("CAMINO ENCONTRADO (DFS):\n")
        mostrar_laberinto(laberinto, camino=estados, inicio=inicio, meta=meta)

        print("Estados recorridos:")
        for e in estados:
            print(e)

        print("\nAcciones:")
        for a in acciones:
            print(a)
    else:
        print("❌ No se encontró solución.")
