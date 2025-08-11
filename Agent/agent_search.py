# Agent de búsqueda (A*) para el repartidor — agent_search.py
# ---------------------------------------------------------
# Instrucciones:
# - Guardar como Agent/agent_search.py o ejecutar celda por celda en Jupyter.
# - Llamar a la función `run_demo()` para ver un ejemplo con salida paso a paso.
#
# Requisitos: solo librerías estándar (heapq, math, time). Opcional: networkx+matplotlib
# si quieres visualizar gráficamente (el código maneja la ausencia de networkx).

import heapq
import math
import time
from collections import deque

# -----------------------------
# 1) Definición del grafo (ejemplo)
# -----------------------------
# Grafo simétrico: cada arista se declara en ambas direcciones.
# Nodos = barrios (ejemplo simplificado). Los pesos son distancias en km (ejemplo).
# Grafo más realista de Medellín con varios barrios y distancias aproximadas (en km)
GRAPH = {
    "Robledo":       [("Laureles", 3.0), ("Doce de Octubre", 2.5), ("San Cristóbal", 4.0)],
    "Doce de Octubre": [("Robledo", 2.5), ("Castilla", 1.8)],
    "Castilla":      [("Doce de Octubre", 1.8), ("Tricentenario", 1.2)],
    "Tricentenario": [("Castilla", 1.2), ("Aranjuez", 1.5), ("Prado", 2.0)],
    "Aranjuez":      [("Tricentenario", 1.5), ("Buenos Aires", 3.5)],
    "Prado":         [("Tricentenario", 2.0), ("Centro", 1.2)],
    "Centro":        [("Prado", 1.2), ("Laureles", 2.5), ("Buenos Aires", 2.0), ("El Poblado", 4.0)],
    "Buenos Aires":  [("Centro", 2.0), ("Aranjuez", 3.5), ("San Diego", 1.5)],
    "San Diego":     [("Buenos Aires", 1.5), ("El Poblado", 2.0)],
    "Laureles":      [("Robledo", 3.0), ("Centro", 2.5), ("Belén", 2.0), ("Estadio", 1.0)],
    "Estadio":       [("Laureles", 1.0), ("El Poblado", 4.5)],
    "Belén":         [("Laureles", 2.0), ("El Poblado", 3.0)],
    "El Poblado":    [("Belén", 3.0), ("Estadio", 4.5), ("Centro", 4.0), ("San Diego", 2.0)],
    "San Cristóbal": [("Robledo", 4.0)]
}

# Coordenadas aproximadas (solo para heurística)
COORDS = {
    "San Cristóbal": (0.0, 5.0),
    "Robledo": (2.0, 5.0),
    "Doce de Octubre": (3.5, 6.5),
    "Castilla": (5.0, 6.0),
    "Tricentenario": (6.0, 5.0),
    "Aranjuez": (8.0, 6.0),
    "Prado": (7.0, 4.0),
    "Centro": (8.0, 3.0),
    "Buenos Aires": (10.0, 4.0),
    "San Diego": (11.0, 3.0),
    "Laureles": (4.0, 4.0),
    "Estadio": (5.0, 3.0),
    "Belén": (3.0, 3.0),
    "El Poblado": (9.0, 1.0)
}


# -----------------------------
# 2) Heurísticas
# -----------------------------
def euclidean_heuristic(node_a, node_b, coords=COORDS):
    """Distancia euclidiana entre node_a y node_b (en mismas unidades que coords)."""
    (x1, y1) = coords[node_a]
    (x2, y2) = coords[node_b]
    return math.hypot(x1 - x2, y1 - y2)

def manhattan_heuristic(node_a, node_b, coords=COORDS):
    (x1, y1) = coords[node_a]
    (x2, y2) = coords[node_b]
    return abs(x1 - x2) + abs(y1 - y2)

# -----------------------------
# 3) Implementación de A*
# -----------------------------
def a_star(graph, start, goal, heuristic=euclidean_heuristic, verbose=False):
    """
    Ejecuta A* sobre `graph` desde `start` hasta `goal`.
    - graph: dict nodo -> list of (vecino, costo)
    - heuristic: funcion(node, goal) -> float
    - verbose: si True imprime paso a paso la exploración y frontier
    Retorna: dict con keys: path (lista), cost_total, nodes_expanded, explored_sequence, time_s
    """
    t0 = time.perf_counter()
    # frontier: heap de tuplas (f, contador, nodo)
    frontier = []
    counter = 0
    g_score = {start: 0.0}        # costo desde start hasta el nodo
    f_start = heuristic(start, goal)
    heapq.heappush(frontier, (f_start, counter, start))
    parent = {start: None}
    explored = set()
    explored_sequence = []

    while frontier:
        f_current, _, current = heapq.heappop(frontier)

        # Si ya expandimos lo ignoramos (pueden haber entradas antiguas en el heap)
        if current in explored:
            continue

        explored.add(current)
        explored_sequence.append(current)

        if verbose:
            print(f"> Pop nodo: {current}  (f={f_current:.3f}, g={g_score.get(current, float('inf')):.3f}, h={f_current - g_score.get(current,0):.3f})")
            # Mostrar frontier actual (no ordenado)
            frontier_snapshot = [(item[2], item[0]) for item in frontier]
            print("  Frontier (nodo, f):", frontier_snapshot)

        if current == goal:
            # reconstruir camino
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            t1 = time.perf_counter()
            return {
                "path": path,
                "cost_total": g_score[goal],
                "nodes_expanded": len(explored),
                "explored_sequence": explored_sequence,
                "time_s": t1 - t0
            }

        # Expandir vecinos
        for neighbor, cost in graph.get(current, []):
            tentative_g = g_score[current] + cost
            # Si no tiene g_score conocido o encontramos uno mejor
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                parent[neighbor] = current
                h = heuristic(neighbor, goal)
                f_new = tentative_g + h
                counter += 1
                heapq.heappush(frontier, (f_new, counter, neighbor))
                if verbose:
                    print(f"    -> Considerando vecino: {neighbor}  (costo={cost:.2f})  tentative_g={tentative_g:.3f}  h={h:.3f}  f={f_new:.3f}")
    # Si se vació frontier y no encontramos goal
    t1 = time.perf_counter()
    return {
        "path": None,
        "cost_total": None,
        "nodes_expanded": len(explored),
        "explored_sequence": explored_sequence,
        "time_s": t1 - t0
    }

# -----------------------------
# 4) BFS para comparar (búsqueda ciega)
# -----------------------------
def bfs(graph, start, goal, verbose=False):
    """
    BFS clásico por niveles — útil para comparar exploración (nota: no optimiza distancias ponderadas).
    Retorna similar a a_star.
    """
    t0 = time.perf_counter()
    q = deque([start])
    parent = {start: None}
    visited = {start}
    explored_sequence = []

    while q:
        current = q.popleft()
        explored_sequence.append(current)
        if verbose:
            print(f"> Expandido: {current}  Frontier: {list(q)}")
        if current == goal:
            # reconstruir camino (en términos de aristas, no de suma de pesos)
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            t1 = time.perf_counter()
            return {
                "path": path,
                "cost_total": None,  # no relevante para BFS con pesos
                "nodes_expanded": len(visited),
                "explored_sequence": explored_sequence,
                "time_s": t1 - t0
            }
        for neighbor, _ in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                q.append(neighbor)
    t1 = time.perf_counter()
    return {
        "path": None,
        "cost_total": None,
        "nodes_expanded": len(visited),
        "explored_sequence": explored_sequence,
        "time_s": t1 - t0
    }

# -----------------------------
# 5) Función helper para mostrar resultados
# -----------------------------
def print_result(result, algorithm_name="A*"):
    print("\n--- Resultado:", algorithm_name, "---")
    if result["path"] is None:
        print("No se encontró camino.")
    else:
        print("Camino:", " -> ".join(result["path"]))
        if result["cost_total"] is not None:
            print(f"Costo total (distancia): {result['cost_total']:.3f} km")
    print("Nodos expandidos:", result["nodes_expanded"])
    print("Secuencia de expansión:", result["explored_sequence"])
    print(f"Tiempo ejecución: {result['time_s']:.6f} s\n")

# -----------------------------
# 6) Demo / Ejecución de ejemplo
# -----------------------------
def main(verbose=True):
    start = "Robledo"
    goal = "Estadio"
    print(f"\nBuscar camino desde {start} hasta {goal}\n")

    print("Ejecutando A* (heurística euclidiana) ...")
    res_a = a_star(GRAPH, start, goal, heuristic=euclidean_heuristic, verbose=verbose)
    print_result(res_a, "A* (Euclidiana)")

    print("Ejecutando BFS (para comparar) ...")
    res_bfs = bfs(GRAPH, start, goal, verbose=verbose)
    print_result(res_bfs, "BFS (no ponderado)")

if __name__ == "__main__":
    # Si ejecutas el script directamente, correrá la demo con verbose True.
    main(verbose=True)
