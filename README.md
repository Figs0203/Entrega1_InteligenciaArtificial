# Entrega 1 Inteligencia Artificial S2566 Universidad EAFIT
### Esteban Álvarez Zuluaga, Agustín Figueroa, Alejandro Sepúlveda

---

Este proyecto implementa agentes de búsqueda (A*, BFS) y un algoritmo genético para encontrar rutas óptimas entre barrios de Medellín, usando grafos y heurísticas. Incluye visualización de rutas y comparación de métodos.

¡Visita la wiki donde encontrarás más información al respecto!

---

## Estructura de Carpetas

- **Agent/**  
  Contiene los agentes de búsqueda:
  - `agent_search.ipynb`: Notebook explicativo y ejecutable.

- **GA/**  
  Algoritmo genético para rutas:
  - `genetic_agent.ipynb`: Notebook con todo el flujo del algoritmo genético.

- **img/**  
  Imágenes generadas de rutas y soluciones.

- **Graph.py**  
  Clase auxiliar para graficar rutas y grafos usando NetworkX y Matplotlib.

- **requirements.txt**  
  Lista de dependencias necesarias.

---

## Instalación

1. Clona el repositorio o descarga los archivos.
2. Instala las dependencias con:

   ```
   pip install -r requirements.txt
   ```

---

## Uso

### 1. Agente de Búsqueda (A*, BFS)

- O abre y ejecuta el notebook `Agent/agent_search.ipynb` en Jupyter/VS Code.

### 2. Algoritmo Genético

- Abre y ejecuta el notebook `GA/genetic_agent.ipynb` en Jupyter/VS Code.
- El notebook genera rutas óptimas y visualizaciones en la carpeta `img/`.

---

## Dependencias

- `matplotlib`
- `networkx`

Instaladas automáticamente con el archivo `requirements.txt`.

---

## Descripción de Archivos Principales

- **agent_search.ipynb**  
  Implementa  los algoritmos A* y BFS para encontrar rutas entre barrios, mostrando resultados y pasos de ejecución.

- **genetic_agent.ipynb**  
  Implementa un algoritmo genético para encontrar el mejor orden de visita entre varios barrios, usando A* para calcular distancias y visualizando el resultado.

- **Graph.py**  
  Permite crear y visualizar grafos y rutas usando NetworkX y Matplotlib.

---

## Ejemplo de Ejecución

```python
# Ejecuta el agente de búsqueda
# Abre Agent/agent_search.ipynb y ejecuta todas las celdas

# Ejecuta el algoritmo genético (en notebook)
# Abre GA/genetic_agent.ipynb y ejecuta todas las celdas
```





