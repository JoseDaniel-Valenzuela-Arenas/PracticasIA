import tkinter as tk
import heapq
from collections import deque

#MAPA
MAPA = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
INICIO = (0, 0)
META = (7, 9)
MOVIMIENTOS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def es_valido(grid, r, c):
    return 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 0

#BFS
def bfs(grid, start, goal):
    queue = deque([[start]])
    explored_set = set([start])
    explored_order = [start]
    
    while queue:
        path = queue.popleft()
        curr = path[-1]
        
        if curr == goal:
            return path, explored_order
            
        for dr, dc in MOVIMIENTOS:
            r, c = curr[0] + dr, curr[1] + dc
            if es_valido(grid, r, c) and (r, c) not in explored_set:
                explored_set.add((r, c))
                explored_order.append((r, c))
                queue.append(path + [(r, c)])
    return None, explored_order

def heuristica(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

#A-ESTRELLA
def a_star(grid, start, goal):
    pq = [(heuristica(start, goal), 0, start, [start])]
    explored_set = set()
    explored_order = []
    g_scores = {start: 0}
    
    while pq:
        f, g, curr, path = heapq.heappop(pq)
        
        if curr in explored_set: continue
        explored_set.add(curr)
        explored_order.append(curr)
        
        if curr == goal:
            return path, explored_order
            
        for dr, dc in MOVIMIENTOS:
            r, c = curr[0] + dr, curr[1] + dc
            neighbor = (r, c)
            if es_valido(grid, r, c):
                temp_g = g + 1
                if neighbor not in g_scores or temp_g < g_scores[neighbor]:
                    g_scores[neighbor] = temp_g
                    f_score = temp_g + heuristica(neighbor, goal)
                    heapq.heappush(pq, (f_score, temp_g, neighbor, path + [neighbor]))
    return None, explored_order

#Interfaz Gráfica (GUI)
class AppBusqueda:
    def __init__(self, root):
        self.root = root
        self.root.title("Comparación BFS vs A*")
        
        self.filas = len(MAPA)
        self.cols = len(MAPA[0])
        self.tamano_celda = 40
        
        # Canvas para dibujar el mapa
        self.canvas = tk.Canvas(root, width=self.cols*self.tamano_celda, height=self.filas*self.tamano_celda)
        self.canvas.pack(pady=10)
        
        # Panel de controles
        panel = tk.Frame(root)
        panel.pack(pady=5)
        
        tk.Button(panel, text="Ejecutar BFS", command=lambda: self.ejecutar_algoritmo('bfs')).grid(row=0, column=0, padx=5)
        tk.Button(panel, text="Ejecutar A*", command=lambda: self.ejecutar_algoritmo('astar')).grid(row=0, column=1, padx=5)
        tk.Button(panel, text="Limpiar Mapa", command=self.dibujar_mapa).grid(row=0, column=2, padx=5)
        
        self.label_info = tk.Label(root, text="Selecciona un algoritmo para comenzar", font=("Arial", 12))
        self.label_info.pack(pady=5)
        
        self.dibujar_mapa()

    def dibujar_mapa(self):
        self.canvas.delete("all")
        self.rectangulos = {}
        for r in range(self.filas):
            for c in range(self.cols):
                x1, y1 = c * self.tamano_celda, r * self.tamano_celda
                x2, y2 = x1 + self.tamano_celda, y1 + self.tamano_celda
                
                color = "white"
                if MAPA[r][c] == 1: color = "black"
                elif (r, c) == INICIO: color = "green"
                elif (r, c) == META: color = "red"
                
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")
                self.rectangulos[(r, c)] = rect

    def ejecutar_algoritmo(self, tipo):
        self.dibujar_mapa() # Limpiar antes de ejecutar
        
        if tipo == 'bfs':
            ruta, explorados = bfs(MAPA, INICIO, META)
            nombre = "BFS"
        else:
            ruta, explorados = a_star(MAPA, INICIO, META)
            nombre = "A*"
            
        self.label_info.config(text=f"Calculando {nombre}...")
        self.animar_exploracion(explorados, ruta, nombre, 0)

    def animar_exploracion(self, explorados, ruta, nombre, index):
        # Animar celdas exploradas una por una
        if index < len(explorados):
            nodo = explorados[index]
            if nodo != INICIO and nodo != META:
                self.canvas.itemconfig(self.rectangulos[nodo], fill="lightblue")
            self.root.after(20, self.animar_exploracion, explorados, ruta, nombre, index + 1)
        else:
            #dibuja la ruta final
            if ruta:
                for nodo in ruta:
                    if nodo != INICIO and nodo != META:
                        self.canvas.itemconfig(self.rectangulos[nodo], fill="yellow")
            
            self.label_info.config(text=f"{nombre}: {len(explorados)} estados explorados | Ruta: {len(ruta)-1} pasos")

if __name__ == "__main__":
    ventana = tk.Tk()
    app = AppBusqueda(ventana)
    ventana.mainloop()