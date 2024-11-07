
import agentpy as ap
import numpy as np
import random
import heapq

# Definir la clase del agente Tractor
class TractorAgent(ap.Agent):

    def setup(self):
        # Inicializar los atributos del tractor
        self.capacity = self.p.capacity  # Capacidad máxima de carga
        self.load = 0  # Carga actual
        self.fuel_level = self.p.max_fuel  # Nivel de combustible inicial
        self.fuel_consumption_rate = self.p.fuel_consumption_rate  # Consumo de combustible por movimiento
        self.speed = self.p.speed  # Velocidad del tractor
        self.broken_down = False  # Estado del tractor (averiado o no)
        self.repair_time = 0  # Tiempo de reparación restante
        self.grid = self.model.grid  # Referencia a la cuadrícula del modelo

    def move(self):
        # Método para mover el tractor en cada paso de tiempo
        if self.broken_down:
            # Si el tractor está averiado, reducir el tiempo de reparación
            self.repair_time -= 1
            if self.repair_time <= 0:
                self.broken_down = False  # El tractor ha sido reparado
            return  # No se puede mover si está averiado

        if self.fuel_level <= 0:
            # Si no tiene combustible, no puede moverse
            return

        if self.load >= self.capacity:
            # Si ha alcanzado la capacidad máxima, regresar al punto de descarga
            target = self.model.unload_point

        elif self.fuel_level <= self.p.fuel_threshold:
            # Si el combustible es bajo, ir a la estación de recarga
            target = self.model.refuel_station

        else:
            # Buscar la parcela más cercana lista para cosechar
            target = self.find_nearest_parcel()
            if target is None:
                # No hay parcelas para cosechar, permanecer en su lugar
                return

        # Obtener la posición actual del tractoe
        if self in self.grid.positions:  # Verificar si el tractor tiene una posición asignada
            current_pos = self.grid.positions[self]
        else:
            return  # Si el tractor no tiene posición, salir de la función

        # Encontrar la ruta más corta hacia el objetivo (A*)
        path = self.a_star_path(current_pos, target)
        if len(path) > 1:
            # Moverse al siguiente paso en el camino
            next_position = path[1]
            self.grid.move_to(self, next_position)
            self.fuel_level -= self.fuel_consumption_rate  # Reducir el combustible
            # Posibilidad aleatoria de que el tractor se averíe
            if random.random() < self.p.breakdown_chance:
                self.broken_down = True
                self.repair_time = self.p.repair_steps

            # Verificar si ha llegado al objetivo
            if next_position == target:
                if target == self.model.unload_point:
                    self.unload()
                elif target == self.model.refuel_station:
                    self.refuel()
                else:
                    self.harvest(target)

    def find_nearest_parcel(self):
        # Encontrar la parcela más cercana lista para cosechar
        parcels = self.model.parcels_ready
        if not parcels:
            return None
        
        # Verificar si el tractor tiene una posición asignada
        if self in self.grid.positions:  
            current_pos = self.grid.positions[self]
        else:
            return None  # Si el tractor no tiene posición, regresar None
        
        distances = [self.get_distance(current_pos, p) for p in parcels]
        nearest_parcel = parcels[np.argmin(distances)]

        return nearest_parcel

    # Asumiendo que movimientos en diagonal no están permitidos. Si sí, cambiar esto a Euclidian 
    def get_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def harvest(self, parcel_pos):
        # Cosechar la parcela en la posición dada
        self.model.state_grid[parcel_pos] = 'harvested'
        self.load += self.p.harvest_amount
        self.model.parcels_ready.remove(parcel_pos)

    def unload(self):
        # Descargar la carga en el punto de descarga
        self.load = 0

    def refuel(self):
        # Recargar combustible en la estación de recarga
        self.fuel_level = self.p.max_fuel
    
    def a_star_path(self, start, goal):
        # A* algorithm implementation
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return [start]  # Return a path containing only the start if no path found

    def heuristic(self, pos1, pos2):
        # Manhattan distance heuristic
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_neighbors(self, pos):
    # Get neighboring positions (up, down, left, right)
        neighbors = []
        x, y = pos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (x + dx, y + dy)
            if (0 <= neighbor[0] < self.grid.shape[0] and 
                0 <= neighbor[1] < self.grid.shape[1] and 
                neighbor not in self.grid.positions):  # Check that the cell is empty
                neighbors.append(neighbor)
        return neighbors


    def reconstruct_path(self, came_from, current):
        # Reconstruct the path from start to goal
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Return reversed path
