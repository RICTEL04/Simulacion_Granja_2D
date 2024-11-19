
import agentpy as ap
import numpy as np
import random
import heapq
from collections import defaultdict
import os
import pickle

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

        # Agregar más si es necesario para graficar
        self.fuel_levels = []
        self.loads = []

        # Q learning
        q_table_filename = f'q_table_{self.id}.pkl'
        if os.path.exists(q_table_filename):
            with open(q_table_filename, 'rb') as f:
                self.q_table = defaultdict(float, pickle.load(f))
            print(f"Tractor {self.id}: Tabla Q cargada desde {q_table_filename}")
        else:
            self.q_table = defaultdict(float)
            print(f"Tractor {self.id}: No se encontró una tabla Q previa, iniciando nueva tabla")

        self.alpha = 0.1  # Tasa de aprendizaje
        self.gamma = 0.9  # Factor de descuento
        self.epsilon = 0.1  # Probabilidad para la política epsilon-greedy
        self.last_state = None
        self.last_action = None
    
    def move(self):
        # Guardar los niveles actuales de combustible y carga para graficar
        self.fuel_levels.append(self.fuel_level)
        self.loads.append(self.load)

        # Obtener el estado actual
        state = self.get_state()
        if state is None:
            return

        # Política epsilon-greedy
        if random.random() < self.epsilon:
            action = random.choice(self.get_possible_actions())
        else:
            q_values = [self.q_table[(state, a)] for a in self.get_possible_actions()]
            max_q = max(q_values)
            # Puede haber múltiples acciones con el mismo valor Q
            actions_with_max_q = [a for a, q in zip(self.get_possible_actions(), q_values) if q == max_q]
            action = random.choice(actions_with_max_q)

        # Ejecutar la acción y obtener la recompensa y el nuevo estado
        reward, next_state = self.take_action(action)

        # Actualizar la tabla Q
        if self.last_state is not None and self.last_action is not None:
            old_value = self.q_table[(self.last_state, self.last_action)]
            next_max = max([self.q_table[(next_state, a)] for a in self.get_possible_actions()])
            new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
            self.q_table[(self.last_state, self.last_action)] = new_value

        # Actualizar el estado y acción anteriores
        self.last_state = state
        self.last_action = action
    
    def attempt_move(self, next_pos):
        if (0 <= next_pos[0] < self.grid.shape[0] and
            0 <= next_pos[1] < self.grid.shape[1]):
            if next_pos in self.grid.positions.values():
                return -100  # Penalización por colisión
            else:
                self.grid.move_to(self, next_pos)
                self.fuel_level -= self.fuel_consumption_rate
                return 0  # Movimiento válido sin recompensa adicional
        else:
            return -10  # Penalización por intentar salir del grid


    def take_action(self, action):
        current_pos = self.grid.positions.get(self, None)
        if current_pos is None:
            return -10, None  # Penalización por no tener posición

        reward = -1  # Penalización por tiempo para incentivar eficiencia

        if action == 'move_up':
            next_pos = (current_pos[0] - 1, current_pos[1])
            reward += self.attempt_move(next_pos)
        elif action == 'move_down':
            next_pos = (current_pos[0] + 1, current_pos[1])
            reward += self.attempt_move(next_pos)
        elif action == 'move_left':
            next_pos = (current_pos[0], current_pos[1] - 1)
            reward += self.attempt_move(next_pos)
        elif action == 'move_right':
            next_pos = (current_pos[0], current_pos[1] + 1)
            reward += self.attempt_move(next_pos)
        elif action == 'harvest':
            if self.model.state_grid[current_pos] == 'ready_to_harvest':
                self.harvest(current_pos)
                reward += 10  # Recompensa por cosechar
            else:
                reward -= 5  # Penalización por intentar cosechar donde no hay cultivo
        elif action == 'unload':
            if current_pos == self.model.unload_point and self.load > 0:
                self.unload()
                reward += 5  # Recompensa por descargar
            else:
                reward -= 5  # Penalización por intentar descargar en lugar incorrecto
        elif action == 'refuel':
            if current_pos == self.model.refuel_station:
                self.refuel()
                reward += 5  # Recompensa por recargar combustible
            else:
                reward -= 5  # Penalización por intentar recargar en lugar incorrecto

        # Obtener el nuevo estado
        next_state = self.get_state()
        return reward, next_state


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
    
    # No consideré la posibilidad de choques con otros agentes
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
    
    def get_state(self):
        current_pos = self.grid.positions.get(self, None)
        if current_pos is None:
            return None

        # Discretizar el nivel de combustible y carga
        fuel_level = 'High' if self.fuel_level > self.p.max_fuel * 0.5 else 'Low'
        load_level = 'Full' if self.load >= self.capacity else 'NotFull'

        # Detectar cultivos en celdas adyacentes
        directions = ['up', 'down', 'left', 'right']
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        crops = []
        for move in moves:
            neighbor_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
            if (0 <= neighbor_pos[0] < self.grid.shape[0] and
                0 <= neighbor_pos[1] < self.grid.shape[1]):
                state = self.model.state_grid[neighbor_pos]
                crops.append(1 if state == 'ready_to_harvest' else 0)
            else:
                crops.append(-1)  # Indica borde del grid

        # Presencia de otros tractores en celdas adyacentes
        tractors = []
        for move in moves:
            neighbor_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
            if neighbor_pos in self.grid.positions.values():
                tractors.append(1)
            else:
                tractors.append(0)

        # Construir el estado como una tupla
        state = (fuel_level, load_level, tuple(crops), tuple(tractors))
        return state

    def get_possible_actions(self):
        actions = ['move_up', 'move_down', 'move_left', 'move_right', 'harvest', 'unload', 'refuel']
        return actions
    
    def save_q_table(self):
        q_table_filename = f'q_table_{self.id}.pkl'
        with open(q_table_filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

