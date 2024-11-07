# Importar los módulos necesarios
import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation  # Para guardar la animación

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

        # Obtener la posición actual del tractor
        if self in self.grid.positions:  # Verificar si el tractor tiene una posición asignada
            current_pos = self.grid.positions[self]
        else:
            return  # Si el tractor no tiene posición, salir de la función

        # Encontrar la ruta más corta hacia el objetivo, evitando otros agentes
        path = self.grid.shortest_path(current_pos, target, avoid_agents=True)
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
        if self in self.grid.positions:  # Verificar si el tractor tiene una posición asignada
            current_pos = self.grid.positions[self]
        else:
            return None  # Si el tractor no tiene posición, regresar None
        distances = [self.grid.distance(current_pos, p) for p in parcels]
        nearest_parcel = parcels[np.argmin(distances)]
        return nearest_parcel

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

# Definir la clase del modelo
class HarvestModel(ap.Model):

    def setup(self):
        # Crear la cuadrícula con track_agents=True
        self.grid = ap.Grid(self, [self.p.field_size, self.p.field_size], track_empty=True, track_agents=True)

        # Crear la matriz de estados de las celdas
        self.state_grid = np.full(self.grid.shape, 'empty', dtype=object)

        # Establecer aleatoriamente algunas parcelas como 'ready_to_harvest'
        self.parcels_ready = []
        for pos in self.grid.positions:
            if random.random() < self.p.initial_ready_fraction:
                self.state_grid[pos] = 'ready_to_harvest'
                self.parcels_ready.append(pos)
            else:
                self.state_grid[pos] = 'empty'

        # Crear los tractores
        self.tractors = ap.AgentList(self, self.p.num_tractors, TractorAgent)
        # Colocar los tractores aleatoriamente en la cuadrícula
        empty_cells = [pos for pos in self.grid.empty if self.grid.agents[pos] is None]
        num_tractors = min(len(empty_cells), len(self.tractors))  # Limitar el número de tractores si es necesario
        tractor_positions = self.random.sample(empty_cells, num_tractors)
        self.grid.add_agents(self.tractors[:num_tractors], positions=tractor_positions)

        # Establecer la semilla aleatoria para reproducibilidad
        self.random.seed(self.p.seed)

        # Definir el punto de descarga y la estación de recarga
        self.unload_point = (0, 0)  # Puede ajustarse a cualquier posición
        self.refuel_station = (self.p.field_size - 1, self.p.field_size - 1)  # Esquina opuesta

    def step(self):
        # Eventos aleatorios (crecimiento y marchitamiento de cultivos)
        self.random_events()
        # Cada tractor realiza su movimiento
        for tractor in self.tractors:
            tractor.move()
        # Actualizar los datos recolectados
        self.record('Parcels left to harvest', len(self.parcels_ready))

    def random_events(self):
        # Simular eventos aleatorios que afectan al campo
        for pos in self.grid.positions:
            state = self.state_grid[pos]
            if state == 'empty' and random.random() < self.p.growth_chance:
                # La parcela vacía tiene una probabilidad de crecer un cultivo
                self.state_grid[pos] = 'ready_to_harvest'
                self.parcels_ready.append(pos)
            elif state == 'ready_to_harvest' and random.random() < self.p.wither_chance:
                # El cultivo listo para cosechar tiene una probabilidad de marchitarse
                self.state_grid[pos] = 'empty'
                if pos in self.parcels_ready:
                    self.parcels_ready.remove(pos)

    def end(self):
        # Al final de la simulación
        total_harvested = np.sum(self.state_grid == 'harvested')
        self.report('Total parcels harvested', total_harvested)

# Definir los parámetros
parameters = {
    'field_size': 20,
    'num_tractors': 3,
    'capacity': 10,
    'max_fuel': 100,
    'fuel_consumption_rate': 1,
    'fuel_threshold': 10,
    'speed': 1,
    'harvest_amount': 1,
    'initial_ready_fraction': 0.2,
    'breakdown_chance': 0.01,
    'repair_steps': 3,
    'growth_chance': 0.01,
    'wither_chance': 0.005,
    'steps': 100,
    'seed': 42
}

# Crear el modelo (no ejecutar model.run())
model = HarvestModel(parameters)

# Visualización
def plot_field(model, ax):
    grid = model.grid
    state_colors = {'empty': 0, 'ready_to_harvest': 1, 'harvested': 2}
    parcel_grid = np.zeros(grid.shape)
    for pos in grid.positions:
        state = model.state_grid[pos]
        parcel_grid[pos] = state_colors[state]
    # Usar el mapa de colores correctamente
    cmap = plt.cm.YlGn
    ax.imshow(parcel_grid.T, cmap=cmap, origin='lower')
    x_coords = [model.grid.positions[agent][0] for agent in model.tractors if agent in model.grid.positions]
    y_coords = [model.grid.positions[agent][1] for agent in model.tractors if agent in model.grid.positions]
    ax.scatter(x_coords, y_coords, c='red', s=100, label='Tractores')
    ax.legend(loc='upper right')
    ax.set_title(f"Paso {model.t}")
    ax.set_xticks([])
    ax.set_yticks([])

# Crear la animación y guardarla como archivo MP4
fig, ax = plt.subplots(figsize=(6,6))
animation = ap.animate(model, fig, ax, plot_field)

# Guardar la animación como un archivo MP4
animation.save("harvest_simulation.mp4", writer="ffmpeg")

print("Animación guardada como 'harvest_simulation.mp4'")
