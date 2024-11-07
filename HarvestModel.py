import agentpy as ap
import numpy as np
import random
from TractorAgent import TractorAgent 

# Definir la clase del modelo
class HarvestModel(ap.Model):

    def setup(self):
        # Crear la cuadrícula con track_agents=True
        self.grid = ap.Grid(self, [self.p.field_size, self.p.field_size], track_empty=True, track_agents=True)
        # self.grid = ap.Grid(self, [20, 20], track_empty=True, track_agents=True)
        
        ## CULTIVOS
        # Crear la matriz de estados de las celdas
        self.state_grid = np.full(self.grid.shape, 'empty', dtype=object)

        # Se cambio esto ya que self.grid.positions es para agentes
        # Establecer aleatoriamente algunas parcelas como 'ready_to_harvest'
        self.parcels_ready = []
        for x in range(self.grid.shape[0]):  # Recorrer las filas (y)
            for y in range(self.grid.shape[1]):  # Recorrer las columnas (x)
                # Asignar 'ready_to_harvest' con un 90% de probabilidad y 'empty' con un 10%
                if random.random() < 0.9:
                    self.state_grid[x, y] = 'ready_to_harvest'
                    self.parcels_ready.append((x, y))  # Add parcel position to parcels_ready

                else:
                    self.state_grid[x, y] = 'empty'


        # Crear los tractores
        self.tractors = ap.AgentList(self, self.p.num_tractors, TractorAgent)
        self.grid.add_agents(self.tractors, random=True)  # `random=True` will place agents in random empty cells

         # Confirm placement by printing the positions of each tractor
        for tractor in self.tractors:
            if tractor in self.grid.positions:
                print(f"Tractor {tractor} placed at {self.grid.positions[tractor]}")
            else:
                print(f"Warning: Tractor {tractor} was not assigned a position.")


        # Establecer la semilla aleatoria para reproducibilidad
        self.random.seed(self.p.seed)

        # Definir el punto de descarga y la estación de recarga
        self.unload_point = (0, 0)  # Puede ajustarse a cualquier posición
        self.refuel_station = (self.p.field_size - 1, self.p.field_size - 1)  # Esquina opuesta

    def step(self):
        # Eventos aleatorios (crecimiento y marchitamiento de cultivos)
        # Por ahora inhabilitados los random events, por cambiarse => self.random_events()
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
