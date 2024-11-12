import agentpy as ap
import numpy as np
import random
import matplotlib.pyplot as plt
from TractorAgent import TractorAgent 

# Definir la clase del modelo
class HarvestModel(ap.Model):

    def setup(self):
        # Configuración de la cuadrícula
        self.grid = ap.Grid(self, [self.p.field_size, self.p.field_size], track_empty=True, track_agents=True)
        
        # Inicializar el estado de la cuadrícula
        self.state_grid = np.full(self.grid.shape, 'empty', dtype=object)
        self.parcels_ready = []

        # Definir el ancho del perímetro
        perimeter_width = 1

        # Población del área interior con cultivos
        for x in range(perimeter_width, self.grid.shape[0] - perimeter_width):
            for y in range(perimeter_width, self.grid.shape[1] - perimeter_width):
                if random.random() < 0.9:
                    self.state_grid[x, y] = 'ready_to_harvest'
                    self.parcels_ready.append((x, y))
                else:
                    self.state_grid[x, y] = 'empty'

        # Recolectar celdas del perímetro
        perimeter_cells = []
        grid_size = self.grid.shape[0]

        # Fila superior e inferior
        for x in range(grid_size):
            perimeter_cells.append((x, 0))                # Fila superior
            perimeter_cells.append((x, grid_size - 1))    # Fila inferior

        # Columnas izquierda y derecha
        for y in range(1, grid_size - 1):  # Evitar duplicar esquinas
            perimeter_cells.append((0, y))                # Columna izquierda
            perimeter_cells.append((grid_size - 1, y))    # Columna derecha

        # Colocar tractores aleatoriamente en el perímetro
        self.tractors = ap.AgentList(self, self.p.num_tractors, TractorAgent)
        tractor_positions = random.sample(perimeter_cells, len(self.tractors))
        self.grid.add_agents(self.tractors, positions=tractor_positions)

        for i, tractor in enumerate(self.tractors):
            if tractor in self.grid.positions:
                print(f"Tractor {tractor} placed at {self.grid.positions[tractor]}")
            else:
                print(f"Warning: Tractor {tractor} was not assigned a position.")

            # Asignación de secciones con un margen para cada tractor
            tractor.assign_section(i, margin=2)  # Ajusta el margen según lo necesario

        # Configuración de la estación de recarga y el punto de descarga
        self.refuel_station = random.choice(perimeter_cells)
        self.state_grid[self.refuel_station] = 'refuel_station'
        print(f"Refuel station set at: {self.refuel_station}")

        self.unload_point = random.choice(perimeter_cells)
        self.state_grid[self.unload_point] = 'unload_point'
        print(f"Unload point set at: {self.unload_point}")


    def step(self):
        for tractor in self.tractors:
            tractor.move()
        
        self.record('Parcels left to harvest', len(self.parcels_ready))
        
        if len(self.parcels_ready) == 0 and all(self.grid.positions[tractor] == self.unload_point for tractor in self.tractors):
            self.stop()

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

    def plot_tractor_data(self):
        for i, tractor in enumerate(self.tractors):
            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
            
            # Fuel level over time
            ax[0].plot(tractor.fuel_levels, label='Nivel de Combustible')
            ax[0].set_title(f'Tractor {i + 1} - Nivel de Combustible')
            ax[0].set_xlabel('Paso de Tiempo')
            ax[0].set_ylabel('Combustible')
            
            # Load over time
            ax[1].plot(tractor.loads, label='Carga')
            ax[1].set_title(f'Tractor {i + 1} - Carga')
            ax[1].set_xlabel('Paso de Tiempo')
            ax[1].set_ylabel('Carga')
            
            plt.tight_layout()

            # Save the figure as a JPEG
            filename = f"tractor_{i + 1}_data.jpeg"
            plt.savefig(filename, format="jpeg")
            print(f"Saved plot for Tractor {i + 1} as {filename}")

            # Optionally, close the figure to free up memory if running many tractors
            plt.close(fig)

    def end(self):
        # Al final de la simulación
        # After running the simulation, plot data
        self.plot_tractor_data()
        total_harvested = np.sum(self.state_grid == 'harvested')
        self.report('Total parcels harvested', total_harvested)