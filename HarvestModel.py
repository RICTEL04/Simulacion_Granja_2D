import agentpy as ap
import numpy as np
import random
import matplotlib.pyplot as plt
from TractorAgent import TractorAgent 

# Definir la clase del modelo
class HarvestModel(ap.Model):

    def setup(self):
        # Set up the grid with a perimeter and a harvestable inner area
        self.grid = ap.Grid(self, [self.p.field_size, self.p.field_size], track_empty=True, track_agents=True)
        
        # Initialize the grid state
        self.state_grid = np.full(self.grid.shape, 'empty', dtype=object)
        self.parcels_ready = []

        # Define the perimeter width
        perimeter_width = 1

        # Populate the inner area with crops, leaving the perimeter empty
        for x in range(perimeter_width, self.grid.shape[0] - perimeter_width):
            for y in range(perimeter_width, self.grid.shape[1] - perimeter_width):
                if random.random() < 0.9:
                    self.state_grid[x, y] = 'ready_to_harvest'
                    self.parcels_ready.append((x, y))
                else:
                    self.state_grid[x, y] = 'empty'

        # Gather perimeter cells
        perimeter_cells = []
        grid_size = self.grid.shape[0]

        # Top and bottom rows
        for x in range(grid_size):
            perimeter_cells.append((x, 0))                # Top row
            perimeter_cells.append((x, grid_size - 1))    # Bottom row

        # Left and right columns
        for y in range(1, grid_size - 1):  # Avoid adding corners twice
            perimeter_cells.append((0, y))                # Left column
            perimeter_cells.append((grid_size - 1, y))    # Right column

        # Place tractors randomly on the perimeter
        self.tractors = ap.AgentList(self, self.p.num_tractors, TractorAgent)
        tractor_positions = random.sample(perimeter_cells, len(self.tractors))
        self.grid.add_agents(self.tractors, positions=tractor_positions)

        for tractor in self.tractors:
            if tractor in self.grid.positions:
                print(f"Tractor {tractor} placed at {self.grid.positions[tractor]}")
            else:
                print(f"Warning: Tractor {tractor} was not assigned a position.")

        self.random.seed(self.p.seed)


        # Set a refuel station at a random perimeter position
        self.refuel_station = random.choice(perimeter_cells)
        self.state_grid[self.refuel_station] = 'refuel_station'  # Mark in state grid

        print(f"Refuel station set at: {self.refuel_station}")

        # Set an unload point on the opposite side of the grid
        self.unload_point = random.choice(perimeter_cells)
        self.state_grid[self.unload_point] = 'unload_point'  # Mark in state grid

        print(f"Unload point set at: {self.unload_point}")

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
    
    def save_q_tables(self):
        for tractor in self.tractors:
            tractor.save_q_table()


    def end(self):
        # Al final de la simulación
        # After running the simulation, plot data
        self.plot_tractor_data()
        total_harvested = np.sum(self.state_grid == 'harvested')
        self.report('Total parcels harvested', total_harvested)
