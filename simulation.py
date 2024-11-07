# Importar los módulos necesarios
import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import random
from HarvestModel import HarvestModel

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
    ax.clear()  # Limpiar el eje al inicio

    # Generate a color-mapped grid from the `state_grid`
    state_grid = np.vectorize({
        'empty': 0,
        'ready_to_harvest': 1,
        'harvested': 2
    }.get)(model.state_grid)
    
    # Plot the grid with colors for each state
    ap.gridplot(state_grid, cmap='YlGn', ax=ax)

    ##Cultivos
    state_to_int = {
        'empty': 0,
        'ready_to_harvest': 1,
        'harvested': 2
    }
    # Convertir la matriz de estados a valores numéricos
    state_grid_numeric = np.vectorize(state_to_int.get)(model.state_grid).astype(int)
    #print(model.state_grid)
    # Crear el cmap con los colores correspondientes
    cmap = mcolors.ListedColormap(['#d2b48c', 'green', 'gray'])  
    bounds = [0, 1, 2, 3]  # Definir límites para los valores de los estados
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Mostrar la cuadrícula con los colores correspondientes
    ax.imshow(state_grid_numeric, cmap=cmap, norm=norm)
    

    ##Tractores
    # Add tractor positions to the plot
    x_coords = [model.grid.positions[agent][0] for agent in model.tractors if agent in model.grid.positions]
    y_coords = [model.grid.positions[agent][1] for agent in model.tractors if agent in model.grid.positions]
    ax.scatter(x_coords, y_coords, c='red', s=100, label='Tractores')
    
    ##Cuadricula
    # Añadir cuadrícula visible y ajustar los ticks
    ax.set_xticks(np.arange(-0.5, parameters['field_size'], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, parameters['field_size'], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    # Add a title and other plot details
    ax.legend(loc='upper right')
    ax.set_title(f"Paso {model.t}")
    ax.set_xticks([])
    ax.set_yticks([])


fig, ax = plt.subplots(figsize=(6,6))
animation = ap.animate(model, fig, ax, plot_field)

# Guardar la animación como un archivo MP4
animation.save("harvest_simulation.mp4", writer="ffmpeg")

print("Animación guardada como 'harvest_simulation.mp4'")
