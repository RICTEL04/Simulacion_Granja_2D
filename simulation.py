# Importar los módulos necesarios
import agentpy as ap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import random
from HarvestModel import HarvestModel

# Definir los parámetros
parameters = {
    'field_size': 50,
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
model = HarvestModel(parameters)
# model.run()  # Descomentar esto para generar graficas

# Visualización
def plot_field(model, ax):
    ax.clear()  # Clear the axis for each frame

    # Map the state_grid to integer values for colors
    state_to_int = {
        'empty': 0,
        'ready_to_harvest': 1,
        'harvested': 2,
        'refuel_station': 3,
        'unload_point': 4
    }
    state_grid_numeric = np.vectorize(state_to_int.get)(model.state_grid)

    # Create a color map for the states
    cmap = mcolors.ListedColormap(['#d2b48c', 'green', 'gray', 'blue', 'purple'])
    bounds = [0, 1, 2, 3, 4, 5]  # Define boundaries for each color
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot the grid with updated colors
    ax.imshow(state_grid_numeric, cmap=cmap, norm=norm)

    # Plot tractor positions
    x_coords = [model.grid.positions[agent][1] for agent in model.tractors if agent in model.grid.positions]
    y_coords = [model.grid.positions[agent][0] for agent in model.tractors if agent in model.grid.positions]
    ax.scatter(x_coords, y_coords, c='red', s=100, label='Tractores')

    # Custom legend for the plot
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Parcela Vacía', markersize=10, 
                   markerfacecolor='#d2b48c'),
        plt.Line2D([0], [0], marker='s', color='w', label='Lista para Cosechar', markersize=10, 
                   markerfacecolor='green'),
        plt.Line2D([0], [0], marker='s', color='w', label='Cosechada', markersize=10, 
                   markerfacecolor='gray'),
        plt.Line2D([0], [0], marker='s', color='w', label='Punto de Recarga', markersize=10, 
                   markerfacecolor='blue'),
        plt.Line2D([0], [0], marker='s', color='w', label='Punto de Descarga', markersize=10, 
                   markerfacecolor='purple'),
        plt.Line2D([0], [0], marker='o', color='w', label='Tractor', markersize=10, 
                   markerfacecolor='red')
    ]

    # Add the legend to the plot
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Leyenda")

    # Add grid lines and other plot details
    ax.set_xticks(np.arange(-0.5, model.p['field_size'], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, model.p['field_size'], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    # Add title and remove default ticks
    ax.set_title(f"Step {model.t}")
    ax.set_xticks([])
    ax.set_yticks([])

# Create the animation using the tracked data
fig, ax = plt.subplots(figsize=(8,6))
fig.tight_layout(rect=[0, 0, 0.85, 1])  # Add space on the right for the legend

animation = ap.animate(model, fig, ax, plot_field)

# Save the animation as an MP4 file with a duration that reflects the number of frames
animation.save("harvest_simulation.mp4", writer="ffmpeg", fps=5)  # Adjust fps as needed for smoother video

# Aprender y guardar las tablas Q para la siguiente interacion
model.save_q_tables()

print("Animación guardada como 'harvest_simulation.mp4'")
