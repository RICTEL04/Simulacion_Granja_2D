import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

def load_obstacles_from_file(filename, obstacles_list):
    """
    Carga obstáculos desde un archivo de texto y los agrega a la lista de obstáculos.
    
    Formato del archivo:
    Cada línea contiene 4 valores separados por comas que representan las coordenadas (x1, x2, x3, x4)
    y (y1, y2, y3, y4) de un obstáculo.

    Ejemplo:
    0.44, -0.08, 0.21, 0.74
    0.60, 0.30, -0.22, 0.07

    Args:
    - filename: Nombredel archivo a cargar.
    - obstacles_list: Lista existente de obstáculos a la que se añadirán los nuevos.
    """
    try:
        with open(filename, 'r') as file:
            x =  [float(a)  for a in file.readline().split(',')]
            y =  [float(a) for a in file.readline().split(',')]
            obstacles_list.append([[x[0],x[1],x[2],x[3]],[y[0],y[1],y[2],y[3]]]) #y
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")


#Targets

target_x = [-0.86717069892473,-0.277318548387096,0.286122311827957,-1.01683467741935,0.673487903225808,-1.37778897849462,1.54506048387097]
target_y = [-0.356552419354838,0.550235215053764,-0.497412634408602,1.52745295698925,0.629469086021506,-1.36898521505376,-0.999227150537633]

# Cargar datos iniciales
car_positions_x = [-3, 0]
car_positions_y = [-1.5, -2]

# Puntos de los caminos para ambos carros (puedes ajustar estos puntos según tus necesidades)
car_paths = [
    [(-3, -1.5),                
     (target_x[5], target_y[5]),
     (0,-1.5),                      
     (target_x[2],target_y[2]),
     (target_x[1],target_y[1]),
     (target_x[0],target_y[0]),
     (target_x[3],target_y[3]),
     (target_x[4],target_y[4]),
     (target_x[6],target_y[6])],



    [(0, -2), 
     (1.4,-1.5),
     (target_x[6],target_y[6]),
     (1.6,0.5),
     (target_x[4],target_y[4]),
     (-0,0.8),
     (-0.5, 1.2),
     (-0.5, 1.5),
     (target_x[3],target_y[3]),
     (-0.5,1.5),
     (-0.5,1.2),
     (target_x[1],target_y[1]), 
     (0,0.2),
     (0,-0.4),
     (target_x[2],target_y[2]),
     (0,-1),
     (-1, -0.7),
     (target_x[0],target_y[0]),
     (-2,-0.5),
     (target_x[5],target_y[5])]
]

# Obstáculos (x1, x2, x3, x4) y (y1, y2, y3, y4)
obstacles = []

file_ob  = [f"Obstacle_{x}.txt" for x in range(1,7) ]
[load_obstacles_from_file(x, obstacles) for x in file_ob ]


# Configuración de parámetros
time_step = 1  # Simular un segundo por iteración
car_speed = 0.1  # Velocidad de los carros

# Función para calcular la distancia euclidiana
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Simulación de movimiento
positions = []
stop_flags = [False, False]  # Si un carro colisiona, se detiene

# Configurar la visualización
fig, ax = plt.subplots(figsize=(6,4))
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 2)

# Dibujar obstáculos
for i, obs in enumerate(obstacles):
    x, y = obs
    # Cerrar el polígono conectando el último punto con el primero
    x_closed = x + [x[0]]
    y_closed = y + [y[0]]

    l = ["A",'B','C','D',"A","B"]
    v = ["a","b","c","d","a"]
    for j,(xx, yy) in enumerate(zip(x,y)):
        print(f"{l[j]}{i}=({xx},{yy})")
    print(f"tl{i} = Polígono(A{i}, B{i}, C{i}, D{i})")
    for j in range(i):
        print(f"{v[j]}{i} = Segmento({l[j]}, {l[j+1]}, tl{i})")
    
    ax.plot(x_closed, y_closed, 'r-', linewidth=3, alpha=1)  # Líneas rojas
    
# Variables para la animación
car_scatters = [ax.plot([], [], 'bo', markersize=8)[0] for _ in range(2)]

#Objetivos de la animacion
for i, (x,y) in enumerate(zip(target_x, target_y)):
    print(f"OBJ{i} = ({x},{y})")
    ax.plot(x, y, 'go', markersize=6)
    ax.text(x -0.1,y + 0.15,i, color= 'red', fontsize= 10)

#Camino
# Trazar el camino de car_paths[0] con líneas
path_x_0, path_y_0 = zip(*car_paths[0])  # Extraer coordenadas x e y del primer camino
ax.plot(path_x_0, path_y_0, 'k-', linewidth=1.5, label="Camino Carro 0", alpha = 0.3)  # Línea negra

for i,(x,y) in enumerate(car_paths[0]):
    ax.text(x, y, i,fontsize=10, color = "black")

# Trazar el camino de car_paths[1] con líneas
path_x_1, path_y_1 = zip(*car_paths[1])  # Extraer coordenadas x e y del segundo camino
ax.plot(path_x_1, path_y_1, 'b-', linewidth=1.5, label="Camino Carro 1", alpha = 0.3)  # Línea azul

for i, (x,y) in enumerate(car_paths[1]):
    ax.text(x  + 0.15,y, i, fontsize=10, color= "blue")

def update(frame):
    global car_positions_x, car_positions_y, stop_flags
    positions_frame = []
    for i in range(2):  # Para ambos carros
        if stop_flags[i]:
            positions_frame.append((car_positions_x[i], car_positions_y[i]))
            continue
        
        current_position = (car_positions_x[i], car_positions_y[i])
        target_position = car_paths[i][0]  # Siguiente punto objetivo
        
        # Calcular la dirección hacia el siguiente punto
        direction = np.array(target_position) - np.array(current_position)
        distance_to_target = np.linalg.norm(direction)
        
        if distance_to_target < car_speed:
            # Mover al punto y avanzar al siguiente objetivo
            car_positions_x[i], car_positions_y[i] = target_position
            if len(car_paths[i]) > 1:
                car_paths[i].pop(0)
            else:
                stop_flags[i] = True
        else:
            # Normalizar dirección y mover
            direction = direction / distance_to_target
            car_positions_x[i] += direction[0] * car_speed
            car_positions_y[i] += direction[1] * car_speed
        
        # Verificar colisiones
        for obs in obstacles:
            x, y = obs
            if x[0] <= car_positions_x[i] <= x[2] and y[0] <= car_positions_y[i] <= y[2]:
                stop_flags[i] = True
        
        for j in range(2):  # Checar colisión con el otro carro
            if i != j and distance((car_positions_x[i], car_positions_y[i]), 
                                    (car_positions_x[j], car_positions_y[j])) < 0.2:
                stop_flags[i] = True
                stop_flags[j] = True
        
        positions_frame.append((car_positions_x[i], car_positions_y[i]))
    
    # Actualizar posiciones en la gráfica
    for i, scatter in enumerate(car_scatters):
        scatter.set_data(positions_frame[i][0], positions_frame[i][1])
    
    return car_scatters

# Configuración de la animación
ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

# Guardar el resultado
ani.save("car_simulation.mp4", writer="ffmpeg", fps=5)
plt.close(fig)

