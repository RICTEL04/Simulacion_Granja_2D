#!/bin/bash

# Número de simulaciones que deseas ejecutar
num_simulaciones=20

for ((i=1; i<=num_simulaciones; i++))
do
    echo "Ejecutando simulación número $i"
    python3 simulation.py
done

