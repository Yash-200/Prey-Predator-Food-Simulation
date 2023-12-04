# Simulation of Prey-Predator Dynamics with Neural Networks
This Python project simulates the dynamics between prey and predators using neural networks implemented with TensorFlow and Pygame. It employs a genetic algorithm to evolve the neural networks controlling the behavior of prey and predators in a simulated environment.

# Overview
The simulation involves entities existing in a 2D environment where preys (green) attempt to survive while being hunted by predators (blue). Both preys and predators are controlled by neural networks that determine their movement and decision-making processes.

# Features
Entities:
Preys: Green entities that seek food, reproduce, and avoid predators to survive.
Predators: Blue entities that hunt and consume preys for energy to survive and reproduce.
Food: Scattered throughout the environment, providing energy to preys on consumption.
Neural Networks:

# Prey Neural Network: Controls movement based on sensory inputs (rays detecting food and predators).
Predator Neural Network: Dictates predator behavior based on the presence of preys and food.
Genetic Algorithm:Utilizes a genetic algorithm to evolve neural networks over generations, improving survival and reproduction traits.

# Requirements
Python 3.11/n

Pygame

TensorFlow

NumPy

Matplotlib

# Setup and Execution
Install the required dependencies.
Run the Python script main.py.
Observe the simulation in the Pygame window.
The simulation will run for multiple generations, evolving the entities' behaviors.
Usage

main.py: The main script that initializes the simulation environment and executes the simulation loop.
prey_nn.h5 and predator_nn.h5: Saved neural network models after the simulation.
Statistics: Contains class methods for collecting and plotting simulation statistics.


License
This project is licensed under MIT License.

