import math
import pygame
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define constants
MAP_WIDTH = 800
MAP_HEIGHT = 600
FOOD_COLOR = (0, 255, 0)
PREY_COLOR = (255, 0, 0)
PREDATOR_COLOR = (0, 0, 255)
PREY_ENERGY = 50
PREDATOR_ENERGY = 50
PREY_REPRODUCTION_THRESHOLD = 100
PREDATOR_REPRODUCTION_THRESHOLD = 100
FOOD_SIZE = 3
PREY_SIZE = 5
PREDATOR_SIZE = 5
FOOD_ENERGY = 100
GENERATIONS = 50
PREY_RAY_COUNT = 20
PREDATOR_RAY_COUNT = 20
PREY_RAY_DISTANCE = 40
PREDATOR_RAY_DISTANCE = 55

# Initialize Pygame
pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((MAP_WIDTH, MAP_HEIGHT))
clock = pygame.time.Clock()

# Neural Network setup
def create_neural_network(input_size, output_size):
    """
    Create a neural network model with the specified input and output sizes.
    """
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_size,)),
        Dense(16, activation='relu'),
        Dense(output_size, activation='tanh')  # Output actions: move_x and move_y
    ])
    model.compile(optimizer='adam', loss='mse')
    model.build((None, input_size))  # Initialize model weights
    return model

# Prey and Predator Neural Networks
prey_nn = create_neural_network(PREY_RAY_COUNT * 2 + 2, 2)  # Two outputs for prey movement
predator_nn = create_neural_network(PREDATOR_RAY_COUNT * 2 + 2, 2)  # Two outputs for predator movement



class Food:
    """
    Represents a food object in the simulation.
    """
    def __init__(self):
        self.x = random.uniform(0, MAP_WIDTH)
        self.y = random.uniform(0, MAP_HEIGHT)
        self.energy = FOOD_ENERGY
        self.size = FOOD_SIZE

    def draw(self):
        """
        Draw the food object on the screen.
        """
        pygame.draw.circle(screen, FOOD_COLOR, (int(self.x), int(self.y)), FOOD_SIZE)

def genetic_algorithm(entities, selection_rate=0.1, mutation_rate=0.01, mutation_strength=0.1):
    """
    Perform a genetic algorithm on the entities to evolve their neural networks.
    """
    # Selection
    entities.sort(key=lambda x: x.energy, reverse=True)  # Sort entities by energy (fitness)
    num_selected = int(len(entities) * selection_rate)
    selected = entities[:num_selected]  # Select a percentage of the top entities

    if not selected:  # Check if selected is empty
        return

    # Crossover (simple cloning in this case, could be improved with actual crossover)
    offspring = []
    for _ in range(len(entities) - num_selected):  # Fill the rest of the population with offspring
        parent = random.choice(selected)  # Choose a parent from the selected individuals
        child = tf.keras.models.clone_model(parent.model)
        child.set_weights(parent.model.get_weights())
        offspring.append(child)

    # Mutation
    for child_model in offspring:
        weights = child_model.get_weights()
        new_weights = []
        for weight in weights:
            if random.random() < mutation_rate:
                weight += np.random.normal(0, mutation_strength, size=weight.shape)
            new_weights.append(weight)
        child_model.set_weights(new_weights)

    # Replace the old population with the new one
    for i in range(num_selected, len(entities)):
        entities[i].model = offspring[i - num_selected]
class Entity:
    """
    Represents a generic entity in the simulation.
    """
    def __init__(self, x, y, color, size, energy, model, rays, ray_distance):
        self.x = x
        self.y = y
        self.color = color
        self.size = size
        self.energy = energy
        self.model = model
        self.rays = rays
        self.ray_distance = ray_distance
        self.ray_angles = np.linspace(0, 2 * math.pi, self.rays, endpoint=False)
        self.speed = 30
        self.last_energy_update = time.time()

    def draw(self):
        """
        Draw the entity on the screen.
        """
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)
        self.draw_rays()

    def draw_rays(self):
        """
        Draw the rays emitted by the entity on the screen.
        """
        for angle in self.ray_angles:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
            ray_end_x = self.x + self.ray_distance * math.cos(angle)
            ray_end_y = self.y + self.ray_distance * math.sin(angle)
            pygame.draw.line(screen, self.color, (int(self.x), int(self.y)), (int(ray_end_x), int(ray_end_y)), 1)

    def cast_ray(self, angle, objects):
        """
        Cast a ray in the specified angle and check for collisions with objects.
        """
        closest_distance = self.ray_distance
        ray_vector = np.array([np.cos(angle), np.sin(angle)])

        for obj in objects:
            obj_pos = np.array([obj.x, obj.y])
            obj_to_ray_start = obj_pos - np.array([self.x, self.y])
            projection_length = np.dot(obj_to_ray_start, ray_vector)

            if 0 <= projection_length < closest_distance:
                distance_to_ray = np.linalg.norm(obj_to_ray_start - projection_length * ray_vector)
                if distance_to_ray < self.size + obj.size:
                    closest_distance = projection_length

        return 1 - (closest_distance / self.ray_distance)

    def reproduce(self, entity_list, entity_class):
        """
        Reproduce the entity by creating an offspring with mutated neural network weights.
        """
        if self.energy >= self.reproduction_energy:
            self.energy /= 2  # Split the energy between parent and offspring
            offspring_x = self.x + random.uniform(-10, 10)  # Offspring position is close to the parent
            offspring_y = self.y + random.uniform(-10, 10)
            offspring = entity_class(offspring_x, offspring_y)
            offspring.model = tf.keras.models.clone_model(self.model)  # Clone the model structure
            offspring.model.set_weights(self.mutate_weights(self.model.get_weights()))  # Set mutated weights
            offspring.energy = self.energy  # Set the energy of the offspring
            entity_list.append(offspring)

    def mutate_weights(self, weights):
        """
        Mutate the weights of the neural network.
        """
        mutation_rate = 0.1
        mutation_strength = 0.1
        new_weights = []
        for weight in weights:
            if np.random.rand() < mutation_rate:
                mutation = np.random.randn(*weight.shape) * mutation_strength
                new_weights.append(weight + mutation)
            else:
                new_weights.append(weight)
        return new_weights

    def collide(self, other_entity):
        """
        Check for collision with another entity.
        """
        distance = math.sqrt((self.x - other_entity.x) ** 2 + (self.y - other_entity.y) ** 2)
        return distance < self.size + other_entity.size

class Prey(Entity):
    """
    Represents a prey entity in the simulation.
    """
    def __init__(self, x, y):
        super().__init__(x, y, PREY_COLOR, PREY_SIZE, PREY_ENERGY, prey_nn, PREY_RAY_COUNT, PREY_RAY_DISTANCE)
        self.reproduction_energy = PREY_REPRODUCTION_THRESHOLD
        # Keep FOV as 360 degrees by using the full circle of angles
        self.ray_angles = np.linspace(0, 2 * math.pi, self.rays, endpoint=False)

    def update(self, foods, preys, predators):
        """
        Update the prey entity's position, energy, and reproduction.
        """
        inputs = self.get_neural_inputs(foods, predators)
        decision = self.model.predict(np.array([inputs]))[0]
        self.move(decision)

        self.energy -= 0.5  # Energy decrement as a cost of living
        if self.energy <= 0:
            preys.remove(self)
        else:
            for food in foods[:]:
                if self.collide(food):
                    self.energy += food.energy
                    foods.remove(food)
                    if self.energy >= self.reproduction_energy:
                        self.reproduce(preys, Prey)

        self.reproduce(preys, Prey)

        for predator in predators[:]:
            if self.collide(predator):
                predator.energy += self.energy
                self.energy = 0
                preys.remove(self)
                break

    def get_neural_inputs(self, foods, predators):
        """
        Get the inputs for the prey's neural network.
        """
        rays_foods = [self.cast_ray(angle, foods) for angle in self.ray_angles]
        rays_predators = [self.cast_ray(angle, predators) for angle in self.ray_angles]
        return rays_foods + rays_predators + [self.energy / PREY_ENERGY, self.ray_distance / PREY_RAY_DISTANCE]



    def move(self, decision):
        """ Move the entity based on the decision made by the neural network. """
        move_x = decision[0] * self.speed
        move_y = decision[1] * self.speed
        new_x = self.x + move_x
        new_y = self.y + move_y

        # Update the entity's position
        self.x = new_x
        self.y = new_y

        # Recalculate the ray angles based on the direction of movement
        direction = np.arctan2(move_y, move_x)
        self.ray_angles = np.linspace(0, 2 * math.pi, self.rays, endpoint=False)

        # Ensure the entity stays within the bounds of the map
        self.x %= MAP_WIDTH
        self.y %= MAP_HEIGHT
    def reproduce(self, entity_list, entity_class):
        """
        Reproduce the entity by creating an offspring with mutated neural network weights.
        """
        if self.energy >= self.reproduction_energy:
            self.energy /= 2  # Split the energy between parent and offspring
            offspring_x = self.x + random.uniform(-10, 10)  # Offspring position is close to the parent
            offspring_y = self.y + random.uniform(-10, 10)
            offspring = entity_class(offspring_x, offspring_y)
            offspring.model = tf.keras.models.clone_model(self.model)  # Clone the model structure
            offspring.model.set_weights(self.mutate_weights(self.model.get_weights()))  # Set mutated weights
            offspring.energy = self.energy  # Set the energy of the offspring
            entity_list.append(offspring)
    
    def mutate_weights(self, weights, mutation_rate=0.1, mutation_strength=0.1):
        """
        Mutate the weights of the neural network.
        """
        new_weights = []
        for weight in weights:
            if np.random.rand() < mutation_rate:
                mutation = np.random.randn(*weight.shape) * mutation_strength
                new_weights.append(weight + mutation)
            else:
                new_weights.append(weight)
        return new_weights

class Predator(Entity):
    """
    Represents a predator entity in the simulation.
    """
    def __init__(self, x, y):
        super().__init__(x, y, PREDATOR_COLOR, PREDATOR_SIZE, PREDATOR_ENERGY, predator_nn, PREDATOR_RAY_COUNT,
                         PREDATOR_RAY_DISTANCE)
        self.reproduction_energy = PREDATOR_REPRODUCTION_THRESHOLD
        self.ray_angles = np.linspace(-math.pi / 3, math.pi / 3, self.rays, endpoint=False)  # Keep FOV as 60 degrees

    def update(self, preys, predators, foods):
        """
        Update the predator entity's position, energy, and reproduction.
        """
        inputs = self.get_neural_inputs(preys, foods)
        decision = self.model.predict(np.array([inputs]))[0]
        self.move(decision)

        self.energy -= 1  # Energy decrement as a cost of living

        if self.energy <= 0:
            predators.remove(self)
        else:
            for prey in preys[:]:
                if self.collide(prey):
                    self.energy += prey.energy
                    preys.remove(prey)
                    if self.energy >= self.reproduction_energy:
                        self.reproduce(predators, Predator)

        self.reproduce(predators, Predator)

        for prey in preys[:]:
            if self.collide(prey):
                self.energy += prey.energy
                preys.remove(prey)
                if self.energy >= self.reproduction_energy:
                    self.reproduce(predators, Predator)

    def get_neural_inputs(self, preys, foods):
        """
        Get the inputs for the predator's neural network.
        """
        rays_preys = [self.cast_ray(angle, preys) for angle in self.ray_angles]
        rays_foods = [self.cast_ray(angle, foods) for angle in self.ray_angles]
        return rays_preys + rays_foods + [self.energy / PREDATOR_ENERGY, self.ray_distance / PREDATOR_RAY_DISTANCE]
    
    def move(self, decision):
        """
        Move the predator entity based on the decision made by the neural network.
        """
        move_x = decision[0] * self.speed
        move_y = decision[1] * self.speed

        new_x = self.x + move_x
        new_y = self.y + move_y

        # Update the entity's position
        self.x = new_x
        self.y = new_y

        # Recalculate the ray angles based on the direction of movement for 60-degree FOV
        direction = np.arctan2(move_y, move_x)
        self.ray_angles = np.linspace(direction - math.pi / 6, direction + math.pi / 6, self.rays, endpoint=False)

        # Ensure the entity stays within the bounds of the map
        self.x %= MAP_WIDTH
        self.y %= MAP_HEIGHT

class Statistics:
    """
    Represents the statistics of the simulation.
    """
    def __init__(self):
        self.time_steps = []
        self.prey_counts = []
        self.predator_counts = []
        self.prey_reproduction_rate = []
        self.predator_reproduction_rate = []
        self.prey_survival_rate = []
        self.predator_survival_rate = []
        self.prey_energy_consumption = []
        self.predator_energy_consumption = []
        self.predator_prey_ratio = []
        self.prey_nn_weights = []
        self.predator_nn_weights = []
        self.predator_avg_reproduction_energy = []
        self.predator_avg_energy = []
        self.prey_avg_speed = []
        self.prey_avg_reproduction_energy = []
        self.prey_avg_energy = []
        self.predator_avg_speed = []
        self.generations = []

    def update(self, current_generation, current_step, preys, predators):
        """
        Update the statistics based on the current state of the simulation.
        """
        self.time_steps.append(current_step)
        self.prey_counts.append(len(preys))
        self.predator_counts.append(len(predators))
        self.generations.append(current_generation)  # Store the current generation

        # Calculate average energy, reproduction energy, and speed for preys
        if preys:
            avg_prey_energy = sum(prey.energy for prey in preys) / len(preys)
            avg_prey_reproduction_energy = sum(prey.reproduction_energy for prey in preys) / len(preys)
            avg_prey_speed = sum(prey.speed for prey in preys) / len(preys)
        else:
            avg_prey_energy = avg_prey_reproduction_energy = avg_prey_speed = 0

        # Calculate average energy, reproduction energy, and speed for predators
        if predators:
            avg_predator_energy = sum(predator.energy for predator in predators) / len(predators)
            avg_predator_reproduction_energy = sum(predator.reproduction_energy for predator in predators) / len(predators)
            avg_predator_speed = sum(predator.speed for predator in predators) / len(predators)
        else:
            avg_predator_energy = avg_predator_reproduction_energy = avg_predator_speed = 0

        # Append the calculated averages to the respective lists
        self.prey_avg_energy.append(avg_prey_energy)
        self.prey_avg_reproduction_energy.append(avg_prey_reproduction_energy)
        self.prey_avg_speed.append(avg_prey_speed)
        self.predator_avg_energy.append(avg_predator_energy)
        self.predator_avg_reproduction_energy.append(avg_predator_reproduction_energy)
        self.predator_avg_speed.append(avg_predator_speed)

    def plot(self):
        """
        Plot the statistics over time.
        """
        fig = plt.figure()

        # Plot for Prey and Predator counts over time
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.plot(self.generations, self.prey_counts, self.predator_counts)
        ax1.set_xlabel('Generations')
        ax1.set_ylabel('Prey Counts')
        ax1.set_zlabel('Predator Counts')
        ax1.set_title('Prey and Predator Counts Over Time')

        # Plot for Prey and Predator average energy over time
        ax2 = fig.add_subplot(232, projection='3d')
        ax2.plot(self.generations, self.prey_avg_energy, self.predator_avg_energy)
        ax2.set_xlabel('Generations')
        ax2.set_ylabel('Prey Average Energy')
        ax2.set_zlabel('Predator Average Energy')
        ax2.set_title('Prey and Predator Average Energy Over Time')

        # Plot for Prey and Predator average reproduction energy over time
        ax3 = fig.add_subplot(233, projection='3d')
        ax3.plot(self.generations, self.prey_avg_reproduction_energy, self.predator_avg_reproduction_energy)
        ax3.set_xlabel('Generations')
        ax3.set_ylabel('Prey Average Reproduction Energy')
        ax3.set_zlabel('Predator Average Reproduction Energy')
        ax3.set_title('Prey and Predator Average Reproduction Energy Over Time')

        # Plot for Prey and Predator average speed over time
        ax4 = fig.add_subplot(234, projection='3d')
        ax4.plot(self.generations, self.prey_avg_speed, self.predator_avg_speed)
        ax4.set_xlabel('Generations')
        ax4.set_ylabel('Prey Average Speed')
        ax4.set_zlabel('Predator Average Speed')
        ax4.set_title('Prey and Predator Average Speed Over Time')

        plt.show()

   
def initialize_pygame():
    """
    Initialize the Pygame library and create the screen and clock objects.
    """
    pygame.init()
    pygame.font.init()                                                                                                                                                                                                                                      
    screen = pygame.display.set_mode((MAP_WIDTH, MAP_HEIGHT))
    clock = pygame.time.Clock()
    return screen, clock

def create_entities(num_foods, num_preys, num_predators):
    """
    Create the initial entities for the simulation.
    """
    foods = [Food() for _ in range(num_foods)]
    preys = [Prey(random.uniform(0, MAP_WIDTH), random.uniform(0, MAP_HEIGHT)) for _ in range(num_preys)]
    predators = [Predator(random.uniform(0, MAP_WIDTH), random.uniform(0, MAP_HEIGHT)) for _ in range(num_predators)]
    return foods, preys, predators

def display_info(screen, generation, preys, predators, foods):
    """
    Display the current generation, number of preys, predators, and food entities on the screen.
    """
    font = pygame.font.Font(None, 36)
    text = font.render(f"Generation: {generation+1}", 1, (255, 255, 255))
    screen.blit(text, (MAP_WIDTH - 200, 10))
    text = font.render(f"Preys: {len(preys)}", 1, (255, 255, 255))
    screen.blit(text, (MAP_WIDTH - 200, 50))
    text = font.render(f"Predators: {len(predators)}", 1, (255, 255, 255))
    screen.blit(text, (MAP_WIDTH - 200, 90))
    text = font.render(f"Foods: {len(foods)}", 1, (255, 255, 255))
    screen.blit(text, (MAP_WIDTH - 200, 130))

def main():
    """
    Main function to run the simulation.
    """
    statistics = Statistics()
    plot_shown = False  # Add a flag to check if the plot has been shown
    stable_state_steps = 500  # Number of steps to consider the state as stable
    prev_prey_count = 0
    prev_predator_count = 0
    stable_steps_counter = 0
    generation = 0

    try:   
        while generation is None or generation < GENERATIONS:
            screen, clock = initialize_pygame()
            foods, preys, predators = create_entities(50, 20, 10)
            prev_prey_count = len(preys)
            prev_predator_count = len(predators)

            for step in range(1000):
                screen.fill((0, 0, 0))

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise SystemExit

                for food in foods:
                    food.draw()

                for prey in preys:
                    prey.update(foods, preys, predators)
                    prey.draw()

                for predator in predators:
                    predator.update(preys, predators, foods)
                    predator.draw()

                display_info(screen, generation, preys, predators, foods)  # Display the info

                pygame.display.flip()
                clock.tick()

                statistics.update(generation, step, preys, predators)

                # Check if the number of predators or preys has been stable for a certain number of steps
                if len(preys) == prev_prey_count and len(predators) == prev_predator_count:
                    stable_steps_counter += 1
                else:
                    stable_steps_counter = 0

                # Update the previous counts after the stability check
                prev_prey_count = len(preys)
                prev_predator_count = len(predators)

                # Break if all predators or preys are gone or if the state has been stable for a certain number of steps
                if len(predators) == 0 or len(preys) == 0 or stable_steps_counter >= stable_state_steps:
                    break

            genetic_algorithm(preys + predators)
            generation += 1   
            print(generation)                                                                                                                                                                          


    except SystemExit:
        print("Pygame manually closed")
        statistics.plot()
        plot_shown = True  # Set the flag to True after showing the plot
    finally:
        prey_nn.save('prey_nn.h5')
        predator_nn.save('predator_nn.h5')
        pygame.quit()  # Ensure Pygame quits properly
        

    if not plot_shown:  # Only show the plot if it hasn't been shown yet
        statistics.plot()

if __name__ == "__main__":
    main()
