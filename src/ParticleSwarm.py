import random
import sys

import numpy as np
from Network import Network


class ParticleSwarm:
    def __init__(self, population_size, inertia, personal_weight, global_weight, max_velocity, min_velocity,  max_iterations, num_inputs, num_outputs, hidden_layer_sizes,
                 network_type, fitness_data, fitness_labels):
        self.population_size = population_size
        self.inertia = inertia
        self.personal_weight = personal_weight
        self.global_weight = global_weight
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self.max_iterations = max_iterations
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layer_sizes = hidden_layer_sizes
        self.network_type = network_type
        # self.sorted_individuals = []
        self.population = []
        self.fitness_labels = fitness_labels
        self.fitness_data = fitness_data

        self.network = Network(len(hidden_layer_sizes), hidden_layer_sizes, num_inputs, num_outputs, network_type, [])

        self.global_best_position = []
        self.global_best_fitness = np.inf
        for i in range(population_size):
            position = []
            for j in range(len(self.network.layers)):
                for k in range(len(self.network.layers[j].node_list)):
                    node_weight_size = len(self.network.layers[j].node_list[k].weights)
                    for x in range(node_weight_size):
                        position.append(random.random() * 0.01)

            self.network.update_weights(position)
            fitness = self.network.fitness_function(self.fitness_data, self.fitness_labels)
            particle = Particle(position, fitness, max_velocity, min_velocity, inertia, personal_weight, global_weight)
            self.population.append(particle)
            if fitness < self.global_best_fitness:
                self.global_best_position = position
                self.global_best_fitness = fitness


    def update(self):
        for i in range(len(self.population)):
            self.population[i].update_velocity(self.global_best_position)
            self.population[i].update_position()
            self.network.update_weights(self.population[i].position)
            fitness = self.network.fitness_function(self.fitness_data, self.fitness_labels)
            if fitness < self.population[i].best_fitness:
                self.population[i].update_best(self.population[i].position, fitness)
            if fitness < self.global_best_fitness:
                self.global_best_position = self.population[i].position
                self.global_best_fitness = fitness
            ## Mostly done -- think it needs some more steps


class Particle:
    def __init__(self, position, fitness, max_velocity, min_velocity, inertia, personal_weight, global_weight):
        self.position = position
        self.best_position = position
        self.best_fitness = fitness
        self.velocity = []
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self.inertia = inertia
        self.personal_weight = personal_weight
        self.global_weight = global_weight
        for i in range(len(self.position)):
            velocity_val = random.uniform(self.min_velocity[i], self.max_velocity[i])
            self.velocity.append(velocity_val)

    def update_best(self, best_position, best_fitness):
        self.best_position = best_position
        self.best_fitness = best_fitness

    def update_position(self):
        self.position = self.position + self.velocity

    def update_velocity(self, global_best):
        personal_rand = random.random()
        global_rand = random.random()
        position = np.array(self.position)
        velocity = np.array(self.velocity)
        personal_best = np.array(self.best_position)
        global_best = np.array(global_best)
        self.velocity = (self.inertia * velocity) + (personal_rand * self.personal_weight * (personal_best - position)) + (global_rand * self.global_weight * (global_best - position))