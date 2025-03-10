import random
import sys

import numpy as np
from Network import Network


class ParticleSwarm:
    def __init__(self, population_size, inertia, personal_weight, global_weight, max_velocity, min_velocity, num_inputs, num_outputs, hidden_layer_sizes,
                 network_type, fitness_data, fitness_labels):
        self.population_size = population_size
        self.inertia = inertia
        self.personal_weight = personal_weight
        self.global_weight = global_weight
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layer_sizes = hidden_layer_sizes
        self.network_type = network_type
        # self.sorted_individuals = []
        self.population = []
        self.fitness_labels = fitness_labels
        self.fitness_data = fitness_data

        self.network = Network(len(hidden_layer_sizes), hidden_layer_sizes, num_inputs, num_outputs, network_type, [])

        self.max_velocity = np.ones(self.network.get_weight_vec_size()) * max_velocity
        self.min_velocity = np.ones(self.network.get_weight_vec_size()) * min_velocity

        self.global_best_position = []
        self.global_best_fitness = np.inf
        # initialize a weight vector for each member of the population
        for i in range(population_size):
            position = []
            for j in range(len(self.network.layers)):
                for k in range(len(self.network.layers[j].node_list)):
                    node_weight_size = len(self.network.layers[j].node_list[k].weights)
                    for x in range(node_weight_size):
                        position.append(random.random() * 0.01)
            # check the fitness of the given weight vector
            self.network.update_weights(position)
            fitness = self.network.fitness_function(self.fitness_data, self.fitness_labels)
            # create a particle with the given weight vector and add it to the population
            particle = Particle(position, fitness, self.max_velocity, self.min_velocity, inertia, personal_weight, global_weight)
            self.population.append(particle)
            # if the fitness of this weight vector is better than the global best, set it as the global best position
            if fitness < self.global_best_fitness:
                self.global_best_position = position
                self.global_best_fitness = fitness

    # do one iteration of training
    def update(self):
        # for every particle in the population...
        for i in range(len(self.population)):
            # update the velocity and position of the given particle
            self.population[i].update_velocity(self.global_best_position)
            self.population[i].update_position()
            # check the fitness of the given particle
            self.network.update_weights(self.population[i].position)
            fitness = self.network.fitness_function(self.fitness_data, self.fitness_labels)
            # if this fitness is better than the particle's personal best, set the personal best
            if fitness < self.population[i].best_fitness:
                self.population[i].update_best(self.population[i].position, fitness)
            # if this fitness is better than the global best, set the global best
            if fitness < self.global_best_fitness:
                self.global_best_position = self.population[i].position
                self.global_best_fitness = fitness

    # train for the specified number of iterations
    def train(self, max_iterations):
        for i in range(max_iterations):
            self.update()
        return self.global_best_position

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
        self.velocity = np.random.uniform(self.min_velocity, self.max_velocity) * 0.5

    # update the best position and fitness value for the particle
    def update_best(self, best_position, best_fitness):
        self.best_position = best_position
        self.best_fitness = best_fitness

    # update the position of the particle
    def update_position(self):
        self.position = self.position + self.velocity

    # update the velocity of the particle
    def update_velocity(self, global_best):
        # initialize random values
        personal_rand = random.random()
        global_rand = random.random()
        position = np.array(self.position)
        velocity = np.array(self.velocity)
        personal_best = np.array(self.best_position)
        global_best = np.array(global_best)
        # update the velocity of the particle
        self.velocity = (self.inertia * velocity) + (personal_rand * self.personal_weight * (personal_best - position)) + (global_rand * self.global_weight * (global_best - position))
        # limit the velocity values to the range specified by min and max velocity
        for i in range(len(self.velocity)):
            if self.velocity[i] < self.min_velocity[i]:
                self.velocity[i] = self.min_velocity[i]
            elif self.velocity[i] > self.max_velocity[i]:
                self.velocity[i] = self.max_velocity[i]