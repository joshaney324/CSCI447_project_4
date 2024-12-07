from Network import Network
import random as random
import numpy as np


class DiffEvolution:
    def __init__(self, mutation_rate, cross_over_rate, population_size, max_generations, num_inputs, num_outputs, hidden_layer_sizes,
                 network_type, fitness_data, fitness_labels):
        self.mutation_rate = mutation_rate
        self.cross_over_rate = cross_over_rate
        self.population_size = population_size
        self.max_generations = max_generations
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layer_sizes = hidden_layer_sizes
        self.network_type = network_type
        # self.sorted_individuals = []
        self.population = []
        self.fitnesses = []
        self.fitness_labels = fitness_labels
        self.fitness_data = fitness_data

        self.network = Network(len(hidden_layer_sizes), hidden_layer_sizes, num_inputs, num_outputs, network_type, [])

        # initialize the population
        for i in range(population_size):
            individual = []
            for j in range(len(self.network.layers)):
                for k in range(len(self.network.layers[j].node_list)):
                    node_weight_size = len(self.network.layers[j].node_list[k].weights)
                    for x in range(node_weight_size):
                        individual.append(random.random() * 0.01)
            # add the fitness of the individual to the fitnesses array
            self.population.append(individual)
            self.network.update_weights(individual)
            self.fitnesses.append(self.network.fitness_function(self.fitness_data, self.fitness_labels))

    # cross over and evolve for a single individual in the population -- create an offspring vector and get its fitness
    def evolve_individual(self, base_vector, vector_2, vector_3):
        # mutate the base vector using vectors two and three to generate the donor vector
        donor_vector = []
        for i in range(len(base_vector)):
            donor_vector.append(base_vector[i] + self.mutation_rate * (vector_2[i] - vector_3[i]))

        # cross over with the donor vector generated through mutation
        offspring_vector = []
        for i in range(len(donor_vector)):
            rand_num = random.random()
            if rand_num <= self.cross_over_rate:
                offspring_vector.append(base_vector[i])
            else:
                offspring_vector.append(donor_vector[i])

        # get the fitness of the offspring vector
        self.network.update_weights(offspring_vector)
        offspring_fitness = self.network.fitness_function(self.fitness_data, self.fitness_labels)

        return (offspring_vector, offspring_fitness)

    # generate the next generation of the population
    def evolve_population(self):
        for i in range(len(self.population)):
            individual = self.population[i]
            fitness = self.fitnesses[i]

            # ensure the trial vectors are distinct
            vector_valid = False
            while not vector_valid:
                vector_2_index = random.randint(1, self.population_size) - 1
                vector_2 = np.array(self.population[vector_2_index])
                if not np.array_equal(individual, vector_2):
                    vector_valid = True
            vector_valid = False
            while not vector_valid:
                vector_3_index = random.randint(1, self.population_size) - 1
                vector_3 = np.array(self.population[vector_3_index])
                if not np.array_equal(individual, vector_3) and not np.array_equal(vector_2, vector_3):
                    vector_valid = True

            # get the offspring of the current individual
            offspring_vector, offspring_fitness = self.evolve_individual(individual, vector_2, vector_3)

            # update the current individual to its offspring if the offspring performs better than the current vector
            if offspring_fitness < fitness:
                self.population[i] = offspring_vector
                self.fitnesses[i] = offspring_fitness

    # evolve for the specified number of generations
    def train(self, max_iterations):
        for i in range(max_iterations):
            self.evolve_population()
        best_index = np.argmin(self.fitnesses)
        return self.population[best_index]