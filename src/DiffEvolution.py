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

        for i in range(population_size):
            individual = []
            for j in range(len(self.network.layers)):
                for k in range(len(self.network.layers[j].node_list)):
                    node_weight_size = len(self.network.layers[j].node_list[k].weights)
                    for x in range(node_weight_size):
                        individual.append(random.random() * 0.01)

            self.population.append(individual)
            self.network.update_weights(individual)
            self.fitnesses.append(self.network.fitness_function(self.fitness_data, self.fitness_labels))

    def mutate(self, base_vector, vector_2, vector_3):
        donor_vector = []
        for i in range(len(base_vector)):
            donor_vector.append(base_vector[i] + self.mutation_rate * (vector_2 - vector_3[i]))
        return donor_vector

    def cross_over(self, donor_vector, base_vector):
        offspring_vector = []
        for i in range(len(donor_vector)):
            rand_num = random.random()
            if rand_num <= self.cross_over_rate:
                offspring_vector.append(base_vector[i])
            else:
                offspring_vector.append(donor_vector[i])
        return offspring_vector

    def evolve_individual(self, base_vector, vector_2, vector_3):
        donor_vector = []
        for i in range(len(base_vector)):
            donor_vector.append(base_vector[i] + self.mutation_rate * (vector_2 - vector_3))

        offspring_vector = []
        for i in range(len(donor_vector)):
            rand_num = random.random()
            if rand_num <= self.cross_over_rate:
                offspring_vector.append(base_vector[i])
            else:
                offspring_vector.append(donor_vector[i])

        self.network.update_weights(offspring_vector)
        offspring_fitness = self.network.fitness_function(self.fitness_data, self.fitness_labels)

        return (offspring_vector, offspring_fitness)

    def evolve_population(self):
        for i in range(len(self.population)):
            individual = self.population[i]
            fitness = self.fitnesses[i]

            vector_valid = False
            while not vector_valid:
                vector_2 = random.randint(1, self.population_size) - 1
                if vector_2 != individual:
                    vector_valid = True
            vector_valid = False
            while not vector_valid:
                vector_3 = random.randint(1, self.population_size) - 1
                if vector_3 != individual and vector_3 != vector_2:
                    vector_valid = True

            offspring_vector, offspring_fitness = self.evolve_individual(individual, vector_2, vector_3)

            if offspring_fitness < fitness:
                self.population[i] = offspring_vector
                self.fitnesses[i] = offspring_fitness

    def train(self, max_iterations):
        for i in range(max_iterations):
            self.evolve_population()

        best_index = np.argmin(self.fitnesses)
        return self.population[best_index]