from Network import Network
import random as random
import numpy as np


def cross_over(self, individual1, individual2):
    new_individual = []
    for i in range(len(individual1)):
        new_individual.append(individual1[i] + individual2[i] / 2)

    return new_individual


def mutate(individual, sigma):
    new_individual = []
    for val in individual:
        new_individual.append(val + np.random.normal(0, sigma))

    return new_individual


class GeneticAlgorithm:
    def __init__(self, mutation_rate, population_size, max_generations, num_inputs, num_outputs, hidden_layer_sizes,
                 network_type, fitness_data, fitness_labels):
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.max_generations = max_generations
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.hidden_layer_sizes = hidden_layer_sizes
        self.network_type = network_type
        self.sorted_individuals = []
        self.population = []
        self.fitness_dict = {}
        self.fitness_labels = fitness_labels
        self.fitness_data = fitness_data

        self.network = Network(len(hidden_layer_sizes), hidden_layer_sizes, num_inputs, num_outputs, network_type, [])

        for i in range(population_size):
            individual = []
            for i in range(len(self.network.layers)):
                for j in range(len(self.network.layers[i].node_list)):
                    node_weight_size = len(self.network.layers[i].node_list[j].weights)
                    for x in range(node_weight_size):
                        individual.append(random.random() * 0.01)

            self.population.append(individual)

        self.recalculate_fitness()

    def selection(self, tournament_size):
        individuals = []

        for j in range(2):
            orig_tournament = []
            for i in range(tournament_size):
                orig_tournament.append(random.choice(list(self.fitness_dict.keys())))

            best_fitness = np.inf
            individual = []
            for idx in orig_tournament:
                if self.fitness_dict[idx] < best_fitness:
                    best_fitness = self.fitness_dict[idx]
                    individual = self.population[idx]

            individuals.append(individual)

        return individuals

    def recalculate_fitness(self):

        self.fitness_dict = {}

        for i, individual in enumerate(self.population):
            self.network.update_weights(individual)
            self.fitness_dict[i] = self.network.fitness_function(self.fitness_data, self.fitness_labels)

        new_sorted_individuals = []
        for key in sorted(self.fitness_dict.items(), key=lambda x: x[1]):
            new_sorted_individuals.append(self.population[key[0]])

        self.sorted_individuals = new_sorted_individuals

    def mutate_population(self):
        for i in range(len(self.population)):
            if random.random() < self.mutation_rate:
                self.population[i] = mutate(self.population[i], 0.5)

    def train(self, tournament_size, max_iterations):
        for i in range(max_iterations):
            print(i)
            self.mutate_population()
            selected_individuals = self.selection(tournament_size)
            for individual in selected_individuals:
                self.population.append(individual)

            self.recalculate_fitness()

            # new_population_keys = [key for key, value in sorted(self.fitness_dict.items(), key=lambda x: x[1])[3:]]
            #
            # new_population = []
            # for key in new_population_keys:
            #     new_population.append(self.population[key])
            #
            # self.population = new_population

            self.population = self.sorted_individuals[:-2]
            self.recalculate_fitness()
            print(max(self.fitness_dict.values()))

        return self.sorted_individuals[0]











