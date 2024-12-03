from Network import Network
import random as random
import numpy as np


# Cross over for continuous values
def cross_over(individual1, individual2):
    new_individual = []
    for i in range(len(individual1)):
        new_individual.append(individual1[i] + individual2[i] / 2)

    return new_individual


# Mutation for an individual
def mutate(individual, sigma, mutation_rate):
    new_individual = []
    for val in individual:
        if mutation_rate > random.random():
            new_individual.append(val + np.random.normal(0, sigma))
        else:
            new_individual.append(val)

    return new_individual


class GeneticAlgorithm:
    def __init__(self, mutation_rate, crossover_rate, population_size, max_generations, num_inputs, num_outputs, hidden_layer_sizes,
                 network_type, fitness_data, fitness_labels):
        self.crossover_rate = crossover_rate
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

        # Set up network to test with when training
        self.network = Network(len(hidden_layer_sizes), hidden_layer_sizes, num_inputs, num_outputs, network_type, [])

        # Create all individuals. This is done by creating a weight vector that can be used to update a network
        for z in range(population_size):
            individual = []
            for i in range(len(self.network.layers)):
                for j in range(len(self.network.layers[i].node_list)):
                    node_weight_size = len(self.network.layers[i].node_list[j].weights)
                    for x in range(node_weight_size):
                        individual.append(random.random() * 0.01)

            self.population.append(individual)

        self.recalculate_fitness()

    # Selection by a tournament
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

    # Update the fitness dictionary to have all the individual indexes in the population and then their fitness
    def recalculate_fitness(self):

        self.fitness_dict = {}

        for i, individual in enumerate(self.population):
            self.network.update_weights(individual)
            self.fitness_dict[i] = self.network.fitness_function(self.fitness_data, self.fitness_labels)

        new_sorted_individuals = []
        for key in sorted(self.fitness_dict.items(), key=lambda x: x[1]):
            new_sorted_individuals.append(self.population[key[0]])

        self.sorted_individuals = new_sorted_individuals

    def train(self, tournament_size):
        for i in range(self.max_generations):
            # print(i)
            # Select 2 individuals
            selected_individuals = self.selection(tournament_size)
            new_individual = None

            # If cross over, cross over the two individuals and then mutate the new individual. Then add it to the
            # population
            if self.crossover_rate > random.random():
                new_individual = cross_over(selected_individuals[0], selected_individuals[1])

            # If no cross over, mutate the two parents and add them to the population
            if new_individual is not None:
                mutated_individual = mutate(new_individual, .5, self.mutation_rate)
                self.population.append(mutated_individual)
            else:
                self.population.append(mutate(selected_individuals[1], .5, self.mutation_rate))
                self.population.append(mutate(selected_individuals[0], .5, self.mutation_rate))

            self.recalculate_fitness()

            # Remove the worst performing individuals
            if len(self.population) == self.population_size + 1:
                self.population = self.sorted_individuals[:-1]
            else:
                self.population = self.sorted_individuals[:-2]

            self.recalculate_fitness()

            # print(len(self.population))

            # Repeat

        return self.sorted_individuals[0]











