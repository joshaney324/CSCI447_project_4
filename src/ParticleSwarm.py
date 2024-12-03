import random

from Network import Network


class ParticleSwarm:
    def __init__(self, population_size, inertia, personalWeight, globalWeight, max_velocity, min_velocity,  max_iterations, num_inputs, num_outputs, hidden_layer_sizes,
                 network_type, fitness_data, fitness_labels):
        self.population_size = population_size
        self.inertia = inertia
        self.personalWeight = personalWeight
        self.globalWeight = globalWeight
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

        for i in range(population_size):
            position = []
            for j in range(len(self.network.layers)):
                for k in range(len(self.network.layers[j].node_list)):
                    node_weight_size = len(self.network.layers[j].node_list[k].weights)
                    for x in range(node_weight_size):
                        position.append(random.random() * 0.01)

            self.network.update_weights(position)
            fitness = self.network.fitness_function(self.fitness_data, self.fitness_labels)
            particle = Particle(position, fitness, max_velocity, min_velocity)
            self.population.append(particle)

class Particle:
    def __init__(self, position, fitness, max_velocity, min_velocity, personalWeight, globalWeight):
        self.position = position
        self.best_position = position
        self.best_fitness = fitness
        self.velocity = []
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self.personalWeight = personalWeight
        self.globalWeight = globalWeight
        for i in range(len(self.position)):
            velocity_val = random.uniform(self.min_velocity[i], self.max_velocity[i])
            self.velocity.append(velocity_val)

    def update_best(self, best_position, best_fitness):
        self.best_position = best_position
        self.best_fitness = best_fitness

    def update_position(self):
        self.position = self.position + self.velocity