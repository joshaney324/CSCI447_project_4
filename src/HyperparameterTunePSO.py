import numpy as np

from src.Network import Network


def hyperparameter_tune_classification_pso(data_folds, label_folds, tune_data, tune_labels, inertias, personal_weights,
                                                      global_weights, max_velocitys, min_velocitys, hidden_layer_sizes, num_inputs,
                                                      num_outputs, output_type, max_iterations, population_sizes):
    from CrossValidateFunctionsPSO import cross_validate_tune_classification

    best_inertia = best_personal_weight = best_global_weight = best_max_velocity = best_min_velocity = \
        best_population_size = 0
    best_performance = 0
    network = Network(len(hidden_layer_sizes), hidden_layer_sizes, num_inputs, num_outputs, output_type,
                      [])
    weight_vec_size = network.get_weight_vec_size()


    for inertia in inertias:
        for personal_weight in personal_weights:
            for global_weight in global_weights:
                for max_velocity in max_velocitys:
                    for min_velocity in min_velocitys:
                        for population_size in population_sizes:

                            max_velocity_vec = np.ones(weight_vec_size) * max_velocity
                            min_velocity_vec = np.ones(weight_vec_size) * min_velocity

                            if cross_validate_tune_classification(data_folds, label_folds, tune_data, tune_labels, inertia, personal_weight,
                                                                  global_weight, max_velocity_vec, min_velocity_vec, hidden_layer_sizes, num_inputs,
                                                                  num_outputs, output_type, max_iterations, population_size) > best_performance:
                                best_inertia = inertia
                                best_personal_weight = personal_weight
                                best_global_weight = global_weight
                                best_max_velocity = max_velocity
                                best_min_velocity = min_velocity
                                best_population_size = population_size

    return best_inertia, best_personal_weight, best_global_weight, best_max_velocity, best_min_velocity, best_population_size


def hyperparameter_tune_regression_de(crossover_rates, mutation_rates, population_sizes,
                                      layer_sizes, max_epochs, data_folds, label_folds, test_data, test_labels):
    from CrossValidateFunctionsDE import cross_validate_tune_regression

    best_crossover_rate = best_mutation_rate = best_population_size = 0
    best_performance = 0

    for crossover_rate in crossover_rates:
        for mutation_rate in mutation_rates:
            for population_size in population_sizes:

                if cross_validate_tune_regression(data_folds, label_folds, test_data, test_labels,
                                                      crossover_rate, mutation_rate, layer_sizes, len(test_data[0]),
                                                      len(test_labels[0]), "regression", max_epochs,
                                                      population_size) > best_performance:
                    best_crossover_rate = crossover_rate
                    best_mutation_rate = mutation_rate
                    best_population_size = population_size

    return best_crossover_rate, best_mutation_rate, best_population_size
