import numpy as np

from src.Network import Network

# This function is meant to tune the classification neural network model. It takes in datafolds, labels folds, a
# test/tune set it also takes a list hyperparameters to test in the grid search


def hyperparameter_tune_classification_pso(data_folds, label_folds, tune_data, tune_labels, inertias, personal_weights,
                                                      global_weights, max_velocitys, min_velocitys, hidden_layer_sizes, num_inputs,
                                                      num_outputs, output_type, max_iterations, population_sizes):
    from CrossValidateFunctionsPSO import cross_validate_tune_classification

    best_inertia = best_personal_weight = best_global_weight = best_max_velocity = best_min_velocity = \
        best_population_size = 0
    best_performance = 0
    network = Network(len(hidden_layer_sizes), hidden_layer_sizes, num_inputs, num_outputs, output_type,
                      [])

    # perform a grid search over hyperparameters
    for inertia in inertias:
        for personal_weight in personal_weights:
            for global_weight in global_weights:
                for max_velocity in max_velocitys:
                    for min_velocity in min_velocitys:
                        for population_size in population_sizes:

                            # if parameters are better save them
                            if cross_validate_tune_classification(data_folds, label_folds, tune_data, tune_labels, inertia, personal_weight,
                                                                  global_weight, max_velocity, min_velocity, hidden_layer_sizes, num_inputs,
                                                                  num_outputs, output_type, max_iterations, population_size) > best_performance:
                                best_inertia = inertia
                                best_personal_weight = personal_weight
                                best_global_weight = global_weight
                                best_max_velocity = max_velocity
                                best_min_velocity = min_velocity
                                best_population_size = population_size

    return best_inertia, best_personal_weight, best_global_weight, best_max_velocity, best_min_velocity, best_population_size


# This function is meant to tune the classification neural network model. It takes in datafolds, labels folds, a
# test/tune set it also takes a list hyperparameters to test in the grid search

def hyperparameter_tune_classification_pso(data_folds, label_folds, tune_data, tune_labels, inertias, personal_weights,
                                                      global_weights, max_velocitys, min_velocitys, hidden_layer_sizes, num_inputs,
                                                      num_outputs, output_type, max_iterations, population_sizes):
    from CrossValidateFunctionsPSO import cross_validate_tune_regression

    best_inertia = best_personal_weight = best_global_weight = best_max_velocity = best_min_velocity = \
        best_population_size = 0
    best_performance = 0


    # perform a grid search over hyperparameters
    for inertia in inertias:
        for personal_weight in personal_weights:
            for global_weight in global_weights:
                for max_velocity in max_velocitys:
                    for min_velocity in min_velocitys:
                        for population_size in population_sizes:

                            # if parameters are better save them
                            if cross_validate_tune_regression(data_folds, label_folds, tune_data, tune_labels, inertia, personal_weight,
                                                                  global_weight, max_velocity, min_velocity, hidden_layer_sizes, num_inputs,
                                                                  num_outputs, output_type, max_iterations, population_size) > best_performance:
                                best_inertia = inertia
                                best_personal_weight = personal_weight
                                best_global_weight = global_weight
                                best_max_velocity = max_velocity
                                best_min_velocity = min_velocity
                                best_population_size = population_size

    return best_inertia, best_personal_weight, best_global_weight, best_max_velocity, best_min_velocity, best_population_size