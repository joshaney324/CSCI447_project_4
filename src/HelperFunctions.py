import numpy as np
from Network import Network
from HyperparameterTuneBackprop import *
from HyperParameterTuneGA import *
from HyperparameterTuneDE import *
from HyperparameterTunePSO import *
from Fold_functions import *
from ParticleSwarm import ParticleSwarm




def binary_encoding(data, indices):
    uniques = [np.unique(data[:, col]) for col in indices]

    # create mapping from category to binary vectors
    category_to_binary = []
    for i in range(len(indices)):
        category_mapping = {}
        identity_matrix = np.eye(len(uniques[i]))

        for j, value in enumerate(uniques[i]):
            category_mapping[value] = identity_matrix[j]
        category_to_binary.append(category_mapping)

    binary_encoded_data = []

    # apply binary encoding
    for row in data:
        encoded_row = []
        for i, value in enumerate(row):
            if i in indices:
                # find corresponding binary vector and extend row
                col_index = indices.index(i)
                encoded_row.extend(category_to_binary[col_index][value])
            else:
                encoded_row.append(float(value))
        binary_encoded_data.append(encoded_row)
    return np.array(binary_encoded_data)


def test_classification_dataset(dataset):
    from CrossValidateFunctionsGA import cross_validate_classification_ga
    from CrossValidateFunctionsDE import cross_validate_classification_de
    from CrossValidateFunctionsPSO import cross_validate_classification_pso

    all_data = dataset.get_data()
    all_labels = dataset.get_labels()
    all_labels = all_labels.reshape(-1, 1)
    all_labels = binary_encoding(all_labels, [0])
    data_folds, label_folds = get_folds_classification(all_data, all_labels, 10)
    tune_data, tune_labels, data, labels = get_tune_folds(data_folds, label_folds)
    test_data_folds, test_label_folds = get_folds_classification(data, labels, 10)
    crossover_rates = [0.7, 0.8, 0.9]
    mutation_rates = [0.01, 0.05, 0.1]
    population_sizes = [50, 100]
    tournament_sizes = [5, 10, 20, 20, 50]
    min_velocities = [-0.1]
    max_velocities = [-0.1]
    inertias = [.7]
    p_weights = [1.49]
    g_weights = [1.49]


    # GENETIC ALGORITHM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # best_crossover_rate, best_mutation_rate, best_population_size, best_tournament_size = (
    #     hyperparameter_tune_classification_ga(crossover_rates, mutation_rates, population_sizes,
    #                                           tournament_sizes, [], 200, data_folds, label_folds,
    #                                           tune_data, tune_labels))

    print("Zero Layers GA:")
    print(cross_validate_classification_ga(test_data_folds, test_label_folds, .9, 0.01,
                                           [], len(test_data_folds[0][0]), len(test_label_folds[0][0]),
                                           "classification", 50, 20, 10))

    # best_crossover_rate, best_mutation_rate, best_population_size, best_tournament_size = (
    #     hyperparameter_tune_classification_ga(crossover_rates, mutation_rates, population_sizes,
    #                                           tournament_sizes, [8], 200, data_folds, label_folds,
    #                                           tune_data, tune_labels))

    print("One Layer GA:")
    print(cross_validate_classification_ga(test_data_folds, test_label_folds, .9, 0.05,
                                           [8], len(test_data_folds[0][0]), len(test_label_folds[0][0]),
                                           "classification", 50, 20, 10))

    # best_crossover_rate, best_mutation_rate, best_population_size, best_tournament_size = (
    #     hyperparameter_tune_classification_ga(crossover_rates, mutation_rates, population_sizes,
    #                                           tournament_sizes, [2, 3], 200, data_folds, label_folds,
    #                                           tune_data, tune_labels))

    print("Two Layers GA:")
    print(cross_validate_classification_ga(test_data_folds, test_label_folds, .9, 0.01,
                                           [4, 6], len(test_data_folds[0][0]), len(test_label_folds[0][0]),
                                           "classification", 50, 20, 10))

    # DIFFERENTIAL EVOLUTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # best_crossover_rate, best_mutation_rate, best_population_size = (
    #     hyperparameter_tune_classification_de(crossover_rates, mutation_rates, population_sizes, [], 1,
    #                                           data_folds, label_folds, tune_data, tune_labels))

    print("Zero Layers DE:")
    print(cross_validate_classification_de(test_data_folds, test_label_folds, .9, 0.01,
                                           [], len(test_data_folds[0][0]), len(test_label_folds[0][0]),
                                           "classification", 50, 20))

    # best_crossover_rate, best_mutation_rate, best_population_size = (
    #     hyperparameter_tune_classification_de(crossover_rates, mutation_rates, population_sizes,
    #                                         [15], 200, data_folds, label_folds,
    #                                           tune_data, tune_labels))

    print("One Layer DE:")
    print(cross_validate_classification_de(test_data_folds, test_label_folds, .8, 0.01,
                                           [8], len(test_data_folds[0][0]), len(test_label_folds[0][0]),
                                           "classification", 50, 20))

    # best_crossover_rate, best_mutation_rate, best_population_size = (
    #     hyperparameter_tune_classification_de(crossover_rates, mutation_rates, population_sizes,
    #                                           [5, 10], 200, data_folds, label_folds,
    #                                           tune_data, tune_labels))

    print("Two Layers DE:")
    print(cross_validate_classification_de(test_data_folds, test_label_folds, .9, 0.01,
                                           [4, 6], len(test_data_folds[0][0]), len(test_label_folds[0][0]),
                                           "classification", 50, 20))

    # # Particle Swarm Optimization!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #
    # best_inertia, best_personal_weight, best_global_weight, best_max_velocity, best_min_velocity, best_population_size = (
    #     hyperparameter_tune_classification_pso(data_folds, label_folds, tune_data, tune_labels, inertias, p_weights,
    #                                            g_weights, max_velocities, min_velocities, [],
    #                                            len(test_data_folds[0][0]), len(test_label_folds[0][0]),
    #                                            "classification", 1, population_sizes))
    #
    #
    # print("PSO zero Hidden Layers")
    # print(cross_validate_classification_pso(data_folds, label_folds, best_inertia, best_personal_weight, best_global_weight,
    #                                         np.ones(len(test_data_folds[0][0])) * best_max_velocity, np.ones(len(test_data_folds[0][0])) * best_min_velocity, [], len(test_data_folds[0][0]),
    #                                         len(test_label_folds[0][0]), "classification", 1, best_population_size))
    #
    #
    #
    # best_inertia, best_personal_weight, best_global_weight, best_max_velocity, best_min_velocity, best_population_size = (
    #     hyperparameter_tune_classification_pso(data_folds, label_folds, tune_data, tune_labels, inertias, p_weights,
    #                                            g_weights, max_velocities, min_velocities, [10],
    #                                            len(test_data_folds[0][0]), len(test_label_folds[0][0]),
    #                                            "classification", 1, population_sizes))
    #
    # print("PSO One Hidden Layer")
    # print(cross_validate_classification_pso(data_folds, label_folds, best_inertia, best_personal_weight,
    #                                         best_global_weight,
    #                                         np.ones(len(test_data_folds[0][0])) * best_max_velocity, np.ones(len(test_data_folds[0][0])) * best_min_velocity, [10], len(test_data_folds[0][0]),
    #                                         len(test_label_folds[0][0]), "classification", 1, best_population_size))
    #
    # best_inertia, best_personal_weight, best_global_weight, best_max_velocity, best_min_velocity, best_population_size = (
    #     hyperparameter_tune_classification_pso(data_folds, label_folds, tune_data, tune_labels, inertias, p_weights,
    #                                            g_weights, max_velocities, min_velocities, [10, 10],
    #                                            len(test_data_folds[0][0]), len(test_label_folds[0][0]),
    #                                            "classification", 1, population_sizes))
    #
    # print("PSO Two Hidden Layers")
    # print(cross_validate_classification_pso(data_folds, label_folds, best_inertia, best_personal_weight,
    #                                         best_global_weight,
    #                                         np.ones(len(test_data_folds[0][0])) * best_max_velocity, np.ones(len(test_data_folds[0][0])) * best_min_velocity, [10, 10], len(test_data_folds[0][0]),
    #                                         len(test_label_folds[0][0]), "classification", 1, best_population_size))
    #

def test_regression_dataset(dataset):
    from CrossValidateFunctionsGA import cross_validate_regression_ga
    from CrossValidateFunctionsDE import cross_validate_regression_de
    from CrossValidateFunctionsPSO import cross_validate_regression
    all_data = dataset.get_data()
    all_labels = dataset.get_labels()
    all_labels = all_labels.reshape(-1, 1)
    # all_labels = binary_encoding(all_labels, [0])
    data_folds, label_folds = get_folds_classification(all_data, all_labels, 10)
    tune_data, tune_labels, data, labels = get_tune_folds(data_folds, label_folds)
    test_data_folds, test_label_folds = get_folds_classification(data, labels, 10)
    crossover_rates = [0.7, 0.8, 0.9]
    mutation_rates = [0.01, 0.05, 0.1]
    population_sizes = [50, 100]
    tournament_sizes = [5, 10, 20, 20, 50]
    min_velocities = [-0.1]
    max_velocities = [-0.1]
    inertias = [.7]
    p_weights = [1.49]
    g_weights = [1.49]

    # GENETIC ALGORITHM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # best_crossover_rate, best_mutation_rate, best_population_size, best_tournament_size = (
    #     hyperparameter_tune_classification_ga(crossover_rates, mutation_rates, population_sizes,
    #                                           tournament_sizes, [], 200, data_folds, label_folds,
    #                                           tune_data, tune_labels))

    print("Zero Layers GA:")
    print(cross_validate_regression_ga(test_data_folds, test_label_folds, .9, 0.01,
                                           [], len(test_data_folds[0][0]), 1,
                                           "regression", 200, 30, 15))

    # best_crossover_rate, best_mutation_rate, best_population_size, best_tournament_size = (
    #     hyperparameter_tune_classification_ga(crossover_rates, mutation_rates, population_sizes,
    #                                           tournament_sizes, [8], 200, data_folds, label_folds,
    #                                           tune_data, tune_labels))

    print("One Layer GA:")
    print(cross_validate_regression_ga(test_data_folds, test_label_folds, .9, 0.05,
                                           [5], len(test_data_folds[0][0]), 1,
                                           "regression", 100, 20, 10))

    # best_crossover_rate, best_mutation_rate, best_population_size, best_tournament_size = (
    #     hyperparameter_tune_classification_ga(crossover_rates, mutation_rates, population_sizes,
    #                                           tournament_sizes, [2, 3], 200, data_folds, label_folds,
    #                                           tune_data, tune_labels))

    print("Two Layers GA:")
    print(cross_validate_regression_ga(test_data_folds, test_label_folds, .9, 0.01,
                                           [15, 5], len(test_data_folds[0][0]), 1,
                                           "regression", 100, 20, 10))

    # DIFFERENTIAL EVOLUTION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # best_crossover_rate, best_mutation_rate, best_population_size = (
    #     hyperparameter_tune_classification_de(crossover_rates, mutation_rates, population_sizes, [], 1,
    #                                           data_folds, label_folds, tune_data, tune_labels))

    print("Zero Layers DE:")
    print(cross_validate_regression_de(test_data_folds, test_label_folds, .9, 0.01,
                                           [], len(test_data_folds[0][0]), 1,
                                           "regression", 40, 20))

    # best_crossover_rate, best_mutation_rate, best_population_size = (
    #     hyperparameter_tune_classification_de(crossover_rates, mutation_rates, population_sizes,
    #                                         [15], 200, data_folds, label_folds,
    #                                           tune_data, tune_labels))

    print("One Layer DE:")
    print(cross_validate_regression_de(test_data_folds, test_label_folds, .8, 0.01,
                                           [5], len(test_data_folds[0][0]), 1,
                                           "regression", 40, 20))

    # best_crossover_rate, best_mutation_rate, best_population_size = (
    #     hyperparameter_tune_classification_de(crossover_rates, mutation_rates, population_sizes,
    #                                           [5, 10], 200, data_folds, label_folds,
    #                                           tune_data, tune_labels))

    print("Two Layers DE:")
    print(cross_validate_regression_de(test_data_folds, test_label_folds, .9, 0.01,
                                           [15, 5], len(test_data_folds[0][0]), 1,
                                           "regression", 40, 20))

    # # Particle Swarm Optimization!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #
    # best_inertia, best_personal_weight, best_global_weight, best_max_velocity, best_min_velocity, best_population_size = (
    #     hyperparameter_tune_classification_pso(data_folds, label_folds, tune_data, tune_labels, inertias, p_weights,
    #                                            g_weights, max_velocities, min_velocities, [],
    #                                            len(test_data_folds[0][0]), len(test_label_folds[0][0]),
    #                                            "classification", 1, population_sizes))
    #
    #
    # print("PSO zero Hidden Layers")
    # print(cross_validate_classification_pso(data_folds, label_folds, best_inertia, best_personal_weight, best_global_weight,
    #                                         np.ones(len(test_data_folds[0][0])) * best_max_velocity, np.ones(len(test_data_folds[0][0])) * best_min_velocity, [], len(test_data_folds[0][0]),
    #                                         len(test_label_folds[0][0]), "classification", 1, best_population_size))
    #
    #
    #
    # best_inertia, best_personal_weight, best_global_weight, best_max_velocity, best_min_velocity, best_population_size = (
    #     hyperparameter_tune_classification_pso(data_folds, label_folds, tune_data, tune_labels, inertias, p_weights,
    #                                            g_weights, max_velocities, min_velocities, [10],
    #                                            len(test_data_folds[0][0]), len(test_label_folds[0][0]),
    #                                            "classification", 1, population_sizes))
    #
    # print("PSO One Hidden Layer")
    # print(cross_validate_classification_pso(data_folds, label_folds, best_inertia, best_personal_weight,
    #                                         best_global_weight,
    #                                         np.ones(len(test_data_folds[0][0])) * best_max_velocity, np.ones(len(test_data_folds[0][0])) * best_min_velocity, [10], len(test_data_folds[0][0]),
    #                                         len(test_label_folds[0][0]), "classification", 1, best_population_size))
    #
    # best_inertia, best_personal_weight, best_global_weight, best_max_velocity, best_min_velocity, best_population_size = (
    #     hyperparameter_tune_classification_pso(data_folds, label_folds, tune_data, tune_labels, inertias, p_weights,
    #                                            g_weights, max_velocities, min_velocities, [10, 10],
    #                                            len(test_data_folds[0][0]), len(test_label_folds[0][0]),
    #                                            "classification", 1, population_sizes))
    #
    # print("PSO Two Hidden Layers")
    # print(cross_validate_classification_pso(data_folds, label_folds, best_inertia, best_personal_weight,
    #                                         best_global_weight,
    #                                         np.ones(len(test_data_folds[0][0])) * best_max_velocity, np.ones(len(test_data_folds[0][0])) * best_min_velocity, [10, 10], len(test_data_folds[0][0]),
    #                                         len(test_label_folds[0][0]), "classification", 1, best_population_size))
    #


