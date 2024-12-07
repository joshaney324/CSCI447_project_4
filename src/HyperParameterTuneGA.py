# This function is meant to tune the classification neural network model. It takes in datafolds, labels folds, a
# test/tune set it also takes a list hyperparameters to test in the grid search


def hyperparameter_tune_classification_ga(crossover_rates, mutation_rates, population_sizes, tournament_sizes,
                                          layer_sizes, max_epochs, data_folds, label_folds, test_data, test_labels):
    from CrossValidateFunctionsGA import cross_validate_tune_classification

    best_crossover_rate = best_mutation_rate = best_population_size = best_tournament_size = 0
    best_performance = 0


    # Perform grid search over hyperparameters
    for crossover_rate in crossover_rates:
        for mutation_rate in mutation_rates:
            for population_size in population_sizes:
                for tournament_size in tournament_sizes:

                    # If the set is better save them
                    if cross_validate_tune_classification(data_folds, label_folds, test_data, test_labels,
                                                          crossover_rate, mutation_rate, layer_sizes, len(test_data[0]),
                                                          len(test_labels[0]), "classification", max_epochs,
                                                          population_size, tournament_size) > best_performance:
                        best_crossover_rate = crossover_rate
                        best_mutation_rate = mutation_rate
                        best_population_size = population_size
                        best_tournament_size = tournament_size

    return best_crossover_rate, best_mutation_rate, best_population_size, best_tournament_size


# This function is meant to tune the classification neural network model. It takes in datafolds, labels folds, a
# test/tune set it also takes a list hyperparameters to test in the grid search

def hyperparameter_tune_regression_ga(crossover_rates, mutation_rates, population_sizes, tournament_sizes,
                                      layer_sizes, max_epochs, data_folds, label_folds, test_data, test_labels):
    from CrossValidateFunctionsGA import cross_validate_tune_regression

    best_crossover_rate = best_mutation_rate = best_population_size = best_tournament_size = 0
    best_performance = 0

    # Perform grid search over hyperparameters
    for crossover_rate in crossover_rates:
        for mutation_rate in mutation_rates:
            for population_size in population_sizes:
                for tournament_size in tournament_sizes:

                    # If set is better save them
                    if cross_validate_tune_regression(data_folds, label_folds, test_data, test_labels,
                                                      crossover_rate, mutation_rate, layer_sizes, len(test_data[0]),
                                                      len(test_labels[0]), "regression", max_epochs,
                                                      population_size, tournament_size) < best_performance:
                        best_crossover_rate = crossover_rate
                        best_mutation_rate = mutation_rate
                        best_population_size = population_size
                        best_tournament_size = tournament_sizes

    return best_crossover_rate, best_mutation_rate, best_population_size, best_tournament_size