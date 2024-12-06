import datetime

import numpy as np
from Metric_functions import precision, recall, accuracy, mean_squared_error
from src.GeneticAlgorithm import GeneticAlgorithm


def cross_validate_classification_ga(data_folds, label_folds, crossover_rate, mutation_rate,
                                     hidden_layer_sizes, num_inputs, num_outputs, output_type, max_iterations,
                                     population_size, tournament_size):
    from Network import Network
    # the cross_validate function is meant to get the precision, recall and accuracy values from each fold then print
    # out the average across folds. this function takes in a list of data folds, a list of label folds, a tuning hold
    # out set for training, and the hyperparameters for the model. This function returns a list of all of our metrics

    # Set up variables
    precision_avg = 0.0
    recall_avg = 0.0
    accuracy_avg = 0.0
    folds = len(data_folds)


    # For each testing fold, set up a training and testing set and then append the loss function values
    for i in range(len(data_folds)):
        train_data = []
        test_data = []
        train_labels = []
        test_labels = []
        for j in range(len(data_folds)):
            if i != j:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    train_data.append(instance)
                    train_labels.append(label)
            else:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    test_data.append(instance)
                    test_labels.append(label)

        # make all the data into np arrays
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        ga = GeneticAlgorithm(mutation_rate, crossover_rate, population_size, max_iterations, num_inputs, num_outputs,
                              hidden_layer_sizes, output_type, train_data, train_labels)
        weight_vector = ga.train(tournament_size)

        network = Network(len(hidden_layer_sizes), hidden_layer_sizes, num_inputs, num_outputs, output_type,
                          [])
        network.update_weights(weight_vector)
        # Get all predictions
        predictions = []
        for datapoint in test_data:
            predictions.append(network.predict(datapoint))

        # Get averages of precision recall and accuracy
        pre_vals = precision(predictions, test_labels)
        precision_vals = []
        if len(pre_vals) != 0 and len(pre_vals[0]) != 1:
            for val in pre_vals:
                precision_vals.append(val[1])
            precision_avg += np.mean(precision_vals)
        rec_vals = recall(predictions, test_labels)
        recall_vals = []
        if len(rec_vals) != 0 and len(rec_vals[0]) != 1:
            for val in rec_vals:
                recall_vals.append(val[1])
            recall_avg += np.mean(recall_vals)
        acc, matrix = accuracy(predictions, test_labels)
        accuracy_vals = []
        for val in acc:
            accuracy_vals.append(val[1])

        accuracy_avg += np.mean(accuracy_vals)
        # print(str(datetime.datetime.now()))
        # Print final accuracy and return
    # print(str(datetime.datetime.now()) + " Final Accuracy value: " + str(accuracy_avg / folds))
    return [precision_avg / folds, recall_avg / folds, accuracy_avg / folds]


def cross_validate_regression_ga(data_folds, label_folds, crossover_rate, mutation_rate, hidden_layer_sizes, num_inputs,
                              num_outputs, output_type, max_iterations, population_size, tournament_size):
    from Network import Network
    # the cross_validate function is meant to get the precision, recall and accuracy values from each fold then print
    # out the average across folds. this function takes in a list of data folds, a list of label folds, a tuning hold
    # out set for training, and the hyperparameters for the model. This function returns a list of all of our metrics

    # Set up variables
    mse_avg = 0.0
    folds = len(data_folds)
    network = Network(len(hidden_layer_sizes), hidden_layer_sizes, num_inputs, num_outputs, output_type,
                      [])

    # For each testing fold, set up a training and testing set and then append the loss function values
    for i in range(len(data_folds)):
        train_data = []
        test_data = []
        train_labels = []
        test_labels = []
        for j in range(len(data_folds)):
            if i != j:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    train_data.append(instance)
                    train_labels.append(label)
            else:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    test_data.append(instance)
                    test_labels.append(label)

        # Convert to np arrays
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        # Train network
        ga = GeneticAlgorithm(mutation_rate, crossover_rate, population_size, max_iterations, num_inputs, num_outputs,
                              hidden_layer_sizes, output_type, train_data, train_labels)
        weight_vector = ga.train(tournament_size)

        network = Network(len(hidden_layer_sizes), hidden_layer_sizes, num_inputs, num_outputs, output_type,
                          [])
        network.update_weights(weight_vector)

        # Get all predictions
        predictions = []
        for datapoint in test_data:
            predictions.append(network.predict(datapoint))
        # print(mean_squared_error(test_labels, predictions, len(predictions)))
        mse_avg += mean_squared_error(test_labels, predictions, len(predictions))
    # print mse average and return it
    # print(str(datetime.datetime.now()) + " Final mse value: " + str(mse_avg / folds))
    return mse_avg / folds


def cross_validate_tune_regression(data_folds, label_folds, tune_data, tune_labels, crossover_rate, mutation_rate,
                                   hidden_layer_sizes, num_inputs, num_outputs, output_type, max_iterations, population_size, tournament_size):
    from Network import Network
    # This function is meant to cross validate the regression sets but leave out the testing folds and use the tune
    # folds instead. This function takes in data folds, label folds, a hold out fold, and model hyperparameters

    # Set up variables
    mean_squared_error_avg = 0.0
    folds = len(data_folds)

    # For each testing fold, set up a training and testing set and then append the loss function values
    for i in range(len(data_folds)):
        train_data = []
        train_labels = []
        for j in range(len(data_folds)):
            if i != j:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    train_data.append(instance)
                    train_labels.append(label)

        # make all the data into np arrays and set up the test data and labels as the hold out fold
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(tune_data)
        test_labels = np.array(tune_labels)

        # Train network
        ga = GeneticAlgorithm(mutation_rate, crossover_rate, population_size, max_iterations, num_inputs, num_outputs,
                              hidden_layer_sizes, output_type, train_data, train_labels)
        weight_vector = ga.train(tournament_size)

        network = Network(len(hidden_layer_sizes), hidden_layer_sizes, num_inputs, num_outputs, output_type,
                          [])
        network.update_weights(weight_vector)

        # get predictions and append them
        predictions = []
        for datapoint in test_data:
            predictions.append(network.predict(datapoint))

        mean_squared_error_avg += mean_squared_error(test_labels, predictions, len(predictions))

    # return avg mean squared error
    return mean_squared_error_avg / folds


def cross_validate_tune_classification(data_folds, label_folds, test_data, test_labels, crossover_rate, mutation_rate,
                                     hidden_layer_sizes, num_inputs, num_outputs, output_type, max_iterations,
                                     population_size, tournament_size):
    from Network import Network
    # This function is meant to cross validate the classification sets but leave out the testing folds and use the tune
    # folds instead. This function takes in data folds, label folds, a hold out fold, and model hyperparameters

    # Set up variables
    precision_avg = 0.0
    recall_avg = 0.0
    accuracy_avg = 0.0
    folds = len(data_folds)
    matrix_total = np.zeros((2,2))
    accuracies = []


    # For each testing fold, set up a training and testing set and then append the loss function values
    for i in range(len(data_folds)):
        hidden_layer_size = list(hidden_layer_sizes)
        network = Network(len(hidden_layer_sizes), hidden_layer_sizes, num_inputs, num_outputs, output_type,
                          [])
        train_data = []
        train_labels = []
        for j in range(len(data_folds)):
            if i != j:
                for instance, label in zip(data_folds[j], label_folds[j]):
                    train_data.append(instance)
                    train_labels.append(label)

        # make all the data into np arrays and set the test data and labels as the hold out fold
        train_data = np.array(train_data)
        train_labels = np.array(train_labels)
        test_data = np.array(test_data)
        test_labels = np.array(test_labels)

        ga = GeneticAlgorithm(mutation_rate, crossover_rate, population_size, max_iterations, num_inputs, num_outputs,
                              hidden_layer_sizes, output_type, train_data, train_labels)
        weight_vector = ga.train(tournament_size)

        network = Network(len(hidden_layer_sizes), hidden_layer_sizes, num_inputs, num_outputs, output_type,
                          [])
        network.update_weights(weight_vector)

        predictions = []

        # Get all predictions
        for datapoint in test_data:
            predictions.append(network.predict(datapoint))

        # Get all precision recall and accuracy vals
        pre_vals = precision(predictions, test_labels)
        precision_vals = []
        for val in pre_vals:
            precision_vals.append(val[1])
        rec_vals = recall(predictions, test_labels)
        recall_vals = []
        for val in rec_vals:
            recall_vals.append(val[1])
        acc, matrix = accuracy(predictions, test_labels)
        accuracy_vals = []
        for val in acc:
            accuracy_vals.append(val[1])

        precision_avg += np.mean(precision_vals)
        recall_avg += np.mean(recall_vals)
        accuracy_avg += np.mean(accuracy_vals)
    # print(str(datetime.datetime.now()) + " Accuracy value: " + str(accuracy_avg / folds))
    # Return the metrics
    return (precision_avg / folds + recall_avg / folds + accuracy_avg / folds) / 3
