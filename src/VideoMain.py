from Hardware import MachineSet
from SoyBeanSet import SoyBeanSet
from Fold_functions import *
from GeneticAlgorithm import GeneticAlgorithm
from ParticleSwarm import ParticleSwarm
from DiffEvolution import DiffEvolution
from Network import Network
from Metric_functions import accuracy, mean_squared_error
from CrossValidateFunctionsPSO import *
from CrossValidateFunctionsGA import *
from CrossValidateFunctionsDE import *
from CrossValidateFunctionsBackProp import *
from HelperFunctions import binary_encoding

soy = SoyBeanSet()
machine = MachineSet()

regression_data = machine.get_data()
regression_labels = machine.get_labels()
regression_labels = regression_labels.reshape(-1, 1)

classification_data = soy.get_data()
classification_labels = soy.get_labels()
classification_labels = classification_labels.reshape(-1, 1)
classification_labels = binary_encoding(classification_labels, [0])

regression_data_folds, regression_label_folds = get_folds_regression(regression_data, regression_labels, 10)
classification_data_folds, classification_label_folds = get_folds_classification(classification_data, classification_labels, 10)

regression_test_data, regression_test_labels, regression_train_data, regression_train_labels = get_tune_folds(regression_data_folds, regression_label_folds)

classification_test_data, classification_test_labels, classification_train_data, classification_train_labels = get_tune_folds(classification_data_folds, classification_label_folds)

# -----------------------------------------------REGRESSION GENETIC ALGORITHM----------------------------------------

regression_ga = GeneticAlgorithm(.01, .9, 50, 50, len(regression_train_data[0]), 1,
                                 [15, 5], "regression", regression_train_data, regression_train_labels)

weight_vector = regression_ga.train(25)

network = Network(2, [15, 5], len(regression_train_data[0]), 1, "regression", [])
network.update_weights(weight_vector)

predictions = []
for data in regression_test_data:
    predictions.append(network.predict(data))

print("GA MSE: ")
print(mean_squared_error(predictions, regression_test_labels, len(predictions)))

# -----------------------------------------------CLASSIFICATION GENETIC ALGORITHM----------------------------------------

classification_ga = GeneticAlgorithm(.01, .9, 50, 50, len(classification_train_data[0]), len(classification_train_labels[0]),
                                 [15, 5], "classification", classification_train_data, classification_train_labels)

weight_vector = regression_ga.train(25)

network = Network(2, [15, 5], len(classification_train_data[0]), len(classification_train_labels[0]), "classification", [])
network.update_weights(weight_vector)

predictions = []
for data in regression_test_data:
    predictions.append(network.predict(data))

print("GA Accuracy: ")
acc_list, _ = accuracy(predictions, classification_test_labels)

print(acc_list)


# -----------------------------------------------REGRESSION DIFFERENTIAL EVOLUTION----------------------------------------

regression_de = DiffEvolution(.01, .9, 50, 50, len(regression_train_data[0]), 1,
                                 [15, 5], "regression", regression_train_data, regression_train_labels)

weight_vector = regression_de.train(50)

network = Network(2, [15, 5], len(regression_train_data[0]), 1, "regression", [])
network.update_weights(weight_vector)

predictions = []
for data in regression_test_data:
    predictions.append(network.predict(data))

print("DE MSE: ")
print(mean_squared_error(predictions, regression_test_labels, len(predictions)))

# -----------------------------------------------CLASSIFICATION DIFFERENTIAL EVOLUTION----------------------------------------

classification_ga = GeneticAlgorithm(.01, .9, 50, 50, len(classification_train_data[0]), len(classification_train_labels[0]),
                                 [15, 5], "classification", classification_train_data, classification_train_labels)

weight_vector = regression_ga.train(50)

network = Network(2, [15, 5], len(classification_train_data[0]), len(classification_train_labels[0]), "classification", [])
network.update_weights(weight_vector)

predictions = []
for data in regression_test_data:
    predictions.append(network.predict(data))

print("DE Accuracy: ")
acc_list, _ = accuracy(predictions, classification_test_labels)

print(acc_list)


# -----------------------------------------------REGRESSION PARTICLE SWARM----------------------------------------


regression_pso = ParticleSwarm(50, .7, 1.49, 1.49, 0.1, -0.1,
                               len(classification_train_data[0]), 1, [15, 10], "regression", regression_train_data,
                               regression_train_labels)

weight_vector = regression_pso.train(50)

network = Network(2, [15, 5], len(regression_train_data[0]), 1, "regression", [])
network.update_weights(weight_vector)

predictions = []
for data in regression_test_data:
    predictions.append(network.predict(data))

print("PSO MSE: ")
print(mean_squared_error(predictions, regression_test_labels, len(predictions)))

# -----------------------------------------------CLASSIFICATION PARTICLE SWARM----------------------------------------


classification_pso = ParticleSwarm(50, .7, 1.49, 1.49, 0.1, -0.1,
                               len(classification_train_data[0]), len(classification_train_labels[0]), [15, 10],
                               "classification", classification_train_data, classification_train_labels)

weight_vector = classification_pso.train(50)

network = Network(2, [15, 5], len(regression_train_data[0]), len(classification_train_labels[0]), "classification", [])
network.update_weights(weight_vector)

predictions = []
for data in regression_test_data:
    predictions.append(network.predict(data))

print("PSO Accuracy: ")

acc_list, _ = accuracy(predictions, classification_test_labels)

print(acc_list)


# -----------------------------------------------CLASSIFICATION BACKPROP ALGORITHM----------------------------------------

network = Network(2, [15, 5], len(classification_train_data[0]), len(classification_train_labels[0]), "classification", [])
network.train(classification_train_data, classification_train_labels, classification_test_data,
              classification_test_labels, 100, .1)

predictions = []
for data in regression_test_data:
    predictions.append(network.predict(data))

print("Backprop Accuracy: ")
acc_list, _ = accuracy(predictions, classification_test_labels)

print(acc_list)


# -----------------------------------------------REGRESSION BACKPROP ALGORITHM----------------------------------------

network = Network(2, [15, 5], len(regression_train_data[0]), 1, "regression", [])
network.train(regression_train_data, regression_train_labels, regression_test_data,
              regression_test_labels, 100, .1)

predictions = []
for data in regression_test_data:
    predictions.append(network.predict(data))

print("Backprop MSE: ")

print(mean_squared_error(predictions, regression_test_labels, len(predictions)))




# --------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------BACKPROPAGATION-----------------------------------------------------

print("Cross validate Back Prop no hidden layers")
print("Precision, Recall, Accuracy")
print(cross_validate_classification(classification_data_folds, classification_label_folds, classification_test_data,
                                    classification_test_labels, 0.1, 0, [],
                                    len(classification_train_data[0]), len(classification_train_labels[0]),
                                    "classification", [], 100))

print("MSE")
print(cross_validate_regression(regression_data_folds, regression_label_folds, regression_test_data,
                                regression_test_labels, 0.1, 0, [],
                                len(regression_train_data[0]), 1,
                                "regression", [], 100))

print("Cross validate Back Prop one hidden layer")
print("Precision, Recall, Accuracy")
print(cross_validate_classification(classification_data_folds, classification_label_folds, classification_test_data,
                                    classification_test_labels, 0.1, 0, [10],
                                    len(classification_train_data[0]), len(classification_train_labels[0]),
                                    "classification", [], 100))

print("MSE")
print(cross_validate_regression(regression_data_folds, regression_label_folds, regression_test_data,
                                regression_test_labels, 0.1, 0, [10],
                                len(regression_train_data[0]), 1,
                                "regression", [], 100))

print("Cross validate Back Prop two hidden layers")
print("Precision, Recall, Accuracy")
print(cross_validate_classification(classification_data_folds, classification_label_folds, classification_test_data,
                                    classification_test_labels, 0.1, 0, [10, 10],
                                    len(classification_train_data[0]), len(classification_train_labels[0]),
                                    "classification", [], 100))

print("MSE")
print(cross_validate_regression(regression_data_folds, regression_label_folds, regression_test_data,
                                regression_test_labels, 0.1, 0, [10, 10],
                                len(regression_train_data[0]), 1,
                                "regression", [], 100))


# ------------------------------------------------GA-----------------------------------------------------

print("Cross validate GA zero hidden layers")
print("Precision, Recall, Accuracy")

print(cross_validate_classification_ga(classification_data_folds, classification_label_folds, .9, 0.01,
                                           [], len(classification_train_labels[0]), len(classification_train_labels[0]),
                                           "classification", 75, 20, 10))

print("MSE")

print(cross_validate_regression_ga(regression_data_folds, regression_label_folds, .9, 0.01,
                                           [], len(classification_train_labels[0]), 1,
                                           "regression", 75, 20, 10))


print("Cross validate GA one hidden layer")
print("Precision, Recall, Accuracy")

print(cross_validate_classification_ga(classification_data_folds, classification_label_folds, .9, 0.01,
                                           [10], len(classification_train_data[0]), len(classification_train_labels[0]),
                                           "classification", 75, 20, 10))

print("MSE")

print(cross_validate_regression_ga(regression_data_folds, regression_label_folds, .9, 0.01,
                                           [10], len(regression_data[0]), 1,
                                           "regression", 75, 20, 10))


print("Cross validate GA two hidden layers")
print("Precision, Recall, Accuracy")

print(cross_validate_classification_ga(classification_data_folds, classification_label_folds, .9, 0.01,
                                           [10, 10], len(classification_train_data[0]), len(classification_train_labels[0]),
                                           "classification", 75, 20, 10))

print("MSE")

print(cross_validate_regression_ga(regression_data_folds, regression_label_folds, .9, 0.01,
                                           [10, 10], len(regression_train_data[0]), 1,
                                           "regression", 75, 20, 10))


# ------------------------------------------------DE-----------------------------------------------------

print("Cross validate DE zero hidden layers")
print("Precision, Recall, Accuracy")

print(cross_validate_classification_de(classification_data_folds, classification_label_folds, .9, 0.01,
                                           [], len(classification_train_labels[0]), len(classification_train_labels[0]),
                                           "classification", 75, 20))

print("MSE")

print(cross_validate_regression_de(regression_data_folds, regression_label_folds, .9, 0.01,
                                           [], len(regression_train_data[0]), 1,
                                           "regression", 75, 20))

print("Cross validate DE one hidden layer")
print("Precision, Recall, Accuracy")

print(cross_validate_classification_de(classification_data_folds, classification_label_folds, .9, 0.01,
                                           [10], len(classification_train_labels[0]), len(classification_train_labels[0]),
                                           "classification", 75, 20))

print("MSE")

print(cross_validate_regression_de(regression_data_folds, regression_label_folds, .9, 0.01,
                                           [10], len(regression_train_data[0]), 1,
                                           "regression", 75, 20))

print("Cross validate DE two hidden layers")
print("Precision, Recall, Accuracy")

print(cross_validate_classification_de(classification_data_folds, classification_label_folds, .9, 0.01,
                                           [10, 10], len(classification_train_labels[0]), len(classification_train_labels[0]),
                                           "classification", 75, 20))

print("MSE")

print(cross_validate_regression_de(regression_data_folds, regression_label_folds, .9, 0.01,
                                           [10, 10], len(regression_train_data[0]), 1,
                                           "regression", 75, 20))

# ------------------------------------------------PSO-----------------------------------------------------

print("Cross validate PSO zero hidden layers")
print("Precision, Recall, Accuracy")


print(cross_validate_classification_pso(classification_data_folds, classification_label_folds, .7, 1.49, 1.49,
                                            .1, -.1, [], len(classification_train_labels[0]),
                                        len(classification_train_labels[0]), "classification", 75, 50))

print("MSE")

print(cross_validate_classification_pso(regression_data_folds, regression_label_folds, .7, 1.49, 1.49,
                                            .1, -.1, [], len(regression_train_labels[0]),
                                        1, "regression", 75, 50))

print("Cross validate PSO one hidden layer")
print("Precision, Recall, Accuracy")


print(cross_validate_classification_pso(classification_data_folds, classification_label_folds, .7, 1.49, 1.49,
                                            .1, -.1, [10], len(classification_train_labels[0]),
                                        len(classification_train_labels[0]), "classification", 75, 50))

print("MSE")

print(cross_validate_classification_pso(regression_data_folds, regression_label_folds, .7, 1.49, 1.49,
                                            .1, -.1, [10], len(regression_train_labels[0]),
                                        1, "regression", 75, 50))


print("Cross validate PSO two hidden layers")
print("Precision, Recall, Accuracy")


print(cross_validate_classification_pso(classification_data_folds, classification_label_folds, .7, 1.49, 1.49,
                                            .1, -.1, [10, 10], len(classification_train_labels[0]),
                                        len(classification_train_labels[0]), "classification", 75, 50))

print("MSE")

print(cross_validate_classification_pso(regression_data_folds, regression_label_folds, .7, 1.49, 1.49,
                                            .1, -.1, [10, 10], len(regression_train_labels[0]),
                                        1, "regression", 75, 50))

