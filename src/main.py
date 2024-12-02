from BreastCancerSet import BreastCancerSet
from GlassSet import GlassSet
from SoyBeanSet import SoyBeanSet
from Abalone import AbaloneSet
from Hardware import MachineSet
from HelperFunctions import binary_encoding
from Fold_functions import get_tune_folds, get_folds_classification, get_folds_regression
from HyperparameterTune import hyperparameter_tune_classification, hyperparameter_tune_regression
from CrossValidateFunctions import cross_validate_classification, cross_validate_regression
from ForestFires import ForestFiresSet
from GeneticAlgorithm import GeneticAlgorithm
from Network import Network
from src.DiffEvolution import DiffEvolution

# The process for testing a dataset will be the same for all. The comments on the first will apply to the rest

# Soy

print("Soy Bean")

# No layers

# Set up dataset class and collect data and labels
soy = GlassSet(7)
data = soy.get_data()
labels = soy.get_labels()
labels = labels.reshape(-1, 1)
labels = binary_encoding(labels, [0])

# Get folds before getting hold out tune fold
data_folds, label_folds = get_folds_classification(data, labels, 10)

# Get tuning fold
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)

ga = GeneticAlgorithm(0.1, 20, 1, len(train_data[0]), len(train_labels[0]), [], "classification", train_data,
                      train_labels)
de = DiffEvolution(0.1, 0.4, 20, 1, len(train_data[0]), len(train_labels[0]), [], "classification", train_data,
                      train_labels)

de_weight_vector = de.train(200)

weight_vector = ga.train(10, 200)

network = Network(0, [], len(train_data[0]), len(train_labels[0]), "classification", [])
network.update_weights(weight_vector)
print(network.fitness_function(test_data, test_labels))
network.update_weights(de_weight_vector)
print(network.fitness_function(test_data, test_labels))

