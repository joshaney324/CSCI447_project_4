from BreastCancerSet import BreastCancerSet
from GlassSet import GlassSet
from SoyBeanSet import SoyBeanSet
from Abalone import AbaloneSet
from Hardware import MachineSet
from HelperFunctions import binary_encoding, test_classification_dataset, test_regression_dataset
from Fold_functions import get_tune_folds, get_folds_classification, get_folds_regression
from HyperparameterTuneBackprop import *
from ForestFires import ForestFiresSet
from GeneticAlgorithm import GeneticAlgorithm
from Network import Network
from src.DiffEvolution import DiffEvolution
from ParticleSwarm import ParticleSwarm

# The process for testing a dataset will be the same for all. The comments on the first will apply to the rest

# Soy

# print("abalone")

# No layers

# Set up dataset class and collect data and labels
soy = SoyBeanSet()
# test_classification_dataset(soy)

# breast = BreastCancerSet()
# test_classification_dataset(breast)

# glass = GlassSet(7)
# test_classification_dataset(glass)

# abalone = AbaloneSet()
# test_regression_dataset(abalone)




# # Get folds before getting hold out tune fold
data = soy.get_data()
labels = soy.get_labels()
labels = labels.reshape(-1, 1)
labels = binary_encoding(labels, [0])

data_folds, label_folds = get_folds_classification(data, labels, 10)
#
# # Get tuning fold
test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)


min_velocity = -0.1
max_velocity = 0.1
pso = ParticleSwarm(50, 0.7, 1.49, 1.49, max_velocity, min_velocity, len(train_data[0]), len(train_labels[0]), [10], "classification", train_data,
                      train_labels)
# ga = GeneticAlgorithm(0.08, 0.9, 50, 400, len(train_data[0]), 1, [], "regression", train_data,
#                       train_labels)
# de = DiffEvolution(0.05, 0.9, 50, 1, len(train_data[0]), 1, [], "regression", train_data,
#                       train_labels)
#
pso_weight_vector = pso.train(100)

network = Network(0, [10], len(train_data[0]), len(train_labels[0]), "classification", [])
network.update_weights(pso_weight_vector)
print(network.fitness_function(test_data, test_labels))

# de_weight_vector = de.train(50)
#
# network = Network(0, [], len(train_data[0]), 1, "regression", [])
# network.update_weights(de_weight_vector)
# print(network.fitness_function(test_data, test_labels))
#
#
# weight_vector = ga.train(25)
#
# network = Network(0, [], len(train_data[0]), 1, "regression", [])
# network.update_weights(weight_vector)
# print(network.fitness_function(test_data, test_labels))



# data = soy.get_data()
# labels = soy.get_labels()
# labels = labels.reshape(-1, 1)
# # labels = binary_encoding(labels, [0])
#
# # Get folds before getting hold out tune fold
# data_folds, label_folds = get_folds_regression(data, labels, 10)
#
# # Get tuning fold
# test_data, test_labels, train_data, train_labels = get_tune_folds(data_folds, label_folds)
#
# ga = GeneticAlgorithm(0.08, 0.9, 50, 200, len(train_data[0]), 1, [10, 10], "regression", train_data,
#                       train_labels)
# de = DiffEvolution(0.05, 0.9, 50, 1, len(train_data[0]), 1, [10, 10], "regression", train_data,
#                    train_labels)
#
# # de_weight_vector = de.train(50)
# #
# # network = Network(2, [10, 10], len(train_data[0]), 1, "regression", [])
# # network.update_weights(de_weight_vector)
# # print(network.fitness_function(test_data, test_labels))
#
#
# weight_vector = ga.train(25)
#
# network = Network(2, [10, 10], len(train_data[0]), 1, "regression", [])
# network.update_weights(weight_vector)
# print(network.fitness_function(test_data, test_labels))
#

