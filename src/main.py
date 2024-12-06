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

abalone = AbaloneSet()
forest = ForestFiresSet()
machine = MachineSet()
soy = SoyBeanSet()
breast = BreastCancerSet()
glass = GlassSet(7)
#
# print("---------------------------------ABALONE----------------------------")
# test_regression_dataset(abalone, [5], [15, 10])
# print()
#
# print("---------------------------------MACHINE----------------------------")
# test_regression_dataset(machine, [5], [2, 5])
# print()
#
# print("---------------------------------FOREST----------------------------")
# test_regression_dataset(forest, [15], [15, 5])
# print()




print("---------------------------------SOY----------------------------")
test_classification_dataset(soy, [15], [5, 10])
print()

print("---------------------------------BREAST-CANCER----------------------------")
test_classification_dataset(breast, [8], [2, 3])
print()

print("---------------------------------GLASS----------------------------")
test_classification_dataset(glass, [8], [4, 6])
print()



