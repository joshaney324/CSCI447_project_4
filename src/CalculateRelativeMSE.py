import numpy as np

from Hardware import MachineSet
from Abalone import AbaloneSet
from ForestFires import ForestFiresSet
from Metric_functions import mean_squared_error

machine = MachineSet()
abalone = AbaloneSet()
forest = ForestFiresSet()

machine_labels = machine.get_labels()
abalone_labels = abalone.get_labels()
forest_labels = forest.get_labels()

machine_mean = 0
for label in machine_labels:
    machine_mean += label

machine_mean /= len(machine_labels)
# print(machine_mean)

predictions = []
for label in machine_labels:
    predictions.append(machine_mean)

predictions = np.array(predictions)
predictions = predictions.reshape(-1, 1)
machine_labels = machine_labels.reshape(-1, 1)
print(mean_squared_error(predictions, machine_labels, len(predictions)))

abalone_mean = 0
for label in abalone_labels:
    abalone_mean += label

abalone_mean /= len(abalone_labels)

# print(abalone_mean)

predictions = []
for label in abalone_labels:
    predictions.append(abalone_mean)

predictions = np.array(predictions)
predictions = predictions.reshape(-1, 1)
abalone_labels = abalone_labels.reshape(-1, 1)

print(mean_squared_error(predictions, abalone_labels, len(predictions)))

forest_mean = 0
for label in forest_labels:
    forest_mean += label

forest_mean /= len(forest_labels)
# print(forest_mean)

predictions = []
for label in forest_labels:
    predictions.append(forest_mean)

predictions = np.array(predictions)
predictions = predictions.reshape(-1, 1)
forest_labels = forest_labels.reshape(-1, 1)


print(mean_squared_error(predictions, forest_labels, len(predictions)))





