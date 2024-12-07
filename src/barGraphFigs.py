
# Data
import matplotlib.pyplot as plt
import numpy as np

# Soy Bean

# Data for the bar graph
data = [
    (0.951, 0.763, 0.705),
    (0.88, 0.795, 0.705),
    (0.9574, 0.8175, 0.8275),
    (0.99, 0.98, 0.708)
]

labels = [
    "Genetic Algorithm",
    "Differential Evolution",
    "Particle Swarm Optimization",
    "Backpropagation"
]

# Bar width and positions
bar_width = 0.2
x = np.arange(len(labels))

# Colors for each value category
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot bars
fig, ax = plt.subplots(figsize=(10, 6))
for i, color in enumerate(colors):
    values = [group[i] for group in data]
    ax.bar(x + i * bar_width, values, width=bar_width, label=f"{i} Hidden Layer(s)", color=color)

# Customizing the graph
ax.set_xticks(x + bar_width)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel("Accuracy Values")
ax.set_title("Comparison of Accuracy Values on the Soy Bean Dataset Across Neural Network Training Algorithms")
ax.legend(title="Hidden Layers", loc="upper right")
ax.set_ylim(0, 1.4)

plt.tight_layout()
plt.savefig("C:\\Users\\josh.aney\\OneDrive\\Documents\\CSCI447\\ml_project4\\figures\\SoyAccuracy.png", dpi=300, bbox_inches='tight')
plt.show()


# Breast Cancer

# Data for the bar graph
data = [
    (0.846, 0.828, 0.659),
    (0.873, 0.938, 0.649),
    (0.905, 0.962, 0.966),
    (0.896, 0.950, 0.649)
]

labels = [
    "Genetic Algorithm",
    "Differential Evolution",
    "Particle Swarm Optimization",
    "Backpropagation"
]

# Bar width and positions
bar_width = 0.2
x = np.arange(len(labels))

# Colors for each value category
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot bars
fig, ax = plt.subplots(figsize=(10, 6))
for i, color in enumerate(colors):
    values = [group[i] for group in data]
    ax.bar(x + i * bar_width, values, width=bar_width, label=f"{i} Hidden Layer(s)", color=color)

# Customizing the graph
ax.set_xticks(x + bar_width)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel("Accuracy Values")
ax.set_title("Comparison of Accuracy Values on the Breast Cancer Dataset Across Neural Network Training Algorithms")
ax.legend(title="Hidden Layers", loc="upper right")
ax.set_ylim(0, 1.4)

plt.tight_layout()
plt.savefig("C:\\Users\\josh.aney\\OneDrive\\Documents\\CSCI447\\ml_project4\\figures\\BreastCancerAccuracy.png", dpi=300, bbox_inches='tight')
plt.show()

# Glass

# Data for the bar graph
data = [
    (0.798, 0.799, 0.781),
    (0.842, 0.820, 0.781),
    (0.87, 0.8175, 0.8275),
    (0.874, 0.842, 0.806)
]

labels = [
    "Genetic Algorithm",
    "Differential Evolution",
    "Particle Swarm Optimization",
    "Backpropagation"
]

# Bar width and positions
bar_width = 0.2
x = np.arange(len(labels))

# Colors for each value category
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot bars
fig, ax = plt.subplots(figsize=(10, 6))
for i, color in enumerate(colors):
    values = [group[i] for group in data]
    ax.bar(x + i * bar_width, values, width=bar_width, label=f"{i} Hidden Layer(s)", color=color)

# Customizing the graph
ax.set_xticks(x + bar_width)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel("Accuracy Values")
ax.set_title("Comparison of Accuracy Values on the Glass Dataset Across Neural Network Training Algorithms")
ax.legend(title="Hidden Layers", loc="upper right")
ax.set_ylim(0, 1.4)

plt.tight_layout()
plt.savefig("C:\\Users\\josh.aney\\OneDrive\\Documents\\CSCI447\\ml_project4\\figures\\GlassAccuracy.png", dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------ PRECISION ----------------------------
#Soy

# Data for the bar graph
data = [
    (0.878, 0.557, 0.415),
    (0.8, 0.65, 0.415),
    (0.949, 0.719, 0.739),
    (0.989, 0.975, 0.410)
]

labels = [
    "Genetic Algorithm",
    "Differential Evolution",
    "Particle Swarm Optimization",
    "Backpropagation"
]

# Bar width and positions
bar_width = 0.2
x = np.arange(len(labels))

# Colors for each value category
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot bars
fig, ax = plt.subplots(figsize=(10, 6))
for i, color in enumerate(colors):
    values = [group[i] for group in data]
    ax.bar(x + i * bar_width, values, width=bar_width, label=f"{i} Hidden Layer(s)", color=color)

# Customizing the graph
ax.set_xticks(x + bar_width)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel("Precision Values")
ax.set_title("Comparison of Precision Values on the Soy Bean Dataset Across Neural Network Training Algorithms")
ax.legend(title="Hidden Layers", loc="upper right")
ax.set_ylim(0, 1.4)

plt.tight_layout()
plt.savefig("C:\\Users\\josh.aney\\OneDrive\\Documents\\CSCI447\\ml_project4\\figures\\SoyPrecision.png", dpi=300, bbox_inches='tight')
plt.show()


# Breast Cancer

# Data for the bar graph
data = [
    (0.845, 0.849, 0.649),
    (0.863, 0.940, 0.649),
    (0.892, 0.958, 0.960),
    (0.886, 0.950, 0.649)
]

labels = [
    "Genetic Algorithm",
    "Differential Evolution",
    "Particle Swarm Optimization",
    "Backpropagation"
]

# Bar width and positions
bar_width = 0.2
x = np.arange(len(labels))

# Colors for each value category
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot bars
fig, ax = plt.subplots(figsize=(10, 6))
for i, color in enumerate(colors):
    values = [group[i] for group in data]
    ax.bar(x + i * bar_width, values, width=bar_width, label=f"{i} Hidden Layer(s)", color=color)

# Customizing the graph
ax.set_xticks(x + bar_width)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel("Precision Values")
ax.set_title("Comparison of Precision Values on the Breast Cancer Dataset Across Neural Network Training Algorithms")
ax.legend(title="Hidden Layers", loc="upper right")
ax.set_ylim(0, 1.4)

plt.tight_layout()
plt.savefig("C:\\Users\\josh.aney\\OneDrive\\Documents\\CSCI447\\ml_project4\\figures\\BreastCancerPrecision.png", dpi=300, bbox_inches='tight')
plt.show()

# Glass

# Data for the bar graph
data = [
    (0.450, 0.415, 0.356),
    (0.487, 0.415, 0.356),
    (0.604, 0.642, 0.468),
    (0.874, 0.842, 0.806)
]

labels = [
    "Genetic Algorithm",
    "Differential Evolution",
    "Particle Swarm Optimization",
    "Backpropagation"
]

# Bar width and positions
bar_width = 0.2
x = np.arange(len(labels))

# Colors for each value category
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot bars
fig, ax = plt.subplots(figsize=(10, 6))
for i, color in enumerate(colors):
    values = [group[i] for group in data]
    ax.bar(x + i * bar_width, values, width=bar_width, label=f"{i} Hidden Layer(s)", color=color)

# Customizing the graph
ax.set_xticks(x + bar_width)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel("Precision Values")
ax.set_title("Comparison of Precision Values on the Glass Dataset Across Neural Network Training Algorithms")
ax.legend(title="Hidden Layers", loc="upper right")
ax.set_ylim(0, 1.4)

plt.tight_layout()
plt.savefig("C:\\Users\\josh.aney\\OneDrive\\Documents\\CSCI447\\ml_project4\\figures\\GlassPrecision.png", dpi=300, bbox_inches='tight')
plt.show()

#######################################################################################################################
# -----------------------------------------------REGRESSION------------------------------------------------------------
#######################################################################################################################


# Soy Bean

# Data for the bar graph
data = [
    (1.09, 1.50, .99),
    (5.33, 1.011, 1.1),
    (.513, .487, .498),
    (0.424, 0.545, 0.766)
]

labels = [
    "Genetic Algorithm",
    "Differential Evolution",
    "Particle Swarm Optimization",
    "Backpropagation"
]

# Bar width and positions
bar_width = 0.2
x = np.arange(len(labels))

# Colors for each value category
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot bars
fig, ax = plt.subplots(figsize=(10, 6))
for i, color in enumerate(colors):
    values = [group[i] for group in data]
    ax.bar(x + i * bar_width, values, width=bar_width, label=f"{i} Hidden Layer(s)", color=color)

# Customizing the graph
ax.set_xticks(x + bar_width)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel("Relative Mean Squared Error Values")
ax.set_title("Comparison of Relative Mean Squared Error Values on the Abalone Dataset Across Neural Network Training Algorithms")
ax.legend(title="Hidden Layers", loc="upper right")
ax.set_ylim(0, 5.5)

plt.tight_layout()
plt.savefig("C:\\Users\\josh.aney\\OneDrive\\Documents\\CSCI447\\ml_project4\\figures\\AbaloneMSE.png", dpi=300, bbox_inches='tight')
plt.show()


# Machine

# Data for the bar graph
data = [
    (.522, 1.11, 1.08),
    (1.52, 1.477, 1.51),
    (1.248, .956, .936),
    (0.233, 1.047, 1.052)
]

labels = [
    "Genetic Algorithm",
    "Differential Evolution",
    "Particle Swarm Optimization",
    "Backpropagation"
]

# Bar width and positions
bar_width = 0.2
x = np.arange(len(labels))

# Colors for each value category
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot bars
fig, ax = plt.subplots(figsize=(10, 6))
for i, color in enumerate(colors):
    values = [group[i] for group in data]
    ax.bar(x + i * bar_width, values, width=bar_width, label=f"{i} Hidden Layer(s)", color=color)

# Customizing the graph
ax.set_xticks(x + bar_width)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel("Mean Squared Error Values")
ax.set_title("Comparison of Mean Squared Error Values on the Relative CPU Performance Dataset Across Neural Network Training Algorithms")
ax.legend(title="Hidden Layers", loc="upper right")
ax.set_ylim(0, 1.8)

plt.tight_layout()
plt.savefig("C:\\Users\\josh.aney\\OneDrive\\Documents\\CSCI447\\ml_project4\\figures\\MachineMSE.png", dpi=300, bbox_inches='tight')
plt.show()

# Forest

# Data for the bar graph
data = [
    (1.103, 1.084, 1.085),
    (1.094, 1.085, 1.100),
    (1.007, 1.0199, 1.0147),
    (1.180, 1.109, 1.109)
]

labels = [
    "Genetic Algorithm",
    "Differential Evolution",
    "Particle Swarm Optimization",
    "Backpropagation"
]

# Bar width and positions
bar_width = 0.2
x = np.arange(len(labels))

# Colors for each value category
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot bars
fig, ax = plt.subplots(figsize=(10, 6))
for i, color in enumerate(colors):
    values = [group[i] for group in data]
    ax.bar(x + i * bar_width, values, width=bar_width, label=f"{i} Hidden Layer(s)", color=color)

# Customizing the graph
ax.set_xticks(x + bar_width)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylabel("Mean Squared Error Values")
ax.set_title("Comparison of Mean Squared Error Values on the Forest Fire Dataset Across Neural Network Training Algorithms")
ax.legend(title="Hidden Layers", loc="upper right")
ax.set_ylim(0, 1.6)

plt.tight_layout()
plt.savefig("C:\\Users\\josh.aney\\OneDrive\\Documents\\CSCI447\\ml_project4\\figures\\ForestMSE.png", dpi=300, bbox_inches='tight')
plt.show()

