import numpy as np
import matplotlib.pyplot as plt

# Mapping of algorithms to their corresponding data file paths
files = {
    "PSO": "./numpy_files/optimization_data_pso.npy",
    "GA": "./numpy_files/optimization_data_ga.npy",
    "BAT": "./numpy_files/optimization_data_bat.npy"
}

# Define colors for each algorithm
colors = {
    "PSO": "orange",
    "GA": "green",
    "BAT": "blue"
}

# Plot the learning curves
for label, file in files.items():
    data = np.load(file, allow_pickle=True)
    evaluations, distances = data[0], data[1]
    plt.plot(evaluations, distances, label=label, color=colors[label])

# Set plot labels and title
plt.xlabel("Evaluation")
plt.ylabel("Fitness Value")
plt.title("Learning Curves of Optimization Algorithms")

# Display legend and grid
plt.legend()
plt.grid(True)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
