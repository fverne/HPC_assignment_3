
import matplotlib.pyplot as plt
import numpy as np

# Data for mnk_offload
iterations_mnk_offload = [  9211.767,
 18295.031,
 36156.111,
 52904.882,
 69662.911,
 85531.726,
101412.767,
116387.663,
130441.803,
122748.658,
157244.198
]


# Data for mkn_offload
iterations_mkn_offload = [ 2363.211,
4247.716,
6955.619,
10030.220,
9709.396,
18542.318,
17973.222,
17193.001,
16678.167,
13985.085,
11503.933,
]


# Various team numbers
thread_numbers = [1, 2, 4, 6, 8, 10, 12, 14, 16, 24, 32]

# Get unique labels for each combination of implementation and team number
labels_mnk_offload = [f'mnk_offload' for n in thread_numbers]
labels_mkn_offload = [f'mkn_offload' for n in thread_numbers]
labels = labels_mnk_offload + labels_mkn_offload

# Reshape the data for bar plotting
data_mnk_offload = np.array([iterations_mnk_offload])
data_mkn_offload = np.array([iterations_mkn_offload])


# Plotting with shades of blue for mnk_offload and shades of red for mkn_offload
mnk_offload_colors = plt.cm.Blues(np.linspace(0.3, 1, len(thread_numbers)))
mkn_offload_colors = plt.cm.Reds(np.linspace(0.3, 1, len(thread_numbers)))

plt.plot(thread_numbers, data_mnk_offload[0, :], label=labels_mnk_offload[0], linestyle='-', color=mnk_offload_colors[0])

plt.plot(thread_numbers, data_mkn_offload[0, :], label=labels_mkn_offload[0], linestyle='--', color=mkn_offload_colors[0])

plt.xlabel('Grid Sizes')
plt.xscale("log")
plt.ylabel('MFlops/s [Seconds]')
plt.title('Comparison of Various Thread Numbers for Teams=114')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(thread_numbers, labels=thread_numbers)  # Display thread numbers on the x-axis
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Adjust layout to prevent cropping of labels
plt.savefig("plots/q2plot3")
plt.show()