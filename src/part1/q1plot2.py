
import matplotlib.pyplot as plt
import numpy as np

grid_sizes = [100, 200, 500, 1000, 2000, 5000]

# Data for mnk_omp
iterations_mnk_omp_1_thread = [14201.118, 16508.646, 15411.208, 16393.493, 10076.269, 4638.621]
iterations_mnk_omp_2_threads = [27739.161, 47518.994, 62975.700, 75591.915, 78732.497, 82640.542]
iterations_mnk_omp_4_threads = [39518.785, 56005.024, 49755.920, 53965.578, 38311.945, 15909.741]
iterations_mnk_omp_6_threads = [51985.150, 78642.776, 73542.283, 81346.205, 56368.024, 23687.485]
iterations_mnk_omp_8_threads = [60349.847, 98604.727, 95936.024, 108132.722, 76520.789, 31491.209]
iterations_mnk_omp_10_threads = [67993.320, 117381.272, 118902.415, 133704.176, 76200.477, 38735.952]
iterations_mnk_omp_12_threads = [71758.104, 128650.470, 139742.753, 158524.973, 95543.032, 46251.222]
iterations_mnk_omp_14_threads = [80163.264, 145943.574, 160412.625, 183727.313, 114422.442, 52650.272]
iterations_mnk_omp_16_threads = [81769.493, 160566.032, 180196.213, 207939.310, 133181.348, 48007.722]

# Data for lib
iterations_lib_1_thread = [29793.537, 35214.949, 39167.180, 40763.948, 41538.271, 42436.582]
iterations_lib_2_threads = [27739.161, 47518.994, 62975.700, 75591.915, 78732.497, 82640.542]
iterations_lib_4_threads = [31787.750, 75146.652, 109924.204, 133354.804, 150908.954, 163458.864]
iterations_lib_6_threads = [55115.934, 122319.039, 181618.623, 217850.730, 229419.418, 244444.781]
iterations_lib_8_threads = [48706.284, 133982.321, 214801.214, 266529.264, 294034.488, 324654.708]
iterations_lib_10_threads = [73007.840, 183376.346, 292171.262, 355950.216, 373260.476, 401482.534]
iterations_lib_12_threads = [60719.361, 172299.774, 311656.427, 395728.980, 439253.727, 482378.461]
iterations_lib_14_threads = [75375.960, 205509.019, 391021.475, 486277.067, 510379.786, 554329.767]
iterations_lib_16_threads = [67047.808, 187018.319, 384814.539, 485318.683, 573582.431, 633904.862]

# Various thread numbers
thread_numbers = [1, 2, 4, 6, 8, 10, 12, 14, 16]

# Get unique labels for each combination of implementation and thread number
labels_mnk_omp = [f'mnk_omp - {n} thread(s)' for n in thread_numbers]
labels_lib = [f'lib - {n} thread(s)' for n in thread_numbers]
labels = labels_mnk_omp + labels_lib

# Reshape the data for bar plotting
data_mnk_omp = np.array([iterations_mnk_omp_1_thread, iterations_mnk_omp_2_threads, iterations_mnk_omp_4_threads,
                         iterations_mnk_omp_6_threads, iterations_mnk_omp_8_threads, iterations_mnk_omp_10_threads,
                         iterations_mnk_omp_12_threads, iterations_mnk_omp_14_threads, iterations_mnk_omp_16_threads])
data_lib = np.array([iterations_lib_1_thread, iterations_lib_2_threads, iterations_lib_4_threads,
                     iterations_lib_6_threads, iterations_lib_8_threads, iterations_lib_10_threads,
                     iterations_lib_12_threads, iterations_lib_14_threads, iterations_lib_16_threads])


# Plotting with shades of blue for mnk_omp and shades of red for lib
mnk_omp_colors = plt.cm.Blues(np.linspace(0.3, 1, len(thread_numbers)))
lib_colors = plt.cm.Reds(np.linspace(0.3, 1, len(thread_numbers)))

for i in range(len(thread_numbers)):
    plt.plot(grid_sizes, data_mnk_omp[i, :], label=labels_mnk_omp[i], linestyle='-', color=mnk_omp_colors[i])

for i in range(len(thread_numbers)):
    plt.plot(grid_sizes, data_lib[i, :], label=labels_lib[i], linestyle='--', color=lib_colors[i])

plt.xlabel('Grid Sizes')
plt.xscale("log")
plt.ylabel('MFlops/s [Seconds]')
plt.title('Comparison of Various Thread Numbers')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Adjust layout to prevent cropping of labels
plt.savefig("plots/q1plot2")
plt.show()