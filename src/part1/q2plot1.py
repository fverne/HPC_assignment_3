
import matplotlib.pyplot as plt
import numpy as np

grid_sizes = [100, 200, 500, 1000, 2000, 5000]

# Data for mnk_offload
iterations_mnk_offload_1_thread = [18764.300,
40975.910,
53493.570,
53690.500,
46848.791,
37104.305,]
iterations_mnk_offload_2_threads = [ 23643.305,
 63756.523,
 94582.811,
104097.269,
 94181.741,
 70656.206,]
iterations_mnk_offload_4_threads = [ 25201.634,
 88650.029,
166687.960,
192480.180,
154515.109,
133406.633,]
iterations_mnk_offload_6_threads = [ 27829.364,
100182.640,
218645.489,
270841.974,
211229.587,
190979.232,]
iterations_mnk_offload_8_threads = [ 27601.107,
111378.156,
259744.928,
337667.670,
259659.261,
254897.434,]
iterations_mnk_offload_10_threads = [ 27903.911,
111728.321,
286557.212,
386423.159,
304879.003,
307469.929,]
iterations_mnk_offload_12_threads = [ 27907.929,
117994.062,
316723.650,
452242.226,
354622.646,
361441.577,]
iterations_mnk_offload_14_threads = [ 27834.355,
118632.271,
326419.965,
486119.041,
391678.086,
415655.589,]
iterations_mnk_offload_16_threads = [27847.096,
119944.684,
353313.133,
536023.871,
445812.164,
469101.319,
]

# Data for lib
iterations_mkn_offload_1_thread = [1963.027,
 4083.404,
10011.668,
19308.846,
16926.969,
14104.251]
iterations_mkn_offload_2_threads = [ 1951.443,
 4094.409,
10241.753,
20112.484,
30626.185,
22624.687,]
iterations_mkn_offload_4_threads = [ 1885.275,
 3982.431,
10289.075,
19705.142,
29155.585,
31783.306,]
iterations_mkn_offload_6_threads = [ 1842.155,
 3762.033,
10086.817,
19469.700,
27341.161,
58002.520,]
iterations_mkn_offload_8_threads = [ 1841.259,
 3638.630,
 9984.452,
18829.044,
26019.691,
55977.306,]
iterations_mkn_offload_10_threads = [ 1780.407,
 3525.317,
 9685.958,
18435.459,
24680.027,
53580.580,]
iterations_mkn_offload_12_threads = [ 1775.973,
 3430.236,
 9623.420,
17960.973,
24020.515,
52051.393,]
iterations_mkn_offload_14_threads = [ 1717.906,
 3326.545,
 9288.564,
17144.054,
22509.338,
49511.131,]
iterations_mkn_offload_16_threads = [ 1693.511,
 3240.341,
 8992.555,
16454.120,
21239.001,
48410.035,]

# Various thread numbers
thread_numbers = [1, 2, 4, 6, 8, 10, 12, 14, 16]

# Get unique labels for each combination of implementation and thread number
labels_mnk_offload = [f'mnk_offload - {n} thread(s)' for n in thread_numbers]
labels_mkn_offload = [f'mkn_offload - {n} thread(s)' for n in thread_numbers]
labels = labels_mnk_offload + labels_mkn_offload

# Reshape the data for bar plotting
data_mnk_offload = np.array([iterations_mnk_offload_1_thread, iterations_mnk_offload_2_threads, iterations_mnk_offload_4_threads,
                         iterations_mnk_offload_6_threads, iterations_mnk_offload_8_threads, iterations_mnk_offload_10_threads,
                         iterations_mnk_offload_12_threads, iterations_mnk_offload_14_threads, iterations_mnk_offload_16_threads])
data_mkn_offload = np.array([iterations_mkn_offload_1_thread, iterations_mkn_offload_2_threads, iterations_mkn_offload_4_threads,
                     iterations_mkn_offload_6_threads, iterations_mkn_offload_8_threads, iterations_mkn_offload_10_threads,
                     iterations_mkn_offload_12_threads, iterations_mkn_offload_14_threads, iterations_mkn_offload_16_threads])


# Plotting with shades of blue for mnk_offload and shades of red for mkn_offload
mnk_offload_colors = plt.cm.Blues(np.linspace(0.3, 1, len(thread_numbers)))
mkn_offload_colors = plt.cm.Reds(np.linspace(0.3, 1, len(thread_numbers)))

for i in range(len(thread_numbers)):
    plt.plot(grid_sizes, data_mnk_offload[i, :], label=labels_mnk_offload[i], linestyle='-', color=mnk_offload_colors[i])

for i in range(len(thread_numbers)):
    plt.plot(grid_sizes, data_mkn_offload[i, :], label=labels_mkn_offload[i], linestyle='--', color=mkn_offload_colors[i])

plt.xlabel('Grid Sizes')
plt.xscale("log")
plt.ylabel('MFlops/s [Seconds]')
plt.title('Comparison of Various Thread Numbers for Teams=1024')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Adjust layout to prevent cropping of labels
plt.savefig("plots/q2plot1")
plt.show()