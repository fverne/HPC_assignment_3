import matplotlib.pyplot as plt
import numpy as np

gridsize = [
    100, 200, 500, 1000, 2000, 5000
]

threads = [1, 2, 4, 6, 8, 10, 12, 14, 16]


grid_sizes = [100, 200, 500, 1000, 2000, 5000]
mflops_mnk_omp = [14201.118, 16508.646, 15411.208, 16393.493, 10076.269, 4638.621]
mflops_lib = [29556.080, 35145.500, 39062.466, 40403.069, 41299.614, 42005.140]


plt.plot(grid_sizes, mflops_mnk_omp, linestyle='-', label="mnk_omp", color=plt.cm.Blues(np.linspace(0.3, 1, len(threads)))[1])
plt.plot(grid_sizes, mflops_lib, linestyle='-', label="lib", color=plt.cm.Reds(np.linspace(0.3, 1, len(threads)))[1])
# plt.yscale('log')
plt.xscale('log')  # Set x-axis to log scale for better visualization
plt.xlabel('Grid Size')
plt.ylabel('MFlops/s [Seconds]')
plt.title('Grid Size vs MFlops/s [1 thread]')
plt.grid(True)
plt.legend()
plt.savefig("plots/q1plot1")