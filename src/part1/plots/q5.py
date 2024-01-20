import matplotlib.pyplot as plt

# Matrix sizes
sizes = [100, 200, 500, 1000, 2000, 5000]

# Performance in MFlops
lib_mflops = [388.080, 1826.155, 16633.856, 32481.459, 40376.248, 42072.664]
lib_offload_mflops = [165.514, 674.093, 8826.254, 49824.892, 513949.380, 3333218.305]

# Speedup values for 'lib' and 'lib_offload'
speedup_lib = [0.0919, 0.4074, 7.3895, 14.9582, 21.9770, 89.3015]
speedup_lib_offload = [0.0387, 0.1495, 3.9048, 22.9353, 280.4511, 7083.9945]

# Create figure and first axis
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot performance in MFlops on the first axis
ax1.plot(sizes, lib_mflops, 'r--x', label='Lib (MFlops)')
ax1.plot(sizes, lib_offload_mflops, 'g-.^', label='Lib_offload (MFlops)')
ax1.set_xlabel('Matrix Size')
ax1.set_ylabel('Performance (MFlops)')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend(loc='upper left')
ax1.grid(True)

# Create second axis for speedup
ax2 = ax1.twinx()
ax2.plot(sizes, speedup_lib, 'r:x', label='Lib Speedup')
ax2.plot(sizes, speedup_lib_offload, 'g:^', label='Lib_offload Speedup')
ax2.set_ylabel('Speedup')
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend(loc='upper right')

# Title
plt.title('Performance and Speedup Comparison between lib and lib_offload')

# Show the plot
plt.savefig('q5.png')