import matplotlib.pyplot as plt

# Sizes
sizes = [100, 200, 500, 1000, 2000, 3000, 4000, 5000]

# Running times (Time) for both versions
times_mnk = [8.320808, 8.239031, 31.617880, 35.118103, 267.481089, 860.233068, 2013.257027, 4017.425060]
times_asy = [25.475979, 23.306847, 36.955833, 53.302050, 59.966803, 80.337048, 149.775028, 305.895090]

# MFlops performance for both versions
mflops_mnk = [238.761, 1922.252, 7885.296, 56547.367, 59601.285, 62606.985, 63434.149, 62124.973]
mflops_asy = [78.387, 685.267, 6756.832, 37492.798, 266588.520, 671703.795, 854206.969, 817055.976]

# Set up the plotting layout
fig, ax1 = plt.subplots(figsize=(14, 7))

# Bar chart for running times
bar_width = 0.35
index = range(len(sizes))

bar1 = ax1.bar(index, times_mnk, bar_width, label='mnk_offload Time', color='b')
bar2 = ax1.bar([i + bar_width for i in index], times_asy, bar_width, label='asy_offload Time', color='g')

# Line chart for MFlops performance
ax2 = ax1.twinx()
line1 = ax2.plot(index, mflops_mnk, label='mnk_offload MFlops', color='r', marker='o')
line2 = ax2.plot(index, mflops_asy, label='asy_offload MFlops', color='c', marker='x')

# Labels, title, and legend
ax1.set_xlabel('Matrix Size')
ax1.set_ylabel('Running Time (s)', color='b')
ax2.set_ylabel('MFlops', color='r')
ax1.set_title('Comparison of Running Time and MFlops Performance')
ax1.set_xticks([i + bar_width / 2 for i in index])
ax1.set_xticklabels(sizes)
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='upper left')
ax1.legend(loc='upper right')

# Show plot with a tight layout
plt.tight_layout()
plt.savefig('q4.png')