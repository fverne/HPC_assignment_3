import matplotlib.pyplot as plt

# Data
sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
trans_times = [0.823021, 0.908852, 5.554914, 10.951042, 10.219097, 11.301041, 11.498928, 11.426926, 6.202936, 14.681101]
total_times = [7.467031, 7.707834, 11.756897, 14.395952, 15.083075, 18.498898, 21.031857, 24.595976, 23.999929, 39.447142]

# Calculate the ratio of transfer time over total time
ratio = [t_time / tot_time for t_time, tot_time in zip(trans_times, total_times)]

# Set up the bar width
bar_width = 0.35

# Set up the index for groups
index = range(len(sizes))

# Plotting
fig, ax1 = plt.subplots()

# Create bars for Transfer Time
trans_bar = ax1.bar(index, trans_times, bar_width, label='Transfer Time', color='b')

# Create bars for Total Time
total_bar = ax1.bar([i + bar_width for i in index], total_times, bar_width, label='Total Time', color='g')

# Set the x-axis ticks to be the center between the two bars
ax1.set_xticks([i + bar_width / 2 for i in index])
ax1.set_xticklabels(sizes)

# Adding labels and title
ax1.set_xlabel('Size')
ax1.set_ylabel('Time (ms)', color='b')
ax1.set_title('Running Time by Matrix Size of mnk_offload')

# Plotting the line chart for the ratio
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.plot(index, ratio, label='Transfer/Total Time Ratio', color='r', marker='o')
ax2.set_ylabel('Ratio', color='r')

# Adding a legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show plot with a tight layout
plt.tight_layout()
plt.savefig('q4-transtime-mnk.png')