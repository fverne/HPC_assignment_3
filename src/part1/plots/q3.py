import matplotlib.pyplot as plt

# Your data: pairs of (x, y) points
data = [
    (4, 895672.365),
    (8, 936349.028),
    (12, 712190.446),
    (16, 834244.421),
    (20, 731975.079),
    (24, 506684.608),
    (28, 468813.886),
    (32, 781066.592)
]

# Unzip the data into x and y coordinates
x, y = zip(*data)

# Create the plot
plt.figure(figsize=(10, 5))  # Set the figure size
plt.plot(x, y, 'o-', color='blue')  # Plot x vs y with blue line and circle markers
plt.title('Performance of matmult_blk_offload')  # The title of the plot
plt.xlabel('Block Size')  # X-axis label
plt.ylabel('MFlops')  # Y-axis label
plt.grid(True)  # Show grid
# plt.show()  # Display the plot
plt.savefig("q3.png")