import matplotlib.pyplot as plt

# Metrics for the number of threads used, starting from 1
metrics = [8.55820875, 6.0897705, 4.638502, 3.4742885]

# Create a list of the number of threads used
num_threads = list(range(1, len(metrics) + 1))

# Create a line plot with the number of threads on the x-axis and the metrics on the y-axis
plt.plot(num_threads, metrics, marker='o')

# Add labels to the x and y axes
plt.xlabel('Number of threads')
plt.ylabel('Time (seconds)')

# Add a title to the plot
plt.title('Time change based on the increase of threads')

# Set the x-axis ticks to integers from 1 to 4
plt.xticks(num_threads)

# Show the plot
plt.show()