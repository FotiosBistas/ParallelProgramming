import matplotlib.pyplot as plt

# Metrics for the number of threads used, starting from 1
metrics = [8.55820875, 6.0897705, 4.638502, 3.4742885]
chunk_metrics = [8.6111195, 4.470951, 3.511062, 3.306737]

# Create a list of the number of threads used
num_threads = list(range(1, len(metrics) + 1))

# Create a line plot with the number of threads on the x-axis and the metrics on the y-axis
plt.plot(num_threads, metrics, marker='o', label='Original Metrics')
plt.plot(num_threads, chunk_metrics, marker='o', label='Chunk Metrics')

# Add labels to the x and y axes
plt.xlabel('Number of threads')
plt.ylabel('Time (seconds)')

# Add a title to the plot
plt.title('Time change based on the increase of threads')

# Set the x-axis ticks to integers from 1 to 4
plt.xticks(num_threads)

# Add a legend to differentiate the two metrics
plt.legend()

# Show the plot
plt.show()