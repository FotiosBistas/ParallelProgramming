import matplotlib.pyplot as plt

# Metrics for the number of threads used, starting from 1
metrics = [39.063022, 9.5828655, 6.94163725, 5.1600935]
chunk_metrics = [37.06994975, 19.5018355, 13.333325, 10.347312]

# Create a list of the number of threads used
num_threads = list(range(1, len(chunk_metrics) + 1))

# Create a line plot with the number of threads on the x-axis and the metrics on the y-axis
plt.plot(num_threads, metrics, marker='o', label='Two task implementation')
plt.plot(num_threads, chunk_metrics, marker='o', label='One task implementation')

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