import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define the chunk sizes and execution times
chunk_sizes = [100, 1000, 10000, 100000, 1000000]
execution_times = [3.24352725, 3.2778895, 3.0384715, 2.75777325, 3.595273]

# Plot the data
plt.plot(chunk_sizes, execution_times, marker='o')

# Set the axis labels and title
plt.xlabel('Chunk size')
plt.ylabel('Execution time (s)')
plt.title('Execution time based on chunk size')


# Display the plot
plt.show()