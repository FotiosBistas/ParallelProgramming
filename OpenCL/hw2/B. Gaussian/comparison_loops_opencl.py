import matplotlib.pyplot as plt
import numpy as np

# Data
parallel_loops_output = [10.589234, 10.339203, 10.407211, 10.333202]
opencl_output = [0.045156, 0.044903, 0.044802, 0.044574]
average_opencl = np.mean(opencl_output)
average_loops = np.mean(parallel_loops_output)


# Calculate the difference
difference = average_loops - average_opencl
# Plotting
x = np.arange(len(parallel_loops_output))

# Plotting the data points
plt.scatter(x, parallel_loops_output, color='red', label='Parallel loops Output')
plt.scatter(x, opencl_output, color='blue', label='OpenCL Output')

# Plotting the average line
plt.plot(x, [average_opencl] * len(x), color='green', linestyle='--', label='Average OpenCL')
plt.plot(x, [average_loops] * len(x), color='purple', linestyle='--', label='Average Parallel loops')



# Plotting the difference line
line_x = x[0] + (x[-1] - x[0]) / 2  # x-coordinate for the vertical line
plt.plot([line_x, line_x], [average_loops,average_opencl], color='orange', linestyle=':', label='Difference')

# Adding text for difference
text_x = line_x
text_y = (average_loops+average_opencl) / 2
plt.text(text_x, text_y, f'Difference: {difference:.4f}', ha='right', va='center', color='black')
# Set labels and title
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Comparison of Parallel loops Output and OpenCL Output')

# Add legend
plt.legend()

# Display the graph
plt.show()