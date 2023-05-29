import matplotlib.pyplot as plt
import numpy as np

# Data
serial_output = [108.498990, 109.261769, 110.026139, 109.491663]
opencl_output = [0.044924, 0.044626, 0.045300, 0.045078]
average_opencl = np.mean(opencl_output)
average_serial = np.mean(serial_output)


# Calculate the difference
difference = average_serial - average_opencl
# Plotting
x = np.arange(len(serial_output))

# Plotting the data points
plt.scatter(x, serial_output, color='red', label='Serial Output')
plt.scatter(x, opencl_output, color='blue', label='OpenCL Output')

# Plotting the average line
plt.plot(x, [average_opencl] * len(x), color='green', linestyle='--', label='Average OpenCL')
plt.plot(x, [average_serial] * len(x), color='purple', linestyle='--', label='Average Serial')



# Plotting the difference line
line_x = x[0] + (x[-1] - x[0]) / 2  # x-coordinate for the vertical line
plt.plot([line_x, line_x], [average_serial,average_opencl], color='orange', linestyle=':', label='Difference')

# Adding text for difference
text_x = line_x
text_y = (average_serial+average_opencl) / 2
plt.text(text_x, text_y, f'Difference: {difference:.4f}', ha='right', va='center', color='black')
# Set labels and title
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title('Comparison of Serial Output and OpenCL Output')

# Add legend
plt.legend()

# Display the graph
plt.show()