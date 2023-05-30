import matplotlib.pyplot as plt

# Data from the tables
threads = [1, 2, 3, 4, 6, 8, 10, 12, 14, 15]

# OpenMP parallel loops data
parallel_loops_average = [108.773124, 54.39947425, 37.0724545, 27.85650275, 21.0167265, 15.65917125, 12.675475,
                          10.976052, 10.4172125, 10.4653125]

# OpenMP tasks data
tasks_average = [110.6081145, 55.80442925, 38.59814275, 28.9847925, 20.70226375, 14.870166, 12.28607,
                 10.677438, 12.1798015, 11.84242675]

# Plotting the data
plt.plot(threads, parallel_loops_average, marker='o', label='OpenMP Parallel Loops')
plt.plot(threads, tasks_average, marker='o', label='OpenMP Tasks')

# Set labels and title
plt.xlabel('Threads')
plt.ylabel('Average time (secs)')
plt.title('OpenMP parallel loops and tasks')

# Add legend
plt.legend()

# Display the graph
plt.show()