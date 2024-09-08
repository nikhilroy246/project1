import numpy as np
import matplotlib.pyplot as plt

# Define the attack stats (e.g., accuracy, loss, etc.)
attack_stats = [0.9, 0.8, 0.7, 0.6, 0.5]

# Define the labels for the x-axis
labels = ['Attack 1', 'Attack 2', 'Attack 3', 'Attack 4', 'Attack 5']

# Create a bar chart of the attack stats
plt.bar(labels, attack_stats)

# Set the title and labels
plt.title('Attack Stats')
plt.xlabel('Attack Type')
plt.ylabel('Accuracy')

# Save the image to a file
plt.savefig('attack_stats.png')

# Display the image
plt.show()