import numpy as np
import matplotlib.pyplot as plt

# Create Alignment dataset
height = [-.16, .38, .37, .52, .7, .74, .8, .36, .78, .56, -.19, .35, .59, .34, .57, .75]
bars = ('High jump', 'Long jump', 'Pole vault', 'Triple jump', 'Discus', 'Hammer throw', 'Javelin', 'Shot put',
        '10K', '5K', '3K', '1500m', '800m', '400m', '200m', '100m')
x_pos = np.arange(len(bars))

# Create bars and choose color
plt.bar(x_pos, height, color=(0.5, 0.1, 0.5, 0.6))
plt.subplots_adjust(left=0.105, bottom=0.3, right=0.95, top=0.8)

# Add title and axis names
plt.title('Alignment values')
plt.xlabel('Event')
plt.xticks(rotation=90)
plt.ylabel('Alignment')
# Create names on the x axis
plt.xticks(x_pos, bars)
# Show graph
plt.savefig("Barplot_alignment")
plt.show()

# Create Inconsistency dataset
height = [1.59, 1.32, 2.58, 1.73, .93, 1.83, .61, 1.09, 1.42, .95, .99, 1.09, .82, .75, .82, 1.04]
bars = ('High jump', 'Long jump', 'Pole vault', 'Triple jump', 'Discus', 'Hammer throw', 'Javelin', 'Shot put',
        '10K', '5K', '3K', '1500m', '800m', '400m', '200m', '100m')
x_pos = np.arange(len(bars))

# Create bars and choose color
plt.bar(x_pos, height, color=(0.5, 0.1, 0.5, 0.6))
plt.subplots_adjust(left=0.105, bottom=0.3, right=0.95, top=0.8)

# Add title and axis names
plt.title('Inconsistency values')
plt.xlabel('Event')
plt.xticks(rotation=90)
plt.ylabel('Inconsistency')
# Create names on the x axis
plt.xticks(x_pos, bars)
# Show graph
plt.savefig("Barplot_Inconsistency")
plt.show()

# Create Geographic Dispersion dataset
height = [8.9, 8.3, 9.9, 9, 8.5, 8.9, 9.3, 8.6, ]
bars = ("M high jump", "W high jump", "M long jump", "W long jump", "M pole vault", "W pole vault", "M triple jump", "W triple jump",
                  "M discus", "W discus", "M hammer throw", "W hammer throw", "M javelin", "W javelin", "M shot put", "W shot put")
x_pos = np.arange(len(bars))

# Create bars and choose color
plt.bar(x_pos, height, color=(0.5, 0.1, 0.5, 0.6))
plt.subplots_adjust(left=0.105, bottom=0.3, right=0.95, top=0.8)

# Add title and axis names
plt.title('Inconsistency values')
plt.xlabel('Event')
plt.xticks(rotation=90)
plt.ylabel('Inconsistency')
# Create names on the x axis
plt.xticks(x_pos, bars)
# Show graph
plt.savefig("Barplot_Inconsistency")
plt.show()