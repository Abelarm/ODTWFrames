import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('bmh')
import numpy as np


CNN_rational = [0.93, 0.94, 0.92]
ResNet_rational = [0.86, 0.94, 0.94]

CNN_gunpoint = [0.74, 0.77, 0.75]
ResNet_gunpoint = [0.72, 0.77, 0.75]

ax = plt.gca()
base_line, = ax.plot(CNN_rational, '-x', label='CNN - rational')

for a, b in zip(range(3), CNN_rational):
    plt.text(a-0.05, b+0.001, str(b), size=10, color=base_line.get_color(), alpha=0.8)

base_line, = plt.plot(ResNet_rational, '-o', label='ResNet - rational')

for a, b in zip(range(3), ResNet_rational):
    plt.text(a-0.05, b-0.01, str(b), size=10, color=base_line.get_color(), alpha=0.8)

base_line, = plt.plot(CNN_gunpoint, '-^', label='CNN - gunpoint')
for a, b in zip(range(3), CNN_gunpoint):
    plt.text(a-0.05, b+0.001, str(b), size=10, color=base_line.get_color(), alpha=0.8)

base_line, = plt.plot(ResNet_gunpoint, '-v', label='ResNet - gunpoint')

for a, b in zip(range(3), ResNet_gunpoint):
    plt.text(a-0.05, b-0.01, str(b), size=10, color=base_line.get_color(), alpha=0.8)

plt.xticks(np.arange(3), labels=['small', 'med', 'large'])
plt.ylabel('AUC score')
plt.xlabel('Network size')

plt.legend()
plt.grid(True)
plt.show()
