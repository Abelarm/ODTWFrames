import matplotlib.pyplot as plt

CNN = {2e06: 0.8510, 7e05: 0.8396, 8.5e04: 0.7862}
ResNet = {1.8e06: 0.8570, 1e06: 0.8501, 1.8e04: 0.8251}
# OneDResNet = {1.28e05: 0.5903, 5e04: 0.5909, 1e04: 0.5828}
# RNN = {1.5e04: 0.5830, 5e04: 0.5828, 1e04: 0.5828}

keys = list(map(float, CNN.keys()))
values = list(map(float, CNN.values()))
plt.plot(keys, values, 's-', label='CNN')

# zip joins x and y coordinates in pairs
for x, y in zip(keys, values):
    label = "{:.4f}".format(y)

    # this method is called for each point
    plt.annotate(label,  # this is the text
                 (x, y),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, 10),  # distance from text to points (x,y)
                 ha='center')  # horizontal alignment can be left, right or center

keys = list(map(float, ResNet.keys()))
values = list(map(float, ResNet.values()))
plt.plot(keys, values, '^-', label='ResNet')

# zip joins x and y coordinates in pairs
for x, y in zip(keys, values):
    label = "{:.4f}".format(y)

    # this method is called for each point
    plt.annotate(label,  # this is the text
                 (x, y),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, 10),  # distance from text to points (x,y)
                 ha='center')  # horizontal alignment can be left, right or center

"""
keys = list(map(float, OneDResNet.keys()))
values = list(map(float, OneDResNet.values()))
plt.plot(keys, values, 'o-', label='1D_ResNet')

# zip joins x and y coordinates in pairs
for x, y in zip(keys, values):
    label = "{:.4f}".format(y)

    # this method is called for each point
    plt.annotate(label,  # this is the text
                 (x, y),  # this is the point to label
                 textcoords="offset points",  # how to position the text
                 xytext=(0, 10),  # distance from text to points (x,y)
                 ha='center')  # horizontal alignment can be left, right or center
"""

plt.xticks([1e04, 1e06, 2e06])
plt.axes().set_xticklabels(['100k', '1M', '2M'])


# plt.yscale('logit')
plt.legend(loc='lower right')
plt.show()
