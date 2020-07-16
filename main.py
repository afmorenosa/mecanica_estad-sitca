"""Atractive neuronal network."""
from patterns.patterns import pattern_A, pattern_C, pattern_D, all_patterns
from neurons import neuron as nr
import matplotlib.pyplot as plt
import numpy as np


NN = np.array([[nr.neuron([], [], np.sign, -1), nr.neuron([], [], np.sign, -1),
               nr.neuron([], [], np.sign, -1), nr.neuron([], [], np.sign, -1)],
               [nr.neuron([], [], np.sign, -1), nr.neuron([], [], np.sign, -1),
               nr.neuron([], [], np.sign, -1), nr.neuron([], [], np.sign, -1)],
               [nr.neuron([], [], np.sign, -1), nr.neuron([], [], np.sign, -1),
               nr.neuron([], [], np.sign, -1), nr.neuron([], [], np.sign, -1)],
               [nr.neuron([], [], np.sign, -1), nr.neuron([], [], np.sign, -1),
               nr.neuron([], [], np.sign, -1), nr.neuron([], [], np.sign, -1)]]
              )

Patter = nr.pattern_matrix(NN)

nr.train(NN, all_patterns[4:7])
nr.config_init_system(NN)

plt.matshow(nr.pattern_matrix(NN), cmap="Set1")
print("[{:.2f}%]".format(0.0), nr.energy(NN))
plt.title(nr.energy(NN))
plt.savefig("results/A.pdf")

iterations = 50


for i in range(iterations):
    print("[{:.2f}%]".format((i+1)/iterations*100), nr.system_step(NN))
    plt.close()
    plt.matshow(nr.pattern_matrix(NN), cmap="Set1")
    plt.title(nr.energy(NN))
    plt.savefig("results/ite_{}.pdf".format(i))
