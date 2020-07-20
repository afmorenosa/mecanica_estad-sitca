"""Atractive neuronal network."""
from patterns.patterns import all_patterns
from neurons import network as nt
import matplotlib.pyplot as plt
import os

try:
    os.mkdir("results")
except FileExistsError:
    pass

NN = nt.network(8, 8)

NN.train(all_patterns[4:7])
NN.config_init_system()

plt.matshow(NN.S_out, cmap="Set1")
print("[{:.2f}%]".format(0.0), NN.E)
plt.title(NN.energy())
plt.savefig("results/A.pdf")

iterations = 10


for i in range(iterations):
    NN.system_step()
    print("[{:.2f}%]".format((i+1)/iterations*100), NN.E)
    plt.close()
    plt.matshow(NN.S_out, cmap="Set1")
    plt.title(NN.energy())
    plt.savefig("results/ite_{}.pdf".format(i))
