"""Atractive neuronal network."""
from patterns.patterns import all_patterns
from neurons import network as nt
import matplotlib.pyplot as plt
import os

try:
    os.mkdir("results")
except FileExistsError:
    pass

# try:
    # os.mkdir("let")
# except FileExistsError:
    # pass
#
# for i in range(len(all_patterns)):
    # plt.close()
    # plt.matshow(all_patterns[i], cmap="Set1")
    # plt.savefig("let/letra_{}.pdf".format(i))


def F(x):
    """Sing function."""
    if x >= 0:
        return 1
    else:
        return -1


NN = nt.network(8, 8, F)

NN.train([all_patterns[2]] * 7 + [all_patterns[3]] * 2 + [all_patterns[4]])
NN.config_init_system()

plt.matshow(NN.S_out, cmap="Set1")
print("[{:.2f}%]".format(0.0), NN.E)
plt.title(NN.energy())
plt.savefig("results/A.pdf")

iterations = 10


for i in range(iterations):
    # matrix = open("results/data_{}.data".format(i), "w")
    NN.system_step()
    print("[{:.2f}%]".format((i+1)/iterations*100), NN.E)
    # matrix.write(str(NN.S_out))
    # matrix.close()
    plt.close()
    plt.matshow(NN.S_out, cmap="Set1")
    plt.title(NN.energy())
    plt.savefig("results/ite_{}.pdf".format(i))
