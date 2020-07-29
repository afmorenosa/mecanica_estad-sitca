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

# alpha = []
# errors = []
# number_test = 100

# for i in range(len(all_patterns)):
    # alpha.append((i+1)/NN.N)
    # print("[{:.2f}]%".format((i+1)/len(all_patterns) * 100))
    # errors.append(NN.get_error(all_patterns[0:i+1], number_test))

# plt.clf()
# plt.close()
# plt.plot(alpha, errors)
# plt.grid()
# plt.xlabel(r"$\alpha$")
# plt.ylabel(r"$\epsilon$")
# plt.title("Error de red de memoria")
# plt.savefig("Errors.jpg")
