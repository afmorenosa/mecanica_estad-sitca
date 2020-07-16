"""Neuron class and methods."""
import numpy as np


class neuron:
    """Neuron class."""

    def __init__(self, X, W, f, S):
        """Neuron constructor."""
        self.X = X
        self.W = W
        self.f = f
        self.S = S

    def activate(self):
        """Activation of a neuron."""
        self.S = np.sign(np.dot(self.X, self.W))

    def set_inputs(self, X):
        """Set the neuron inputs."""
        self.X = X

    def set_weigths(self, W):
        """Set the inputs weigths."""
        self.W = W

    def set_output(self, S):
        """Return the inputs weigths."""
        self.S = S


def energy(network):
    """Compute the energy."""
    II = network.shape[0]
    JJ = network.shape[1]
    N = II * JJ

    E = 0

    for i in range(N):
        for j in range(N):
            E += network[int(i / II), i % II].W[j] *\
                 network[int(i / II), i % II].S *\
                 network[int(j / II), j % II].S
    return -E


def train(network, patterns):
    """Train the network."""
    P = len(patterns)
    II = network.shape[0]
    JJ = network.shape[1]
    N = II * JJ
    weigths = []
    for i in range(N):
        weigths.append([])
        for j in range(N):
            if (i == j):
                weigths[i].append(0)
                continue
            else:
                aux = 0.0
                for u in range(P):
                    aux += patterns[u][int(i / II)][i % II] *\
                           patterns[u][int(j / II)][j % II]

                weigths[i].append(aux / N)

    for i in range(II):
        for j in range(JJ):
            network[i, j].set_weigths(weigths[i * II + j])


def init_pattern(network):
    """Create a random pattern."""
    II = network.shape[0]
    JJ = network.shape[1]
    pattern = []

    for i in range(II):
        pattern.append([])
        for j in range(JJ):
            pattern[i].append(np.random.randint(2) * 2 - 1)

    return pattern


def config_init_system(network):
    """Configure the system into an initial condition."""
    II = network.shape[0]
    JJ = network.shape[1]

    pattern = init_pattern(network)

    for i in range(II):
        for j in range(JJ):
            network[i, j].set_output(pattern[i][j])


def pattern_matrix(network):
    """Return the pattern."""
    II = network.shape[0]
    JJ = network.shape[1]

    S_matrix = []

    for i in range(II):
        S_matrix.append([])
        for j in range(JJ):
            S_matrix[i].append(network[i, j].S)

    return S_matrix


def system_step(network):
    """Make a step."""
    II = network.shape[0]
    JJ = network.shape[1]

    inputs = []

    for i in range(II):
        for j in range(JJ):
            inputs.append(network[i, j].S)

    for i in range(II):
        for j in range(JJ):
            network[i, j].set_inputs(inputs)
            network[i, j].activate()

    return energy(network)
