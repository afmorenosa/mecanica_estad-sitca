"""Neuronal network class and methods."""
import neurons.neuron as nr
import numpy as np


class network:
    """Neuronal network calss."""

    def __init__(self, m, n):
        """Network constructor."""
        aux_net = []
        self.m = m
        self.n = n
        for i in range(self.m):
            aux_net.append([])
            for j in range(self.n):
                aux_net[i].append(nr.neuron([], [], np.sign, -1))

        self.net = np.array(aux_net)
        self.E = 0
        self.pattern = []
        self.S_out = []

    def energy(self):
        """Compute the energy."""
        N = self.m * self.n

        E = 0

        for i in range(N):
            for j in range(i):
                E += self.net[int(i / self.m), i % self.m].W[j] *\
                     self.net[int(i / self.m), i % self.m].S *\
                     self.net[int(j / self.m), j % self.m].S
        self.E = -E

    def train(self, patterns):
        """Train the network."""
        P = len(patterns)

        N = self.m * self.n

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
                        aux += patterns[u][int(i / self.m)][i % self.m] *\
                               patterns[u][int(j / self.m)][j % self.m]

                    weigths[i].append(aux / N)

        for i in range(self.m):
            for j in range(self.n):
                self.net[i, j].set_weigths(weigths[i * self.m + j])

    def init_pattern(self):
        """Create a random pattern."""
        pattern = []

        for i in range(self.m):
            pattern.append([])
            for j in range(self.n):
                pattern[i].append(np.random.randint(2) * 2 - 1)

        self.pattern = pattern

    def config_init_system(self):
        """Configure the system into an initial condition."""
        self.init_pattern()

        for i in range(self.m):
            for j in range(self.n):
                self.net[i, j].set_output(self.pattern[i][j])

        self.S_out = self.pattern
        self.energy()

    def pattern_matrix(self):
        """Return the pattern."""
        S_matrix = []

        for i in range(self.m):
            S_matrix.append([])
            for j in range(self.n):
                S_matrix[i].append(self.net[i, j].S)

        self.S_out = S_matrix

    def system_step(self):
        """Make a step."""
        inputs = []

        for i in range(self.m):
            for j in range(self.n):
                inputs.append(self.net[i, j].S)

        for i in range(self.m):
            for j in range(self.n):
                self.net[i, j].set_inputs(inputs)
                self.net[i, j].activate()

        self.pattern_matrix()
        self.energy()
