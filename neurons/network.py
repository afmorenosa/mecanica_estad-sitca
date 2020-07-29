"""Neuronal network class and methods."""
import matplotlib.pyplot as plt
import neurons.neuron as nr
import numpy as np


class network:
    """Neuronal network calss."""

    def __init__(self, m, n, f):
        """Network constructor."""
        aux_net = []
        self.m = m
        self.n = n
        self.N = self.m * self.n
        for i in range(self.m):
            aux_net.append([])
            for j in range(self.n):
                aux_net[i].append(nr.neuron([], [], f, -1))

        self.net = np.array(aux_net)

    def energy(self):
        """Compute the energy."""
        E = 0

        for i in range(self.N):
            for j in range(i):
                E += self.net[int(i / self.m), i % self.m].W[j] *\
                     self.net[int(i / self.m), i % self.m].S *\
                     self.net[int(j / self.m), j % self.m].S
        self.E = -E

    def train(self, patterns):
        """Train the network."""
        P = len(patterns)

        weigths = []
        for i in range(self.N):
            weigths.append([])
            for j in range(self.N):
                if (i == j):
                    weigths[i].append(0)
                    continue
                else:
                    aux = 0.0
                    for u in range(P):
                        aux += patterns[u][int(i / self.m)][i % self.m] *\
                               patterns[u][int(j / self.m)][j % self.m]

                    weigths[i].append(aux / self.N)

        for i in range(self.m):
            for j in range(self.n):
                self.net[i, j].set_weigths(weigths[i * self.m + j])

    def init_pattern(self, patterns):
        """Create a random pattern."""
        self.test_letter = np.random.randint(len(patterns))

        pattern = []

        for i in range(len(patterns[self.test_letter])):
            pattern.append([])
            for j in range(len(patterns[self.test_letter][i])):
                pattern[i].append(patterns[self.test_letter][i][j])
                # pattern[i].append(np.random.randint(2)*2 - 1)

        for i in range(5):
            i = np.random.randint(len(pattern))
            j = np.random.randint(len(pattern[0]))
            pattern[i][j] *= -1

        self.pattern = pattern

    def config_init_system(self, pattners):
        """Configure the system into an initial condition."""
        self.init_pattern(pattners)

        for i in range(self.m):
            for j in range(self.n):
                self.net[i, j].set_output(self.pattern[i][j])

        self.S_out = self.pattern
        self.energy()
        self.E0 = self.E

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

    def test(self, patterns):
        """Test network."""
        self.config_init_system(patterns)

        E = [0, 1, 2, 3]
        ite = 0

        while ((E[0] != E[1] or
               E[0] != E[2] or
               E[0] != E[3] or
               E[1] != E[2] or
               E[1] != E[3] or
               E[2] != E[3]) and
               ite <= 100):
            # matrix = open("results/data_{}.data".format(i), "w")
            print(self.E)
            self.system_step()
            E.remove(E[0])
            E.append(self.E)
            ite += 1

        print(self.E)
        # for pattern in patterns:
            # if ((np.array(self.S_out) == np.array(pattern)).all() or
               # (np.array(self.S_out) == -np.array(pattern)).all()):
                # return 0

        err_pix = 0
        for i in range(len(patterns[self.test_letter])):
            for j in range(len(patterns[self.test_letter][i])):
                if(patterns[self.test_letter][i][j] != self.S_out[i][j]):
                    err_pix += 1

        if (err_pix/self.N < 0.15):
            return 0

        return 1

    def get_error(self, patterns, n_test):
        """Get Network error."""
        self.train(patterns)

        self.mistakes = 0

        for i in range(n_test):
            result = self.test(patterns)
            self.mistakes += result
            if i == int(n_test / 2):
                plt.clf()
                plt.close()
                fig, (ax0, ax1) = plt.subplots(1, 2)
                ax0.matshow(self.pattern)
                ax0.set_title("Patron Inicial,\nE = {:.2f}\n{} patron(es) aprendido(s)".format(self.E0, len(patterns)))
                ax0.set_axis_off()
                ax1.matshow(self.S_out)
                ax1.set_title("Patron Final,\nE = {:.2f}".format(self.E))
                ax1.set_axis_off()
                # ax2.matshow(patterns[self.test_letter])
                # ax2.set_title("Patron Esperado\n{} patron(es) aprendido(s)".format(len(patterns)))
                # ax2.set_axis_off()
                plt.savefig("results/P_#{}_Patrones_{}".format(int(n_test / 2), len(patterns)))
            print("\t[{:.2f}%] Test con {} patron(es) aprendido(s), errores: {}".format((i+1)/n_test*100, len(patterns), self.mistakes))

        return self.mistakes / n_test * 100
