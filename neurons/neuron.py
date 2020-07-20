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
        self.S = self.f(np.dot(self.X, self.W))

    def set_inputs(self, X):
        """Set the neuron inputs."""
        self.X = X

    def set_weigths(self, W):
        """Set the inputs weigths."""
        self.W = W

    def set_output(self, S):
        """Return the inputs outputs."""
        self.S = S
