from . import utils
import numpy as np
from scipy import spatial
from abc import ABC, abstractmethod


class GravityPotentialBase(ABC):
    @abstractmethod
    def potential(self, x):
        pass

    @abstractmethod
    def flow(self, x):
        pass

    @abstractmethod
    def sample(self, num_samples):
        pass


class AngularGravitationalPotential(GravityPotentialBase):
    def __init__(self, mass, emb, scale):
        self.emb = emb
        self.mass = mass / np.sum(mass)
        self.scale = scale

    def potential(self, x):
        angle = utils.angle(x, self.emb)
        return -1 * np.sum(self.mass * np.exp(-self.scale * angle), axis=1)

    def flow(self, x):
        raise NotImplementedError

    def sample(self, num_samples):
        dim = self.emb.shape[1]

        # Draw random directions
        rand_d = np.random.randn(num_samples, dim)
        rand_d = np.einsum("ij,i->ij", rand_d, 1 / np.linalg.norm(rand_d, axis=1))

        # Select the masses
        centers = np.random.choice(len(self.mass), size=num_samples, p=self.mass)

        if self.scale is not None:
            raise ValueError(
                "scale parameter 'scale' needs to be specified for angular metric"
            )
        # Draw angles
        angles = utils.sample_from_exponential(
            scale=self.scale, size=num_samples, max_x=np.pi
        )
        # Calc a vector from the data to a random vector
        q = rand_d - self.emb[centers, :]
        q = np.einsum("ij,i->ij", q, 1 / np.linalg.norm(q, axis=1))
        # Putting them together
        x = self.emb[centers, :] + q * np.sin(angles)
        # Rescale
        return np.einsum("ij,i->ij", x, 1 / np.linalg.norm(x, axis=1))


class EuclideanGravitationalPotential(GravityPotentialBase):
    def __init__(self, mass, emb):
        self.emb = emb
        self.mass = mass / np.sum(mass)

    def potential(self, x):
        dist = spatial.distance.cdist(x, self.emb, "euclidean")
        weights = np.exp(-(dist))
        return -np.einsum("ij,j->i", weights, self.mass)

    def flow(self, x):
        raise NotImplementedError

    def sample(self, num_samples):
        dim = self.emb.shape[1]

        # Draw random directions
        rand_d = np.random.randn(num_samples, dim)
        rand_d = np.einsum("ij,i->ij", rand_d, 1 / np.linalg.norm(rand_d, axis=1))

        # Select the masses
        centers = np.random.choice(len(self.mass), size=num_samples, p=self.mass)

        # Draw distance from the masses
        r = np.random.exponential(size=num_samples)
        return self.emb[centers, :] + np.einsum("i,ij->ij", r, rand_d)
