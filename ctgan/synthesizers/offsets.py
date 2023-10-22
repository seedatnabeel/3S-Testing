# Adapted from https://github.com/flxai/soft-brownian-offset

import numpy as np

from sklearn.metrics import pairwise_distances
from tqdm import tqdm


def SBO(X, d_min, d_off, n_samples=1, show_progress=False, softness=False,
                         random_state=None):
    """Generates OOD samples using SBO on the input X and returns n_samples number of samples constrained by
    other parameters.
    Args:
        X (:obj:`numpy.array`): In-distribution (ID) data to form OOD samples around. First dimension contains samples
        d_min (float): (Likely) Minimum distance to ID data
        d_off (float): Offset distance used in each iteration
        n_samples(int): Number of samples to return
        show_progress(boolean): Whether to show a tqdm progress bar
        softness(float): Describes softness of minimum distance. Parameter between 0 (hard) and 1 (soft)
        random_state(int): RNG state used for reproducibility
    Returns:
        :obj:`numpy.array`:
            Out of distribution samples of shape (n_samples, X.shape[1])
    """
    if softness == 0:
        softness = False
    if random_state is not None:
        np.random.seed(random_state)
    n_dim = X.shape[1]
    ys = []
    iterator = range(n_samples)
    if show_progress:
        iterator = tqdm(iterator)
    for i in iterator:
        # Sample uniformly from X
        y = X[np.random.choice(len(X))].astype(float)
        # Move out of reach of other points
        skip = False
        while True:
            dist = pairwise_distances(y[:, None].T, X)[0]
            if dist.min() > 0:
                if not softness and dist.min() > d_min:
                    skip = True
                elif softness > 0:
                    p = 1 / (1 + np.exp((-dist.min() + d_min) / softness / d_min * 7))
                    if np.random.uniform() < p:
                        skip = True
                elif not isinstance(softness, bool):
                    raise ValueError("Softness should be float greater zero")
            if skip:
                break
            y += gaussian_hyperspheric_offset(1, n_dim=n_dim)[0] * d_off
        ys.append(np.array(y))
    return np.array(ys)


def GHO(X, mu=0, std=1, n_samples=1, show_progress=False, random_state=None):
    """Generates OOD samples using SBO on the input X and returns n_samples number of samples constrained by
    other parameters.
    Args:
        X (:obj:`numpy.array`): In-distribution (ID) data to form OOD samples around. First dimension contains samples
        d_min (float): (Likely) Minimum distance to ID data
        d_off (float): Offset distance used in each iteration
        n_samples(int): Number of samples to return
        show_progress(boolean): Whether to show a tqdm progress bar
        softness(float): Describes softness of minimum distance. Parameter between 0 (hard) and 1 (soft)
        random_state(int): RNG state used for reproducibility
    Returns:
        :obj:`numpy.array`:
            Out of distribution samples of shape (n_samples, X.shape[1])
    """
    if random_state is not None:
        np.random.seed(random_state)
    n_dim = X.shape[1]
    ys = []
    iterator = range(n_samples)
    if show_progress:
        iterator = tqdm(iterator)
    for i in iterator:
        # Sample uniformly from X
        y = X[np.random.choice(len(X))].astype(float)
        y += gaussian_hyperspheric_offset(1, mu=mu, std=std, n_dim=n_dim)[0]
        ys.append(np.array(y))
    return np.array(ys)

# Inspired by https://stackoverflow.com/a/33977530/10484131
def gaussian_hyperspheric_offset(n_samples, mu=4, std=.7, n_dim=3, random_state=None):
    """Generates OOD samples using GHO and returns n_samples number of samples constrained by other
    parameters.
    Args:
        n_samples(int): Number of samples to return
        mu (float): Mean of distribution
        std (float): Standard deviation of distribution
        n_dim (int): Number of dimensions
        random_state(int): RNG state used for reproducibility
    Returns:
        :obj:`numpy.array`:
            Out of distribution samples of shape (n_samples, n_dim)
    """
    if random_state is not None:
        np.random.seed(random_state)
    vec = np.random.randn(n_dim, n_samples)
    vec /= np.linalg.norm(vec, axis=0)
    vec *= np.random.normal(loc=mu, scale=std, size=n_samples)
    vec = vec.T
    return vec