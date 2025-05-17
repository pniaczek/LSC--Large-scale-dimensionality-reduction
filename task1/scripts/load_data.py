import numpy as np
import os
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset


def load_fashion_mnist(flatten: bool = True,
                       normalize: bool = True,
                       subset_size: int = None,
                       save: bool = False, data_path = "data/"):
    """
    Load and preprocess the full Fashion-MNIST dataset (train + test).

    Args:
        flatten (bool): If True, flattens images from 28x28 to 784D.
        normalize (bool): If True, normalizes pixel values to [0, 1].
        subset_size (int or None): If given, randomly samples the dataset.
        save (bool): If True, saves X.npy and y.npy in 'task1/data/'.

    Returns:
        X (np.ndarray): shape (N, 784), feature vectors.
        y (np.ndarray): shape (N,), labels.
    """

    transform = [transforms.ToTensor()]
    if flatten:
        transform.append(transforms.Lambda(lambda x: x.view(-1)))
    transform = transforms.Compose(transform)

    train = datasets.FashionMNIST(root=data_path, train=True, download=True, transform=transform)
    test = datasets.FashionMNIST(root=data_path, train=False, download=True, transform=transform)
    full_dataset = ConcatDataset([train, test])

    X = np.stack([d[0].numpy() for d in full_dataset])
    y = np.array([d[1] for d in full_dataset])

    if normalize:
        X = X / 1.0

    if subset_size is not None and subset_size < len(X):
        idx = np.random.choice(len(X), size=subset_size, replace=False)
        X = X[idx]
        y = y[idx]

    if save:
        os.makedirs(data_path, exist_ok=True)
        np.save(os.path.join(data_path, "X.npy"), X)
        np.save(os.path.join(data_path, "y.npy"), y)

    return X, y
