import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src.dataloading.dataset import get_warcraft_maps, get_grid_dataset, get_knapsack_dataset


class mapDataset(Dataset):
    def __init__(self, tmaps, costs, paths):
        self.tmaps = tmaps
        self.costs = costs
        self.paths = paths
        self.objs = (costs * paths).sum(axis=(1, 2)).reshape(-1, 1)

    def __len__(self):
        return len(self.costs)

    def __getitem__(self, ind):
        tmaps = self.tmaps[ind].transpose(2, 0, 1)/255  # image
        return (
            torch.FloatTensor(tmaps).detach(),
            torch.FloatTensor(self.costs[ind]).reshape(-1),
            torch.FloatTensor(self.paths[ind]).reshape(-1),
            torch.FloatTensor(self.objs[ind]),
        )


def get_dataloaders_and_dataset():
    """
    Returns dataloaders and datasets for the shortest paths on Warcraft maps pipeline.
    """
    # Read warcraft maps dataset as used by Vlastelica et al.
    (tmaps_train, tmaps_test, costs_train,
     costs_test, paths_train, paths_test) = get_warcraft_maps()

    # Create train and test data sets for CNN
    dataset_train_cnn = mapDataset(tmaps_train, costs_train, paths_train)
    dataset_test_cnn = mapDataset(tmaps_test, costs_test, paths_test)

    # Create train and test data sets for VAE
    dataset_train_vae = [[dataset_train_cnn[i][0], dataset_train_cnn[i][2]]
                         for i in range(len(dataset_train_cnn))]
    dataset_test_vae = [[dataset_test_cnn[i][0], dataset_test_cnn[i][2]]
                        for i in range(len(dataset_test_cnn))]

    # Wrap datasets in loaders
    loader_train_cnn = DataLoader(dataset_train_cnn,
                                  batch_size=70, shuffle=True)
    loader_test_cnn = DataLoader(dataset_test_cnn,
                                 batch_size=70, shuffle=True)
    loader_train_vae = DataLoader(dataset_train_vae,
                                  batch_size=32, shuffle=True)
    loader_test_vae = DataLoader(dataset_test_vae,
                                 batch_size=32, shuffle=True)

    return (dataset_train_cnn, dataset_test_cnn,
            loader_train_cnn, loader_test_cnn,
            loader_train_vae, loader_test_vae)

def get_grid_dataloaders_and_dataset(grid = (5,5), n = 1000, num_feat = 5, deg = 4, e = 0.5, batch_size = 32):
    """
    Returns dataloaders and datasets for the shortest paths on a grid pipeline.
    Args:
        grid (tuple of int): grid on which shortest paths are computed.
        n (int): size of the train set. Test set size is always 1000.
        num_feat (int): dimension of  the contextual features.
        deg (int): degree of the polynomial used to generate data. See PyEPO documentation for more details.
        e (float): amount of noise used to generate data. See PyEPO documentation for more details.
        batch_size (int): batch size of the returned dataloaders.
    """
    (dataset_train, dataset_test) = get_grid_dataset(grid, n, num_feat, deg, e)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    return (dataset_train, dataset_test, loader_train, loader_test)

def get_knapsack_dataloaders_and_dataset(m = 16, n = 1000, p = 5, deg = 6, dim = 2, noise_width = 0.5, batch_size = 32):
    """
    Returns dataloaders and datasets for the knapsack pipeline.
    Args:
        m (int): number of items.
        n (int): size of the train set. Test set size is always 1000.
        p (int): contextual dimension.
        deg (int): polynomial degree for data generation. See PyEPO documentation for more details.
        dim (int): knapsack dimension.
        noise_width (float): noise half-width for data generation. See PyEPO documentation for more details.
        batch_size (int): batchsize of the returned dataloaders.
    """
    (dataset_train, dataset_test, weights, caps) = get_knapsack_dataset(m, n, p, deg, dim, noise_width)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    return (dataset_train, dataset_test, loader_train, loader_test, weights, caps)
