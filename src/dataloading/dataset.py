import numpy as np
import pyepo
from sklearn.model_selection import train_test_split
from pyepo.model.grb import shortestPathModel, knapsackModel

def get_warcraft_maps():
    map_size = 12
    FOLDER = "data/warcraft_maps/{}x{}/".format(map_size, map_size)
    tmaps_train = np.load(FOLDER+"train_maps.npy")
    tmaps_test = np.load(FOLDER+"test_maps.npy")

    costs_train = np.load(FOLDER+"train_vertex_weights.npy")
    costs_test = np.load(FOLDER+"test_vertex_weights.npy")

    paths_train = np.load(FOLDER+"train_shortest_paths.npy")
    paths_test = np.load(FOLDER+"test_shortest_paths.npy")

    return (tmaps_train, tmaps_test, costs_train,
            costs_test, paths_train, paths_test)

def get_grid_dataset(grid, n, num_feat, deg, e):
    # generate data for grid network (features and costs)
    feats, costs = pyepo.data.shortestpath.genData(num_data=n+1000, num_features=num_feat, grid=grid, deg=deg, noise_width=e, seed=42)

    # split dataset into train and test datasets
    x_train, x_test, c_train, c_test = train_test_split(feats, costs, test_size=1000, random_state=42)

    # define solver
    optmodel = shortestPathModel(grid)

    # get optDataset
    dataset_train = pyepo.data.dataset.optDataset(optmodel, x_train, c_train)
    dataset_test = pyepo.data.dataset.optDataset(optmodel, x_test, c_test)

    return (dataset_train, dataset_test)

def get_knapsack_dataset(m = 16, n = 1000, p = 5, deg = 6, dim = 2, noise_width = 0.5):

    caps = [20] * dim # capacity of the knapsack
    weights, x, c = pyepo.data.knapsack.genData(n+1000, p, m, deg=deg, dim=dim, noise_width=noise_width)
    x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=1000, random_state=246)

    optmodel = knapsackModel(weights, caps)

    dataset_train = pyepo.data.dataset.optDataset(optmodel, x_train, c_train)
    dataset_test = pyepo.data.dataset.optDataset(optmodel, x_test, c_test)

    return (dataset_train, dataset_test, weights, caps)

