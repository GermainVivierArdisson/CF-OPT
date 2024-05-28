
import torch.nn as nn
import matplotlib.pyplot as plt

from src.counterfactual.solve import solve


def plot_maps(dataset_train_cnn, indices):
    """Plot some maps"""
    _, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, index in enumerate(indices):
        (warcraft_map, _, _, _) = dataset_train_cnn[index]
        map_array = warcraft_map.numpy().transpose((1, 2, 0))
        row, col = divmod(i, 5)
        axes[row, col].imshow(map_array)
        axes[row, col].axis('off')
        axes[row, col].set_title('Image '+str(index))
    plt.tight_layout()
    plt.show()


def plot_costs(dataset_train_cnn, indices, cnn):
    """Plot some predicted costs"""
    _, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, index in enumerate(indices):
        (warcraft_map, _, _, _) = dataset_train_cnn[index]
        predicted_costs = cnn(warcraft_map.unsqueeze(0)).squeeze().detach()
        predicted_costs = nn.Unflatten(0, (12, 12))(predicted_costs)
        costs_array = predicted_costs.numpy()
        row, col = divmod(i, 5)
        axes[row, col].imshow(costs_array)
        axes[row, col].axis('off')
        axes[row, col].set_title('Image '+str(index))
    plt.tight_layout()
    plt.show()


def plot_paths(dataset_train_cnn, indices, cnn):
    """Plot some shortest paths"""
    _, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, index in enumerate(indices):
        (warcraft_map, _, _, _) = dataset_train_cnn[index]
        predicted_costs = cnn(warcraft_map.unsqueeze(0))
        shortest_path = solve(predicted_costs.squeeze())
        shortest_path = nn.Unflatten(0, (12, 12))(shortest_path)
        path_array = shortest_path.numpy()
        row, col = divmod(i, 5)
        axes[row, col].imshow(path_array)
        axes[row, col].axis('off')
        axes[row, col].set_title('Image '+str(index))
    plt.tight_layout()
    plt.show()

