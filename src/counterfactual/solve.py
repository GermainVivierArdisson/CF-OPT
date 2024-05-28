import torch

def solve(cp, opt_model, task):
    """
    Solves an instance of the shortest path problem.
    Args:
        cp (PyTorch tensor): predicted costs.
        opt_model (optGrbModel): instance of the optGrbModel class used.
        task (str): defines the type of pipeline considered. 
              Can be either 'warcraft', 'grid' or 'knapsack'.
    """
    if task=="warcraft":
        grid = (12, 12)
        # Resize the costs
        cp = torch.nn.Unflatten(0, grid)(cp)

    # Set objective function
    opt_model.setObj(cp.detach().cpu().numpy())

    # Solve
    sol, value = opt_model.solve()
    sol = torch.FloatTensor(sol)

    return sol, value
