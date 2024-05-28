import numpy as np
import gurobipy as gp
from gurobipy import GRB

from pyepo.model.grb.grbmodel import optGrbModel


class knapsackModel(optGrbModel):
    """
    This class is optimization model for knapsack problem

    Attributes:
        _model (GurobiPy model): Gurobi model
        weights (np.ndarray / list): Weights of items
        capacity (np.ndarray / listy): Total capacity
        items (list): List of item index
    """

    def __init__(self, weights, capacity, env=None):
        """
        Args:
            weights (np.ndarray / list): weights of items
            capacity (np.ndarray / list): total capacity
        """
        self.env = env
        self.weights = np.array(weights)
        self.capacity = np.array(capacity)
        self.items = list(range(self.weights.shape[1]))
        super().__init__()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        if self.env is not None:
            model = gp.Model("knapsack", env=self.env)
        else:
            model = gp.Model("knapsack")

        # varibles
        x = model.addVars(self.items, name="x", vtype=GRB.BINARY)
        # sense
        model.modelSense = GRB.MAXIMIZE
        # constraints
        for i in range(len(self.capacity)):
            model.addConstr(gp.quicksum(self.weights[i,j] * x[j]
                        for j in self.items) <= self.capacity[i])
        return model, x

    def relax(self):
        """
        A method to get linear relaxation model
        """
        # copy
        model_rel = knapsackModelRel(self.weights, self.capacity)
        return model_rel


class knapsackModelRel(knapsackModel):
    """
    This class is relaxed optimization model for knapsack problem.
    """

    def _getModel(self):
        """
        A method to build Gurobi
        """
        # ceate a model
        m = gp.Model("knapsack")
        # turn off output
        m.Params.outputFlag = 0
        # varibles
        x = m.addVars(self.items, name="x", ub=1)
        # sense
        m.modelSense = GRB.MAXIMIZE
        # constraints
        for i in range(len(self.capacity)):
            m.addConstr(gp.quicksum(self.weights[i,j] * x[j]
                        for j in self.items) <= self.capacity[i])
        return m, x

    def relax(self):
        """
        A forbidden method to relax MIP model
        """
        raise RuntimeError("Model has already been relaxed.")