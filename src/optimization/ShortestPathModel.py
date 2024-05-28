import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pyepo.model.grb.grbmodel import optGrbModel

class ShortestPathModel(optGrbModel):
    """
    This class is optimization model for shortest path problem on a
    rectangle grid.

    Attributes:
        _model (GurobiPy model): Gurobi model
        grid (tuple of int): Size of grid network
        nodes (list): list of vertex
        edges (list): List of arcs
        nodes_map (ndarray): 2D array for node index
    """

    def __init__(self, grid, env=None, task="warcraft"):
        """
        Args:
            grid (tuple of int): size of grid network
        """
        assert len(grid) == 2
        assert task in ["warcraft", "grid"]
        self.env = env
        self.task = task
        self.grid = grid
        if task == "warcraft":
            self.nodes, self.nodes_map = self._get_nodes_from_grid_dimensions(grid)
            self.edges = self._get_edges_from_grid_dimensions(grid)
        elif task=="grid":
            self.arcs = self._getArcs()
        super().__init__()


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def _get_nodes_from_grid_dimensions(self, grid):
        """
        Returns the list of nodes in the input
        grid.

        Returns:
            list: nodes
            dict: associate a node index to its coordinates
        """
        nodes = []
        nodes_map = dict()
        for i in range(grid[0]):
            for j in range(grid[1]):
                u = self._node_coordinates_to_index(i, j)
                nodes_map[u] = (i, j)
                nodes.append(u)
        return nodes, nodes_map

    def _get_edges_from_grid_dimensions(self, grid):
        """
        A method to get list of edges for grid network.

        Loop over all (i,j) coordinates of the grid and
        check if its 8 neighbors are in the grid. If they
        are, add the edge to the list.

        Returns:
            list: arcs
        """
        edges = []
        for i in range(grid[0]):
            for j in range(grid[1]):
                u = self._node_coordinates_to_index(i, j)
                # An edge can have up to 8 neighbors in 2D
                for m in [-1, 0, 1]:
                    for n in [-1, 0, 1]:
                        if not ((m == 0) and (n == 0)):
                            if self._is_node_in_grid(i+m, j+n):
                                v = self._node_coordinates_to_index(i+m, j+n)
                                edges.append((u, v))
        return edges

    def _is_node_in_grid(self, i, j):
        """
        Check whether a node is within the rectangular
        grid.

        Args:
            i (_type_): _description_
            j (_type_): _description_
        """
        if ((i >= 0) and (j >= 0) and
           (i <= self.grid[0] - 1) and
           (j <= self.grid[1] - 1)):
            return True
        else:
            return False

    def _node_coordinates_to_index(self, i, j):
        """
        Return the index of a node based on its
        cartesian coordinates.

        Args:
            i (int): x coordinate
            j (int): y coordinate
        """
        assert i >= 0
        assert j >= 0
        assert i <= self.grid[0] - 1
        assert j <= self.grid[1] - 1
        v = i * self.grid[1] + j
        return v

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """

        if self.task == "warcraft":
            if self.env is not None:
                model = gp.Model(env=self.env)
            else:
                model = gp.Model()
            # Variables
            x = model.addVars(self.edges, ub=1, name="x")
            # sense
            model.modelSense = GRB.MINIMIZE
            # Constraints
            for i in range(self.grid[0]):
                for j in range(self.grid[1]):
                    v = self._node_coordinates_to_index(i, j)
                    expr = 0
                    for e in self.edges:
                        # flow in
                        if v == e[1]:
                            expr += x[e]
                        # flow out
                        elif v == e[0]:
                            expr -= x[e]
                    # source
                    if i == 0 and j == 0:
                        model.addConstr(expr == -1)
                    # sink
                    elif i == self.grid[0] - 1 and j == self.grid[0] - 1:
                        model.addConstr(expr == 1)
                    # transition
                    else:
                        model.addConstr(expr == 0)
        
        elif self.task=="grid":
            if self.env is not None:
                model = gp.Model("shortest path", env=self.env)
            else:
                model = gp.Model("shortest path")
            # varibles
            x = model.addVars(self.arcs, name="x")
            # sense
            model.modelSense = GRB.MINIMIZE
            # constraints
            for i in range(self.grid[0]):
                for j in range(self.grid[1]):
                    v = i * self.grid[1] + j
                    expr = 0
                    for e in self.arcs:
                        # flow in
                        if v == e[1]:
                            expr += x[e]
                        # flow out
                        elif v == e[0]:
                            expr -= x[e]
                    # source
                    if i == 0 and j == 0:
                        model.addConstr(expr == -1)
                    # sink
                    elif i == self.grid[0] - 1 and j == self.grid[0] - 1:
                        model.addConstr(expr == 1)
                    # transition
                    else:
                        model.addConstr(expr == 0)

        return model, x

    def setObj(self, c):
        """
        A method to set objective function

        Args:
            c (np.ndarray): cost of objective function
        """
        if self.task == "warcraft":
            # vector to matrix
            c = c.reshape(self.grid)
            # sum up vector cost
            obj = c[0, 0] + gp.quicksum(c[self.nodes_map[j]] * self.x[i, j]
                                        for i, j in self.x)
            self._model.setObjective(obj)
        
        elif self.task=="grid":
            super().setObj(c)

    def solve(self):
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        if self.task == "warcraft":
            # update gurobi model
            self._model.update()
            # solve
            self._model.optimize()
            # kxk solution map
            sol = np.zeros(self.grid)
            for i, j in self.edges:
                # active edge
                if abs(1 - self.x[i, j].x) < 1e-3:
                    # node on active edge
                    sol[self.nodes_map[i]] = 1
                    sol[self.nodes_map[j]] = 1
            # matrix to vector
            sol = sol.reshape(-1)
            return sol, self._model.objVal
        
        elif self.task == "grid":
            return super().solve()        
    
    def _getArcs(self):
        """
        A helper method to get list of arcs for grid network

        Returns:
            list: arcs
        """
        if self.task == "warcraft":
            return
        else:
            arcs = []
            for i in range(self.grid[0]):
                # edges on rows
                for j in range(self.grid[1] - 1):
                    v = i * self.grid[1] + j
                    arcs.append((v, v + 1))
                # edges in columns
                if i == self.grid[0] - 1:
                    continue
                for j in range(self.grid[1]):
                    v = i * self.grid[1] + j
                    arcs.append((v, v + self.grid[1]))
            return arcs