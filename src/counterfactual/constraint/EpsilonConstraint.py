import torch
import numpy as np
from src.counterfactual.solve import solve


class EpsilonConstraint(torch.autograd.Function):
    """
    Computes the feasibility of the explanation constraint for epsilon-explanations.
    """
    @staticmethod
    def forward(ctx, cp, w_init, epsilon, opt_model, task):

        ctx.task = task

        w_opt, value = solve(cp, opt_model, task)   

        ctx.save_for_backward(w_opt, w_init)
        ctx.eps = epsilon
        sign = np.sign(value)
        ctx.sign = sign

        if task == "knapsack":
            return (sign*epsilon - 1)*value + torch.dot(cp, w_init) # Opposite constraint as the optimization problem is formalized as a Maximization problem.

        return (1 + sign*epsilon)*value - torch.dot(cp, w_init) 
    
    @staticmethod
    def backward(ctx, grad_output):
        w_opt, w_init = ctx.saved_tensors
        epsilon = ctx.eps
        sign = ctx.sign
        if ctx.task == "knapsack":
            return grad_output*( (sign*epsilon - 1)*w_opt + w_init), None, None, None, None # Opposite gradient as the optimization problem is formalized as a Maximization problem.
        
        return grad_output*( (1 + sign*epsilon)*w_opt - w_init), None, None, None, None
